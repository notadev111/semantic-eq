"""
Train Audio Encoder for Adaptive Semantic EQ
============================================

Trains the Audio Encoder to map audio into the same latent space as the EQ Encoder.

Training Strategy:
    1. Load pre-trained V2 model (EQ encoder/decoder)
    2. Freeze EQ encoder weights
    3. Synthesize training data (pink noise + EQ)
    4. Train Audio Encoder with contrastive loss:
       Loss = MSE(z_audio, z_eq) + semantic_contrastive_loss

Usage:
    python train_audio_encoder.py --epochs 100 --batch-size 32
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from core.neural_eq_morphing_safedb_v2 import NeuralEQMorphingSAFEDBV2
from core.audio_encoder import FastAudioEncoder, AudioEncoderConfig
from core.training_data_synthesis import TrainingDataSynthesizer, SynthesizedAudioDataset

# W&B for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss to align audio embeddings with EQ embeddings in latent space

    Two components:
    1. Latent space MSE: z_audio should match z_eq
    2. Semantic contrastive: Same semantic term → close, different → far
    """

    def __init__(self, temperature: float = 0.07, margin: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self,
                z_audio: torch.Tensor,
                z_eq: torch.Tensor,
                semantic_emb_audio: torch.Tensor,
                semantic_emb_eq: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_audio: [batch, 32] audio latent vectors
            z_eq: [batch, 32] EQ latent vectors
            semantic_emb_audio: [batch, 128] audio semantic embeddings
            semantic_emb_eq: [batch, 128] EQ semantic embeddings
            labels: [batch] semantic term indices

        Returns:
            loss: scalar
        """
        batch_size = z_audio.shape[0]

        # Component 1: Latent space alignment
        latent_loss = F.mse_loss(z_audio, z_eq)

        # Component 2: Semantic contrastive loss
        # Normalize embeddings
        semantic_emb_audio = F.normalize(semantic_emb_audio, dim=1)
        semantic_emb_eq = F.normalize(semantic_emb_eq, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(semantic_emb_audio, semantic_emb_eq.T) / self.temperature

        # Create positive/negative mask
        labels_eq = labels.unsqueeze(1)
        labels_audio = labels.unsqueeze(0)
        positive_mask = (labels_eq == labels_audio).float()

        # InfoNCE loss
        exp_sim = torch.exp(similarity)
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Average over positives
        contrastive_loss = -(positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
        contrastive_loss = contrastive_loss.mean()

        # Total loss
        total_loss = latent_loss + 0.5 * contrastive_loss

        return total_loss, latent_loss, contrastive_loss


class AudioEncoderTrainer:
    """
    Trainer for Audio Encoder
    """

    def __init__(self,
                 v2_model_path: str = "neural_eq_safedb_v2.pt",
                 audio_encoder_config: dict = None,
                 device: str = None,
                 use_wandb: bool = True):
        """
        Args:
            v2_model_path: Path to pre-trained V2 model
            audio_encoder_config: Configuration for Audio Encoder
            device: Device to train on (auto-detected if None)
            use_wandb: Whether to use Weights & Biases for logging
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        # Load pre-trained V2 model
        print(f"\nLoading pre-trained V2 model: {v2_model_path}")
        self.v2_system = NeuralEQMorphingSAFEDBV2()
        self.v2_system.load_model(v2_model_path)
        self.v2_system.load_dataset(min_examples=5)

        # Freeze EQ encoder (we're not re-training it)
        for param in self.v2_system.encoder.parameters():
            param.requires_grad = False
        self.v2_system.encoder.eval()

        print(f"  Loaded {len(self.v2_system.eq_settings)} EQ settings")
        print(f"  Semantic terms: {len(self.v2_system.term_to_idx)}")

        # Create Audio Encoder
        if audio_encoder_config is None:
            audio_encoder_config = AudioEncoderConfig.STANDARD

        self.audio_encoder = FastAudioEncoder(**audio_encoder_config).to(self.device)
        print(f"\nAudio Encoder parameters: {self.audio_encoder.get_num_parameters():,}")

        # Loss and optimizer
        self.criterion = ContrastiveLoss()
        self.optimizer = torch.optim.AdamW(
            self.audio_encoder.parameters(),
            lr=1e-3,
            weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100,
            eta_min=1e-5
        )

        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def synthesize_dataset(self, n_augmentations: int = 3) -> SynthesizedAudioDataset:
        """
        Create synthesized training dataset

        Args:
            n_augmentations: Number of audio variations per EQ setting

        Returns:
            dataset: SynthesizedAudioDataset
        """
        print("\n" + "="*70)
        print("SYNTHESIZING TRAINING DATA")
        print("="*70)

        synthesizer = TrainingDataSynthesizer(
            sample_rate=44100,
            audio_duration=2.0
        )

        data_list = synthesizer.create_dataset(
            self.v2_system.eq_settings,
            n_augmentations=n_augmentations
        )

        dataset = SynthesizedAudioDataset(data_list)

        print(f"\nDataset created: {len(dataset)} examples")

        return dataset

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train for one epoch"""

        self.audio_encoder.train()
        epoch_loss = 0.0
        epoch_latent_loss = 0.0
        epoch_contrastive_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, (audio, eq_params_norm, semantic_labels) in enumerate(pbar):
            # Move to device
            audio = audio.to(self.device)  # [batch, 1, n_samples]
            eq_params_norm = eq_params_norm.to(self.device)  # [batch, 13]

            # Convert semantic labels to indices
            label_indices = torch.LongTensor([
                self.v2_system.term_to_idx[label] for label in semantic_labels
            ]).to(self.device)

            # Forward pass - Audio Encoder
            z_audio = self.audio_encoder(audio)  # [batch, 32]

            # Forward pass - EQ Encoder (frozen)
            with torch.no_grad():
                z_eq, semantic_emb_eq = self.v2_system.encoder(eq_params_norm)

            # Audio Encoder semantic projection (need to add this!)
            # For now, use the EQ encoder's projection on audio latents
            semantic_emb_audio = self.v2_system.encoder.semantic_proj(z_audio)

            # Compute loss
            loss, latent_loss, contrastive_loss = self.criterion(
                z_audio, z_eq,
                semantic_emb_audio, semantic_emb_eq,
                label_indices
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.audio_encoder.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_latent_loss += latent_loss.item()
            epoch_contrastive_loss += contrastive_loss.item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'latent': f'{latent_loss.item():.4f}',
                'contr': f'{contrastive_loss.item():.4f}'
            })

            # Log to W&B (per batch)
            if self.use_wandb:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/batch_latent_loss': latent_loss.item(),
                    'train/batch_contrastive_loss': contrastive_loss.item(),
                    'train/step': epoch * len(dataloader) + batch_idx
                })

        # Average losses
        epoch_loss /= len(dataloader)
        epoch_latent_loss /= len(dataloader)
        epoch_contrastive_loss /= len(dataloader)

        print(f"\n  Train Loss: {epoch_loss:.4f} (Latent: {epoch_latent_loss:.4f}, Contrastive: {epoch_contrastive_loss:.4f})")

        # Log epoch metrics to W&B
        if self.use_wandb:
            wandb.log({
                'train/epoch_loss': epoch_loss,
                'train/epoch_latent_loss': epoch_latent_loss,
                'train/epoch_contrastive_loss': epoch_contrastive_loss,
                'epoch': epoch
            })

        return epoch_loss

    def validate(self, dataloader: DataLoader) -> float:
        """Validate on validation set"""

        self.audio_encoder.eval()
        val_loss = 0.0
        val_latent_loss = 0.0
        val_contrastive_loss = 0.0

        with torch.no_grad():
            for audio, eq_params_norm, semantic_labels in dataloader:
                audio = audio.to(self.device)
                eq_params_norm = eq_params_norm.to(self.device)

                label_indices = torch.LongTensor([
                    self.v2_system.term_to_idx[label] for label in semantic_labels
                ]).to(self.device)

                # Forward pass
                z_audio = self.audio_encoder(audio)
                z_eq, semantic_emb_eq = self.v2_system.encoder(eq_params_norm)
                semantic_emb_audio = self.v2_system.encoder.semantic_proj(z_audio)

                # Compute loss
                loss, latent_loss, contrastive_loss = self.criterion(
                    z_audio, z_eq,
                    semantic_emb_audio, semantic_emb_eq,
                    label_indices
                )

                val_loss += loss.item()
                val_latent_loss += latent_loss.item()
                val_contrastive_loss += contrastive_loss.item()

        # Average
        val_loss /= len(dataloader)
        val_latent_loss /= len(dataloader)
        val_contrastive_loss /= len(dataloader)

        print(f"  Val Loss: {val_loss:.4f} (Latent: {val_latent_loss:.4f}, Contrastive: {val_contrastive_loss:.4f})")

        # Log to W&B
        if self.use_wandb:
            wandb.log({
                'val/loss': val_loss,
                'val/latent_loss': val_latent_loss,
                'val/contrastive_loss': val_contrastive_loss
            })

        return val_loss

    def train(self, epochs: int, batch_size: int, n_augmentations: int = 3):
        """
        Full training loop

        Args:
            epochs: Number of epochs
            batch_size: Batch size
            n_augmentations: Audio variations per EQ setting
        """
        print("\n" + "="*70)
        print("TRAINING AUDIO ENCODER FOR ADAPTIVE SEMANTIC EQ")
        print("="*70)

        # Create dataset
        dataset = self.synthesize_dataset(n_augmentations=n_augmentations)

        # Split into train/val
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        print(f"\nDataset split:")
        print(f"  Train: {len(train_dataset)}")
        print(f"  Val: {len(val_dataset)}")

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Windows compatibility
            pin_memory=True if self.device == 'cuda' else False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True if self.device == 'cuda' else False
        )

        print(f"\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)

        # Initialize W&B
        if self.use_wandb:
            wandb.init(
                entity="triplenegative",  # Your W&B username/team
                project="semantic-eq-audio-encoder",
                config={
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'n_augmentations': n_augmentations,
                    'learning_rate': 1e-3,
                    'device': str(self.device),
                    'dataset_size': len(dataset),
                    'train_size': len(train_dataset),
                    'val_size': len(val_dataset),
                    'model_params': self.audio_encoder.get_num_parameters()
                }
            )
            wandb.watch(self.audio_encoder, log='all', log_freq=100)
            print(f"  W&B logging enabled: {wandb.run.name}")

        # Training loop
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            print("-" * 70)

            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)

            # Step scheduler
            self.scheduler.step()

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('audio_encoder_best.pt')
                print(f"  [BEST] Best model saved (val_loss: {val_loss:.4f})")

                # Log best model to W&B
                if self.use_wandb:
                    wandb.run.summary['best_val_loss'] = val_loss
                    wandb.run.summary['best_epoch'] = epoch

            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self.save_checkpoint(f'audio_encoder_epoch_{epoch}.pt')

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"\nBest validation loss: {self.best_val_loss:.4f}")
        print(f"Final model saved to: audio_encoder_best.pt")

        # Finish W&B run
        if self.use_wandb:
            wandb.finish()
            print(f"\nW&B run completed: {wandb.run.url}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'audio_encoder_state_dict': self.audio_encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.audio_encoder.load_state_dict(checkpoint['audio_encoder_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Train Audio Encoder for Adaptive EQ")

    parser.add_argument('--v2-model', default='neural_eq_safedb_v2.pt',
                       help='Path to pre-trained V2 model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--n-augmentations', type=int, default=3,
                       help='Audio variations per EQ setting')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None,
                       help='Device to train on (auto-detect if not specified)')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Weights & Biases logging')

    args = parser.parse_args()

    # Check if V2 model exists
    if not Path(args.v2_model).exists():
        print(f"\nERROR: V2 model not found: {args.v2_model}")
        print("Please train the V2 model first:")
        print("  python train_neural_eq_v2.py")
        return

    # Create trainer
    trainer = AudioEncoderTrainer(
        v2_model_path=args.v2_model,
        audio_encoder_config=AudioEncoderConfig.STANDARD,
        device=args.device,
        use_wandb=not args.no_wandb
    )

    # Train
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_augmentations=args.n_augmentations
    )


if __name__ == "__main__":
    main()
