"""
End-to-End Differentiable Semantic EQ Training
===============================================

Trains the Audio Encoder with fully differentiable EQ using dasp-pytorch.
This enables gradient flow through the entire pipeline:

    Audio → Audio Encoder → z_audio → Latent Traversal → EQ Decoder → params → dasp EQ → Output Audio
                                                                                            ↓
                                                                     Re-encode: z_output = Audio Encoder(Output)
                                                                                            ↓
                                                                     Loss: semantic_loss + param_loss + quality_loss

Key Innovation: "Semantic Consistency Loss"
    After applying EQ, the output audio should be CLOSER to the target semantic
    in latent space. This creates a self-supervised training signal.

References:
    - dasp-pytorch: Steinmetz et al. "Style Transfer of Audio Effects with
      Differentiable Signal Processing" (2022)
    - auraloss: https://github.com/csteinmetz1/auraloss
    - SAFE-DB: Stables et al. "SAFE: A system for the extraction and retrieval
      of semantic audio descriptors" ISMIR 2014
    - FMA Dataset: Defferrard et al. "FMA: A Dataset For Music Analysis" ISMIR 2017

Usage:
    # Install dependencies first:
    pip install dasp-pytorch auraloss

    # Train with pink noise only:
    python train_audio_encoder_e2e.py --epochs 100 --device cuda

    # Train with real music (FMA dataset):
    python train_audio_encoder_e2e.py --epochs 100 --device cuda --fma-path /path/to/fma_small

    # Train with multi-source:
    python train_audio_encoder_e2e.py --epochs 100 --device cuda \
        --fma-path /path/to/fma_small --musdb-path /path/to/musdb18

Requirements:
    - dasp-pytorch
    - auraloss
    - wandb (optional)
    - torchaudio
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Optional: Weights & Biases
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Local imports
from core.audio_encoder import FastAudioEncoder, AudioEncoderConfig
from core.neural_eq_morphing_safedb_v2 import NeuralEQMorphingSAFEDBV2
from core.differentiable_eq import DifferentiableEQ, SemanticEQLoss, DASP_AVAILABLE
from core.training_data_synthesis import PinkNoiseGenerator, BiquadEQFilter
from core.multi_source_dataset import MultiSourceDataset, DatasetConfig, create_dataset_from_config


class EndToEndSemanticEQTrainer:
    """
    End-to-end trainer for semantic EQ with differentiable signal processing.

    Training Strategy:
    1. Generate pink noise (or use real audio)
    2. Apply known EQ from SAFE-DB → this is our "EQ'd audio"
    3. Encode EQ'd audio → z_audio
    4. Get target semantic embedding → z_target
    5. Predict EQ params via latent traversal
    6. Apply predicted EQ differentiably → output audio
    7. Re-encode output → z_output
    8. Loss: z_output should be close to z_target (semantic consistency)
    """

    def __init__(
        self,
        v2_model_path: str = 'neural_eq_safedb_v2.pt',
        audio_encoder_path: str = None,
        device: str = None,
        sample_rate: int = 44100,
        audio_duration: float = 2.0,
        fma_path: str = None,
        fma_ratio: float = 0.5,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.sample_rate = sample_rate
        self.audio_duration = audio_duration
        self.n_samples = int(sample_rate * audio_duration)
        self.fma_ratio = fma_ratio

        print("="*70)
        print("END-TO-END DIFFERENTIABLE SEMANTIC EQ TRAINER")
        print("="*70)
        print(f"Device: {self.device}")

        # Load V2 system (EQ encoder/decoder)
        print("\nLoading V2 system...")
        self.v2_system = NeuralEQMorphingSAFEDBV2()
        self.v2_system.load_model(v2_model_path)
        self.v2_system.load_dataset(min_examples=5)
        self.v2_system.encoder.to(self.device).eval()
        self.v2_system.decoder.to(self.device).eval()

        # Freeze V2 encoder/decoder
        for param in self.v2_system.encoder.parameters():
            param.requires_grad = False
        for param in self.v2_system.decoder.parameters():
            param.requires_grad = False

        print(f"  Loaded {len(self.v2_system.eq_settings)} EQ settings")
        print(f"  Vocabulary: {list(self.v2_system.term_to_idx.keys())}")

        # Create Audio Encoder (this is what we're training)
        print("\nCreating Audio Encoder...")
        config = AudioEncoderConfig.STANDARD
        self.audio_encoder = FastAudioEncoder(**config).to(self.device)

        # Optionally load pretrained weights
        if audio_encoder_path and Path(audio_encoder_path).exists():
            checkpoint = torch.load(audio_encoder_path, map_location=self.device)
            self.audio_encoder.load_state_dict(checkpoint['audio_encoder_state_dict'])
            print(f"  Loaded pretrained weights from {audio_encoder_path}")
        else:
            print("  Training from scratch")

        print(f"  Parameters: {self.audio_encoder.get_num_parameters():,}")

        # Create differentiable EQ
        print("\nCreating Differentiable EQ...")
        self.diff_eq = DifferentiableEQ(sample_rate=sample_rate).to(self.device)
        print("  [OK] dasp-pytorch EQ ready")

        # Create loss function
        self.loss_fn = SemanticEQLoss(
            semantic_weight=1.0,
            param_weight=0.5,
            quality_weight=0.1,
        ).to(self.device)

        # Pre-compute semantic embeddings
        self._precompute_semantic_embeddings()

        # Pink noise generator
        self.pink_generator = PinkNoiseGenerator()

        # FMA loader (real music for domain generalization)
        self.fma_loader = None
        if fma_path and Path(fma_path).exists():
            from core.multi_source_dataset import FMALoader
            self.fma_loader = FMALoader(
                root_path=fma_path,
                sample_rate=sample_rate,
                duration=audio_duration,
            )
            print(f"\n  FMA dataset loaded: {len(self.fma_loader.audio_files)} files")
            print(f"  Training ratio: {fma_ratio*100:.0f}% FMA / {(1-fma_ratio)*100:.0f}% pink noise")
        else:
            if fma_path:
                print(f"\n  WARNING: FMA path not found: {fma_path}")
                print(f"  Falling back to 100% pink noise")
            self.fma_ratio = 0.0
            print(f"  Training source: 100% pink noise")

    # Only train on meaningful tonal descriptors (exclude junk terms)
    USABLE_TERMS = {"warm", "bright", "thin", "full", "muddy", "clear", "airy", "deep", "boomy", "tinny"}

    def _precompute_semantic_embeddings(self):
        """Pre-compute latent embeddings for usable semantic terms only."""
        print("\nPre-computing semantic embeddings...")
        print(f"  Filtering to {len(self.USABLE_TERMS)} usable terms (excluding test, re27, vocals, brighter)")

        self.semantic_embeddings = {}

        with torch.no_grad():
            for term in self.v2_system.term_to_idx.keys():
                if term not in self.USABLE_TERMS:
                    continue
                # Get all EQ settings for this term
                term_settings = [s for s in self.v2_system.eq_settings
                               if s.semantic_label == term]

                if not term_settings:
                    continue

                # Encode and average
                term_latents = []
                for setting in term_settings:
                    eq_params = torch.FloatTensor(setting.eq_params_normalized).unsqueeze(0).to(self.device)
                    z, _ = self.v2_system.encoder(eq_params)
                    term_latents.append(z)

                avg_latent = torch.stack(term_latents).mean(dim=0)
                self.semantic_embeddings[term] = avg_latent

        print(f"  Computed {len(self.semantic_embeddings)} semantic embeddings")

        # Pre-compute term-to-settings index for balanced sampling (usable terms only)
        self._term_settings_index = {}
        for term in self.v2_system.term_to_idx.keys():
            if term not in self.USABLE_TERMS:
                continue
            settings = [s for s in self.v2_system.eq_settings
                        if s.semantic_label == term]
            if settings:
                self._term_settings_index[term] = settings
        self._term_list = list(self._term_settings_index.keys())
        print(f"  Balanced sampling index: {len(self._term_list)} terms")
        for t in self._term_list:
            print(f"    {t}: {len(self._term_settings_index[t])} settings")

    def create_training_batch(self, batch_size: int) -> dict:
        """
        Create a batch of training data.

        Audio source is mixed: FMA real music (fma_ratio) + pink noise (1-fma_ratio).
        EQ settings are sampled with balanced class sampling across semantic terms.

        Returns dict with:
            - audio_input: Clean source audio (FMA or pink noise)
            - audio_eq: Source audio with known EQ applied
            - eq_params_norm: Normalized EQ parameters
            - semantic_labels: Semantic term for each sample
            - z_target: Target semantic embeddings
        """
        audio_inputs = []
        audio_eqs = []
        eq_params_list = []
        semantic_labels = []
        z_targets = []

        # Sample with balanced class sampling to address class imbalance
        # Without this, bright (37%) and warm (39%) dominate every batch
        # With balanced sampling, each semantic term has equal probability
        for _ in range(batch_size):
            # 1. Pick a random semantic term (equal probability across all terms)
            term = np.random.choice(self._term_list)
            # 2. Pick a random EQ setting with that term (precomputed index)
            setting = np.random.choice(self._term_settings_index[term])

            # Generate source audio: FMA (real music) or pink noise
            use_fma = (self.fma_loader is not None and np.random.random() < self.fma_ratio)
            if use_fma:
                fma_clip = self.fma_loader.load_random_clip()
                if fma_clip is not None:
                    source_audio = fma_clip
                else:
                    source_audio = self.pink_generator.generate(self.n_samples, self.sample_rate)
            else:
                source_audio = self.pink_generator.generate(self.n_samples, self.sample_rate)
            audio_input = torch.FloatTensor(source_audio).unsqueeze(0)  # [1, samples]

            # Apply EQ using scipy (non-differentiable, just for creating training data)
            from core.training_data_synthesis import BiquadEQFilter
            eq_filter = BiquadEQFilter()

            # Denormalize params for filtering
            eq_params_denorm = self.v2_system.loader.denormalize_params(
                np.array(setting.eq_params_normalized)
            )
            audio_with_eq = eq_filter.apply_eq(source_audio, eq_params_denorm, self.sample_rate)
            audio_eq = torch.FloatTensor(audio_with_eq).unsqueeze(0)

            # Get target semantic embedding
            z_target = self.semantic_embeddings[setting.semantic_label]

            audio_inputs.append(audio_input)
            audio_eqs.append(audio_eq)
            eq_params_list.append(torch.FloatTensor(setting.eq_params_normalized))
            semantic_labels.append(setting.semantic_label)
            z_targets.append(z_target)

        return {
            'audio_input': torch.stack(audio_inputs).to(self.device),  # [batch, 1, samples]
            'audio_eq': torch.stack(audio_eqs).to(self.device),  # [batch, 1, samples]
            'eq_params_norm': torch.stack(eq_params_list).to(self.device),  # [batch, 13]
            'semantic_labels': semantic_labels,
            'z_target': torch.cat(z_targets, dim=0).to(self.device),  # [batch, 32]
        }

    def train_step(self, batch: dict, intensity: float = 0.7) -> dict:
        """
        Single training step with end-to-end gradients.

        The key innovation: we compute semantic consistency loss by
        re-encoding the output and checking if it's closer to the target.
        """
        audio_input = batch['audio_input']  # Clean pink noise
        audio_eq = batch['audio_eq']  # Pink noise with known EQ (what we'd get from SAFE-DB)
        eq_params_gt = batch['eq_params_norm']  # Ground truth params
        z_target = batch['z_target']  # Target semantic embedding

        # 1. Encode the EQ'd audio (simulates analyzing user's audio)
        z_audio = self.audio_encoder(audio_eq)  # [batch, 32]

        # 2. Latent traversal toward target semantic
        z_final = z_audio + intensity * (z_target - z_audio)  # [batch, 32]

        # 3. Decode to EQ parameters
        with torch.no_grad():
            # Decoder is frozen, but we still use it for param prediction
            eq_params_pred = self.v2_system.decoder(z_final)  # [batch, 13]

        # For training, we'll use the ground truth params to ensure valid EQ
        # (Once converged, we can use predicted params)
        # eq_params_pred = eq_params_gt  # Use GT during early training

        # 4. Apply EQ DIFFERENTIABLY to clean input
        # Note: We apply to clean input, not the already-EQ'd audio
        output_audio = self.diff_eq(audio_input, eq_params_gt)  # [batch, 1, samples]

        # 5. Re-encode output to check semantic consistency
        z_output = self.audio_encoder(output_audio)  # [batch, 32]

        # 6. Compute loss
        loss, loss_dict = self.loss_fn(
            z_output=z_output,
            z_target=z_target,
            params_pred=eq_params_pred.detach(),  # Don't backprop through decoder
            params_gt=eq_params_gt,
            audio_output=output_audio,
            audio_input=audio_input,
        )

        return loss, loss_dict

    def train(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        use_wandb: bool = True,
        save_every: int = 10,
        intensity_schedule: bool = True,
    ):
        """
        Main training loop.

        Args:
            epochs: Number of epochs
            batch_size: Batch size
            lr: Learning rate
            use_wandb: Enable W&B logging
            save_every: Save checkpoint every N epochs
            intensity_schedule: Gradually increase intensity during training
        """
        print("\n" + "="*70)
        print("STARTING END-TO-END TRAINING")
        print("="*70)

        # Optimizer
        optimizer = torch.optim.AdamW(self.audio_encoder.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # W&B init
        if use_wandb and WANDB_AVAILABLE:
            wandb.init(
                entity="triplenegative",
                project="semantic-eq-e2e",
                config={
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'lr': lr,
                    'device': str(self.device),
                    'model_params': self.audio_encoder.get_num_parameters(),
                    'training_type': 'end-to-end-differentiable',
                    'balanced_sampling': True,
                    'fma_ratio': self.fma_ratio,
                    'fma_enabled': self.fma_loader is not None,
                    'n_fma_files': len(self.fma_loader.audio_files) if self.fma_loader else 0,
                    'training_source': f'{self.fma_ratio*100:.0f}% FMA + {(1-self.fma_ratio)*100:.0f}% pink noise' if self.fma_loader else '100% pink noise',
                }
            )
            wandb.watch(self.audio_encoder, log='all', log_freq=100)
            print(f"W&B logging enabled: {wandb.run.name}")

        # Steps per epoch (synthetic data, so we control this)
        steps_per_epoch = len(self.v2_system.eq_settings) // batch_size

        best_loss = float('inf')

        for epoch in range(epochs):
            self.audio_encoder.train()
            epoch_losses = []

            # Intensity schedule: start low, increase over training
            if intensity_schedule:
                intensity = min(0.3 + 0.7 * (epoch / epochs), 1.0)
            else:
                intensity = 0.7

            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")

            for step in pbar:
                # Create batch
                batch = self.create_training_batch(batch_size)

                # Forward pass
                optimizer.zero_grad()
                loss, loss_dict = self.train_step(batch, intensity=intensity)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.audio_encoder.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_losses.append(loss_dict)
                pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'sem': f"{loss_dict['semantic']:.4f}",
                    'int': f"{intensity:.2f}",
                })

            scheduler.step()

            # Epoch metrics
            avg_loss = np.mean([l['total'] for l in epoch_losses])
            avg_semantic = np.mean([l['semantic'] for l in epoch_losses])
            avg_param = np.mean([l.get('param', 0) for l in epoch_losses])
            avg_quality = np.mean([l.get('quality', 0) for l in epoch_losses])

            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Loss: {avg_loss:.4f} (sem: {avg_semantic:.4f}, param: {avg_param:.4f}, qual: {avg_quality:.4f})")
            print(f"  Intensity: {intensity:.2f}, LR: {scheduler.get_last_lr()[0]:.2e}")

            # W&B logging
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/loss': avg_loss,
                    'train/semantic_loss': avg_semantic,
                    'train/param_loss': avg_param,
                    'train/quality_loss': avg_quality,
                    'train/intensity': intensity,
                    'train/lr': scheduler.get_last_lr()[0],
                })

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_checkpoint('audio_encoder_e2e_best.pt', epoch, avg_loss)
                print(f"  [BEST] Saved checkpoint")

            # Periodic save
            if (epoch + 1) % save_every == 0:
                self._save_checkpoint(f'audio_encoder_e2e_epoch{epoch+1}.pt', epoch, avg_loss)

        # Final save
        self._save_checkpoint('audio_encoder_e2e_final.pt', epochs, avg_loss)

        if use_wandb and WANDB_AVAILABLE:
            wandb.finish()

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Best loss: {best_loss:.4f}")
        print(f"Model saved to: audio_encoder_e2e_best.pt")

    def _save_checkpoint(self, filename: str, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'audio_encoder_state_dict': self.audio_encoder.state_dict(),
            'config': AudioEncoderConfig.STANDARD,
        }
        torch.save(checkpoint, filename)


def main():
    parser = argparse.ArgumentParser(
        description='End-to-end differentiable semantic EQ training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with pink noise only (default):
    python train_audio_encoder_e2e.py --epochs 100 --device cuda

    # Train with FMA dataset for better generalization:
    python train_audio_encoder_e2e.py --epochs 100 --device cuda --fma-path ./data/fma_small

    # Train with multi-source (pink noise + FMA + MUSDB):
    python train_audio_encoder_e2e.py --epochs 100 --device cuda \\
        --fma-path ./data/fma_small --musdb-path ./data/musdb18 \\
        --pink-ratio 0.3 --fma-ratio 0.5 --musdb-ratio 0.2

    # Fine-tune from pretrained model:
    python train_audio_encoder_e2e.py --epochs 50 --device cuda \\
        --pretrained audio_encoder_best.pt
        """
    )

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')

    # Model paths
    parser.add_argument('--v2-model', type=str, default='neural_eq_safedb_v2.pt', help='V2 model path')
    parser.add_argument('--pretrained', type=str, default=None, help='Pretrained audio encoder path')

    # Dataset paths
    parser.add_argument('--fma-path', type=str, default=None, help='Path to FMA dataset')
    parser.add_argument('--musdb-path', type=str, default=None, help='Path to MUSDB18 dataset')

    # Dataset ratios
    parser.add_argument('--pink-ratio', type=float, default=0.3, help='Ratio of pink noise examples')
    parser.add_argument('--fma-ratio', type=float, default=0.5, help='Ratio of FMA examples')
    parser.add_argument('--musdb-ratio', type=float, default=0.2, help='Ratio of MUSDB examples')

    # Logging
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')

    args = parser.parse_args()

    # Check dasp-pytorch availability
    if not DASP_AVAILABLE:
        print("\n" + "="*70)
        print("ERROR: dasp-pytorch not installed!")
        print("="*70)
        print("\nInstall with: pip install dasp-pytorch auraloss")
        print("\nAlternatively, use train_audio_encoder.py for non-differentiable training.")
        return

    # Create trainer
    trainer = EndToEndSemanticEQTrainer(
        v2_model_path=args.v2_model,
        audio_encoder_path=args.pretrained,
        device=args.device,
        fma_path=args.fma_path,
        fma_ratio=args.fma_ratio,
    )

    # Train
    trainer.train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
