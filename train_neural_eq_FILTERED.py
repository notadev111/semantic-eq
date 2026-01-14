"""
Train Neural EQ on FILTERED Dataset
====================================

Only use top semantic terms with sufficient examples.
Stronger contrastive loss to encourage clustering.
"""

import torch
import torch.nn as nn
from core.neural_eq_morphing import NeuralEQMorphingSystem, SocialFXDatasetLoader
import numpy as np

def main():
    print("="*70)
    print("TRAINING NEURAL EQ - FILTERED DATASET")
    print("="*70)

    # Configuration
    MIN_EXAMPLES = 10  # Only keep terms with >=10 examples
    LATENT_DIM = 32
    EPOCHS = 100
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    CONTRASTIVE_WEIGHT = 0.5  # INCREASED from 0.1!

    # Step 1: Load SocialFX dataset
    print("\n[1/5] Loading SocialFX dataset...")
    loader = SocialFXDatasetLoader()
    eq_settings = loader.load_socialfx_dataset()

    if not eq_settings:
        print("ERROR: Failed to load dataset")
        return

    print(f"Loaded {len(eq_settings)} total EQ settings")

    # Step 2: Filter to top terms
    print(f"\n[2/5] Filtering to terms with >={MIN_EXAMPLES} examples...")

    # Count examples per term
    from collections import Counter
    term_counts = Counter(eq.semantic_label for eq in eq_settings)

    # Filter
    valid_terms = {term for term, count in term_counts.items() if count >= MIN_EXAMPLES}
    filtered_settings = [eq for eq in eq_settings if eq.semantic_label in valid_terms]

    print(f"Filtered: {len(filtered_settings)}/{len(eq_settings)} examples ({100*len(filtered_settings)/len(eq_settings):.1f}%)")
    print(f"Kept: {len(valid_terms)}/765 unique terms ({100*len(valid_terms)/765:.1f}%)")

    # Show top terms
    filtered_counts = Counter(eq.semantic_label for eq in filtered_settings)
    print(f"\nTop 10 terms in filtered dataset:")
    for i, (term, count) in enumerate(filtered_counts.most_common(10), 1):
        print(f"  {i}. {term:<15} {count:>3} examples")

    # Step 3: Initialize system
    print(f"\n[3/5] Initializing neural network...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    system = NeuralEQMorphingSystem(latent_dim=LATENT_DIM, device=device)

    if not system.load_dataset(filtered_settings):
        print("ERROR: Failed to load dataset into system")
        return

    # Step 4: Modify training for stronger contrastive loss
    print(f"\n[4/5] Training with STRONGER contrastive loss...")
    print(f"Configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Contrastive weight (lambda): {CONTRASTIVE_WEIGHT} (was 0.1)")
    print(f"  Expected time: ~10 min on CPU")
    print("-"*70)

    # Prepare data
    dataset = torch.utils.data.TensorDataset(
        system.training_data['params'],
        system.training_data['labels']
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # Optimizers
    optimizer = torch.optim.Adam(
        list(system.encoder.parameters()) + list(system.decoder.parameters()),
        lr=LEARNING_RATE
    )

    # Loss functions
    reconstruction_loss_fn = nn.MSELoss()
    from core.neural_eq_morphing import ContrastiveEQLoss
    contrastive_loss_fn = ContrastiveEQLoss()

    # Training loop
    system.encoder.train()
    system.decoder.train()

    best_loss = float('inf')
    loss_history = {'epoch': [], 'total': [], 'recon': [], 'contrast': []}

    for epoch in range(EPOCHS):
        total_loss = 0
        total_recon = 0
        total_contrast = 0
        n_batches = 0

        for batch_params, batch_labels in dataloader:
            optimizer.zero_grad()

            # Forward pass
            latent, semantic_emb = system.encoder(batch_params)
            reconstructed = system.decoder(latent)

            # Losses
            recon_loss = reconstruction_loss_fn(reconstructed, batch_params)
            contrast_loss = contrastive_loss_fn(semantic_emb, batch_labels)

            # Combined loss with STRONGER contrastive weight
            loss = recon_loss + CONTRASTIVE_WEIGHT * contrast_loss

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_contrast += contrast_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_contrast = total_contrast / n_batches

        # Save history
        loss_history['epoch'].append(epoch)
        loss_history['total'].append(avg_loss)
        loss_history['recon'].append(avg_recon)
        loss_history['contrast'].append(avg_contrast)

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}: Loss={avg_loss:.4f} (Recon={avg_recon:.4f}, Contrast={avg_contrast:.4f})")

        if avg_loss < best_loss:
            best_loss = avg_loss

    system.is_trained = True
    system.encoder.eval()
    system.decoder.eval()

    print(f"\nTraining completed! Best loss: {best_loss:.4f}")

    # Step 5: Evaluate clustering
    print(f"\n[5/5] Evaluating clustering quality...")

    with torch.no_grad():
        latent_all, _ = system.encoder(system.training_data['params'])
        latent_np = latent_all.numpy()

    labels_np = system.training_data['labels'].numpy()

    from sklearn.metrics import silhouette_score, davies_bouldin_score

    # Remap labels to consecutive integers
    unique_labels = np.unique(labels_np)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    remapped_labels = np.array([label_map[int(l)] for l in labels_np])

    sil = silhouette_score(latent_np, remapped_labels)
    db = davies_bouldin_score(latent_np, remapped_labels)

    print(f"\nClustering metrics (in 32D latent space):")
    print(f"  Silhouette score: {sil:.3f}")
    print(f"  Davies-Bouldin index: {db:.3f}")

    if sil > 0:
        print(f"  [OK] Positive silhouette - clustering improved!")
    else:
        print(f"  [WARNING] Still negative - may need more epochs or higher lambda")

    # Save model
    print(f"\n" + "="*70)
    print("SAVING IMPROVED MODEL")
    print("="*70)

    torch.save({
        'encoder': system.encoder.state_dict(),
        'decoder': system.decoder.state_dict(),
        'semantic_to_idx': system.semantic_to_idx,
        'idx_to_semantic': system.idx_to_semantic,
        'param_means': system.param_means,
        'param_stds': system.param_stds,
        'training_data': system.training_data,
        'loss_history': loss_history,
        'config': {
            'min_examples': MIN_EXAMPLES,
            'latent_dim': LATENT_DIM,
            'epochs': EPOCHS,
            'contrastive_weight': CONTRASTIVE_WEIGHT,
            'n_examples': len(filtered_settings),
            'n_terms': len(valid_terms),
        },
        'metrics': {
            'silhouette': sil,
            'davies_bouldin': db,
        }
    }, 'neural_eq_model_FILTERED.pt')

    print(f"[OK] Model saved to: neural_eq_model_FILTERED.pt")
    print(f"\nModel stats:")
    print(f"  Training examples: {len(filtered_settings)}")
    print(f"  Semantic terms: {len(valid_terms)}")
    print(f"  Silhouette score: {sil:.3f}")
    print(f"  Davies-Bouldin: {db:.3f}")

if __name__ == '__main__':
    main()
