"""
QUICK START: Semantic Interpolation
====================================

Copy-paste ready code to get started with semantic interpolation.
"""

# ============================================================================
# EXAMPLE 1: Basic Interpolation
# ============================================================================

from core.neural_eq_morphing import NeuralEQMorphingSystem, SocialFXDatasetLoader

# Load dataset
loader = SocialFXDatasetLoader()
eq_settings = loader.load_socialfx_dataset()

# Initialize and train system
system = NeuralEQMorphingSystem(latent_dim=32, device='cpu')
system.load_dataset(eq_settings)
system.train(epochs=50, batch_size=16)

# Interpolate between warm and bright
result = system.interpolate_semantic_terms('warm', 'bright', alpha=0.5)

print(f"Description: {result['description']}")
print(f"EQ Parameters: {result['parameters']}")


# ============================================================================
# EXAMPLE 2: Slider Simulation
# ============================================================================

# Simulate moving a slider from 0% to 100%
for slider_percent in [0, 25, 50, 75, 100]:
    alpha = slider_percent / 100.0

    result = system.interpolate_semantic_terms('warm', 'bright', alpha)

    print(f"\nSlider at {slider_percent}%:")
    print(f"  {result['description']}")


# ============================================================================
# EXAMPLE 3: Get Available Terms
# ============================================================================

# See what semantic terms are available
available = list(system.semantic_to_idx.keys())
print(f"You can interpolate between any of these: {available}")


# ============================================================================
# EXAMPLE 4: Multiple Term Pairs
# ============================================================================

# Try different semantic combinations
term_pairs = [
    ('warm', 'bright'),
    ('soft', 'harsh'),
    ('punchy', 'smooth')
]

for term1, term2 in term_pairs:
    if term1 in system.semantic_to_idx and term2 in system.semantic_to_idx:
        result = system.interpolate_semantic_terms(term1, term2, alpha=0.5)
        print(f"\n{term1} + {term2}:")
        print(f"  {result['description']}")


# ============================================================================
# EXAMPLE 5: Real-Time Interactive Control (Pseudocode)
# ============================================================================

"""
In a real audio plugin, you would use it like this:

class AudioPlugin:
    def __init__(self):
        self.eq_system = NeuralEQMorphingSystem(...)
        # Train system on startup

    def on_slider_moved(self, slider_value):
        # slider_value is between 0.0 and 1.0
        result = self.eq_system.interpolate_semantic_terms(
            self.selected_term_a,
            self.selected_term_b,
            alpha=slider_value
        )

        # Apply EQ parameters to audio
        self.set_eq_parameters(result['parameters'])

        # Update UI display
        self.display_text(result['description'])
"""


# ============================================================================
# EXAMPLE 6: Access Individual Parameters
# ============================================================================

result = system.interpolate_semantic_terms('warm', 'bright', alpha=0.7)

params = result['parameters']

# Access specific EQ bands
if 'band1_gain' in params:
    print(f"Low band gain: {params['band1_gain']:.2f} dB")
if 'band5_gain' in params:
    print(f"High band gain: {params['band5_gain']:.2f} dB")

# Iterate over all parameters
print("\nAll EQ parameters:")
for param_name, value in params.items():
    print(f"  {param_name}: {value:.3f}")


# ============================================================================
# EXAMPLE 7: Batch Processing
# ============================================================================

# Generate EQ settings for multiple alpha values
alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

results = []
for alpha in alphas:
    result = system.interpolate_semantic_terms('warm', 'bright', alpha)
    results.append(result)

# Now you have a list of EQ settings transitioning from warm to bright
print(f"\nGenerated {len(results)} EQ configurations")


# ============================================================================
# EXAMPLE 8: Error Handling
# ============================================================================

def safe_interpolate(system, term1, term2, alpha):
    """Safely interpolate with error checking"""

    # Check if terms exist
    available = list(system.semantic_to_idx.keys())

    if term1 not in available:
        print(f"Error: '{term1}' not available. Choose from: {available}")
        return None

    if term2 not in available:
        print(f"Error: '{term2}' not available. Choose from: {available}")
        return None

    # Clamp alpha to valid range
    alpha = max(0.0, min(1.0, alpha))

    # Perform interpolation
    return system.interpolate_semantic_terms(term1, term2, alpha)


# Use it
result = safe_interpolate(system, 'warm', 'bright', 1.5)  # Will clamp to 1.0
if result:
    print(f"Success: {result['description']}")
