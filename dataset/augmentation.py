# dataset/augmentation.py
# Data augmentation configuration for YOLOv8 training
# These settings are passed directly into model.train()

# Augmentation parameters optimized for COCO dataset
AUGMENTATION_CONFIG = {
    # Color augmentation
    "hsv_h": 0.015,       # Hue shift (±1.5%) — handles lighting changes
    "hsv_s": 0.7,         # Saturation (±70%) — handles color variation
    "hsv_v": 0.4,         # Brightness (±40%) — handles dark/bright scenes

    # Geometric augmentation
    "degrees": 10.0,      # Rotation range ±10 degrees
    "translate": 0.1,     # Translation ±10% of image size
    "scale": 0.5,         # Scale ±50%
    "shear": 2.0,         # Shear ±2 degrees
    "perspective": 0.0,   # Perspective distortion (keep 0 for stability)
    "flipud": 0.0,        # Vertical flip (not useful for most objects)
    "fliplr": 0.5,        # Horizontal flip 50% — very effective

    # Advanced augmentation
    "mosaic": 1.0,        # Mosaic (combines 4 images) — best for small objects
    "mixup": 0.1,         # MixUp (blends 2 images) — improves generalization
    "copy_paste": 0.0,    # Copy-paste augmentation (keep 0 for now)

    # Erasing
    "erasing": 0.4,       # Random erasing — simulates occlusion
}

def get_augmentation_config():
    """Returns augmentation config dict to pass into model.train()"""
    return AUGMENTATION_CONFIG

def print_augmentation_summary():
    """Prints a summary of active augmentations."""
    print("\n--- Augmentation Config ---")
    for key, value in AUGMENTATION_CONFIG.items():
        status = "ON" if value > 0 else "OFF"
        print(f"  {key:<20} {value:<10} [{status}]")
    print("---------------------------\n")

if __name__ == '__main__':
    print_augmentation_summary()