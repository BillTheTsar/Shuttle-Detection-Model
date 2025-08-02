import os
from pathlib import Path
import random
import json

def index_dataset(sequences_root, output_dir, train_ratio=0.9, val_ratio=0.05, seed=42):
    """
    Create dataset index files for train/val/test splits.

    Args:
        sequences_root (str): Path to sequences directory.
        output_dir (str): Where to save split files (e.g., configs/).
        train_ratio (float): Portion for training.
        val_ratio (float): Portion for validation.
        seed (int): Random seed for reproducibility.
    """
    sequences_root = Path(sequences_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all sample directories
    sample_dirs = [p for p in sequences_root.rglob('*') if p.is_dir() and (p / "frame.jpg").exists()]
    print(f"✅ Found {len(sample_dirs)} samples")

    # Shuffle for randomness
    random.seed(seed)
    random.shuffle(sample_dirs)

    # Split
    total = len(sample_dirs)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_samples = sample_dirs[:train_end]
    val_samples = sample_dirs[train_end:val_end]
    test_samples = sample_dirs[val_end:]

    # Convert to relative paths for portability
    train_samples = [str(p.relative_to(sequences_root)) for p in train_samples]
    val_samples = [str(p.relative_to(sequences_root)) for p in val_samples]
    test_samples = [str(p.relative_to(sequences_root)) for p in test_samples]

    # Save to JSON
    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples
    }
    with open(output_dir / "dataset_index.json", "w") as f:
        json.dump(splits, f, indent=4)

    print(f"✅ Saved dataset splits to {output_dir / 'dataset_index.json'}")


# Example usage
if __name__ == "__main__":
    index_dataset(
        sequences_root="F:/GitHub/Shuttle Detection Model/sequences",
        output_dir="F:/GitHub/Shuttle Detection Model/configs"
    )
