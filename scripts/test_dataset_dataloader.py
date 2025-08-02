import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import Stage1Dataset, Stage2Dataset
import torch

def show_sample(batch, stage=1, sample_idx=0):
    if stage == 1:
        current_img = batch["current_img"][sample_idx].permute(1, 2, 0).numpy()
        past_imgs = batch["past_imgs"][sample_idx].permute(0, 2, 3, 1).numpy()
        target = batch["target"][sample_idx].numpy()
        img_h, img_w = current_img.shape[:2]
    else:
        current_img = batch["current_crop"][sample_idx].permute(1, 2, 0).numpy()
        past_imgs = batch["past_crops"][sample_idx].permute(0, 2, 3, 1).numpy()
        target = batch["target"][sample_idx].numpy()
        img_h, img_w = current_img.shape[:2]

    # Denormalize if needed (currently in [0,1])
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    # Current frame with target
    axes[0].imshow(current_img)
    if target[2] == 1:  # If visible
        cx = int(target[0] * img_w)
        cy = int(target[1] * img_h)
        axes[0].scatter(cx, cy, c='red', s=40, label='Target')
        axes[0].legend()
    axes[0].set_title("Current Frame")

    # Past frames
    for i in range(3):
        axes[i+1].imshow(past_imgs[i])
        axes[i+1].set_title(f"Past {3-i}")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    root = "F:/GitHub/Shuttle Detection Model/sequences"
    index = "F:/GitHub/Shuttle Detection Model/configs/dataset_index.json"

    # Choose stage: 1 or 2
    stage = 1
    if stage == 1:
        dataset = Stage1Dataset(root, index, split="train", train_mode=True)
    else:
        dataset = Stage2Dataset(root, index, split="train", train_mode=True)

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    print(f"Batch keys: {batch.keys()}")
    print(batch["target"][0].numpy())
    show_sample(batch, stage=stage, sample_idx=0)

