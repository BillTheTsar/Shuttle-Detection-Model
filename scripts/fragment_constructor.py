import os
import pandas as pd
from pathlib import Path
import shutil

def prepare_fragment(csv_path, fragment_id, sequences_root='F:/GitHub/Shuttle Detection Model/sequences'):
    """
    Prepare a structured dataset directory for one video fragment.

    Args:
        csv_path (str): Path to the CSV file for this fragment.
        fragment_id (str): Unique identifier for the fragment, e.g. '001'.
        sequences_root (str): Path to the top-level 'sequences' directory.
    """

    df = pd.read_csv(csv_path)
    sequence_path = Path(sequences_root) / fragment_id

    for idx, row in df.iterrows():
        entry_dir = sequence_path / f"{idx:05d}"
        entry_dir.mkdir(parents=True, exist_ok=True)

        # Copy current frame to frame.jpg
        curr_frame_path = Path(row['currentImagePath'])
        shutil.copy(curr_frame_path, entry_dir / "frame.jpg")

        # Copy the past 3 frames (M = 3)
        for i in range(1, 4):
            frame_key = f"-{i}ImagePath"
            past_frame_path = Path(row[frame_key])
            shutil.copy(past_frame_path, entry_dir / f"frame-{i}.jpg")

        # Save past 30 shuttle points to positions.csv
        # Remember we are saving them in the order, -1, -2, ..., -30, which is reverse the order in the training csv
        x_cols = [f"-{i+1}_x" for i in range(30)]
        y_cols = [f"-{i+1}_y" for i in range(30)]
        v_cols = [f"-{i+1}Visibility" for i in range(30)]

        positions = list(zip(row[x_cols], row[y_cols], row[v_cols]))
        positions_df = pd.DataFrame(positions, columns=["x", "y", "visible"])
        positions_df.to_csv(entry_dir / "positions.csv", index=False)

        # Save current target info to target.csv
        target = {
            "shuttle_x": row["current_x"],
            "shuttle_y": row["current_y"],
            "shuttle_visibility": row["currentVisibility"]
        }
        pd.DataFrame([target]).to_csv(entry_dir / "target.csv", index=False)

    print(f"✅ Fragment '{fragment_id}' prepared under {sequence_path}")

prepare_fragment("F:/GitHub/Shuttle Detection Model/csv/3_csv/3201-3700_training.csv", "011")