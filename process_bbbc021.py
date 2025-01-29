#!/usr/bin/env python3
"""
Script: prep_bbbc021_week1_subset.py

Description:
  Creates a small subset of the BBBC021 dataset (Week1 only),
  merges the 3 TIFF channels (DAPI, tubulin, actin) into a
  single 3-channel PNG for each sample, and writes a metadata.jsonl
  so the folder is ready for training with MorphoDiff.

Usage:
  python preprocessing/prep_bbbc021_week1_subset.py
"""

import os
import json
import pandas as pd
import numpy as np
from PIL import Image

######################################
# User-configurable settings
######################################

# Path to the CSV with full BBBC021 image metadata
BBBC021_IMAGE_CSV = "data/BBBC021/BBBC021_v1_image.csv"

# Directory containing unzipped Week1_XXXX subfolders (like Week1_22123, etc.)
BBBC021_IMAGES_DIR = "data/BBBC021/images"

# We'll store the subset here:
OUTPUT_DIR = "datasets/BBBC021/experiment_01_resized/train_imgs"

# We'll pick which compounds to include in the subset
# "aphidicolin,colchicine,cytochalasin-b,doxorubicin"
SUBSET_COMPOUNDS = [
    "aphidicolin",
    "colchicine",
    "cytochalasin B",
    "doxorubicin",
]  # Modify as you wish

# Max images per compound to include in the subset
IMAGES_PER_COMPOUND = 25

# Whether to restrict to "Week1_" only. Set True to skip anything not in Week1
WEEK1_ONLY = True


######################################
# Helper functions
######################################


def scale_minmax(channel):
    """
    Simple min-max normalization to convert channel to [0..1].
    """
    c_min, c_max = channel.min(), channel.max()
    if c_max > c_min:
        channel = (channel - c_min) / (c_max - c_min)
    else:
        channel = channel - c_min  # likely all zeros
    return channel


def merge_channels_to_rgb(dapi_path, tubulin_path, actin_path):
    """
    Load the three TIFF files, min-max normalize each channel,
    and return a 3-channel RGB (uint8) numpy array.
    """
    dapi_img = np.array(Image.open(dapi_path))
    tubulin_img = np.array(Image.open(tubulin_path))
    actin_img = np.array(Image.open(actin_path))

    # Convert to float in [0..1] range for each channel
    dapi_float = scale_minmax(dapi_img).astype(np.float32)
    tubulin_float = scale_minmax(tubulin_img).astype(np.float32)
    actin_float = scale_minmax(actin_img).astype(np.float32)

    # Stack into pseudo-RGB
    pseudo_rgb = np.stack([dapi_float, tubulin_float, actin_float], axis=-1)

    # Convert to 8-bit
    pseudo_rgb_8bit = (pseudo_rgb * 255).astype(np.uint8)
    return pseudo_rgb_8bit


######################################
# Main script
######################################


def main():
    # 1) Load the BBBC021 image metadata CSV
    if not os.path.isfile(BBBC021_IMAGE_CSV):
        raise FileNotFoundError(f"Cannot find BBBC021 metadata: {BBBC021_IMAGE_CSV}")

    df = pd.read_csv(BBBC021_IMAGE_CSV)

    # 2) Optionally restrict to Week1
    if WEEK1_ONLY:
        df = df[df["Image_PathName_DAPI"].str.startswith("Week1/")]

    # 3) Filter to chosen compounds
    df = df[df["Image_Metadata_Compound"].isin(SUBSET_COMPOUNDS)]

    # 4) For each compound, keep up to N images
    subset_rows = []
    for compound in SUBSET_COMPOUNDS:
        cdf = df[df["Image_Metadata_Compound"] == compound].reset_index(drop=True)
        # slice up to IMAGES_PER_COMPOUND
        cdf = cdf.iloc[:IMAGES_PER_COMPOUND]
        subset_rows.append(cdf)
    subset_df = pd.concat(subset_rows).reset_index(drop=True)

    if subset_df.empty:
        print("No matching images found for your subset settings. Exiting.")
        return

    # 5) Make sure OUTPUT_DIR exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 6) Build lines for metadata.jsonl
    metadata_lines = []

    print(f"Preparing to convert {len(subset_df)} images total...")

    for i, row in subset_df.iterrows():
        # File paths
        dapi_path = os.path.join(
            BBBC021_IMAGES_DIR, row["Image_PathName_DAPI"], row["Image_FileName_DAPI"]
        )
        tubulin_path = os.path.join(
            BBBC021_IMAGES_DIR,
            row["Image_PathName_Tubulin"],
            row["Image_FileName_Tubulin"],
        )
        actin_path = os.path.join(
            BBBC021_IMAGES_DIR, row["Image_PathName_Actin"], row["Image_FileName_Actin"]
        )

        # Convert the 3 TIFFs into a single PNG
        rgb_array = merge_channels_to_rgb(dapi_path, tubulin_path, actin_path)
        out_filename = f"image_{i:04d}.png"
        out_path = os.path.join(OUTPUT_DIR, out_filename)
        Image.fromarray(rgb_array).save(out_path)

        # Build the "perturbation id"
        # MorphoDiff typically uses just the compound name from bbbc021, ignoring concentration
        perturbation_id = row["Image_Metadata_Compound"]
        # If you prefer "compound_concentration", do something like:
        # pert_str = f"{row['Image_Metadata_Compound'].replace(' ', '-').lower()}_{row['Image_Metadata_Concentration']}"

        # For a simple example, we stick with the compound alone
        entry = {"file_name": out_filename, "additional_feature": perturbation_id}
        metadata_lines.append(json.dumps(entry))

    # 7) Write metadata.jsonl
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.jsonl")
    with open(metadata_path, "w") as f:
        for line in metadata_lines:
            f.write(line + "\n")

    print(f"All done! Created {len(subset_df)} PNG images in:")
    print(f"  {OUTPUT_DIR}")
    print(f"Wrote metadata.jsonl with {len(metadata_lines)} lines:")
    print(f"  {metadata_path}")
    print("\nExample lines from metadata.jsonl:")
    for line in metadata_lines[:3]:
        print(" ", line)


if __name__ == "__main__":
    main()
