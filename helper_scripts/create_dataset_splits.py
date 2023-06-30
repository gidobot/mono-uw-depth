# this file splits the whole dataset into training and test data and
# generates respective csv files with the names of the [imagea, ground_truth_depth] pairs

import csv
import glob
from os.path import join, basename, exists, splitext
import cv2
import numpy as np
import pandas as pd
import random

###### CONFIG
location = "0000"
images_folder = (
    f"/media/auv/Seagate_2TB/datasets/ycb/ycb_dataset/ycb_dataset/data/{location}/"
)
ground_truth_depth_folder = (
    f"/media/auv/Seagate_2TB/datasets/ycb/ycb_dataset/ycb_dataset/data/{location}/"
)
features_folder = join(
    images_folder, "matched_features"
)  # precomputed depth features, if any
images_pattern = "-color.png"
ground_truth_depth_pattern = "-depth.png"
features_pattern = "_features.csv"
split_sizes = [
    1.0,
    # 0.0,
    # 0.0,
]  # train, validation, test percentage: Must add up to 1.
split_names = [
    "dataset_with_matched_features",
    # "tmp",
    # "tmp",
]
allow_zero = True  # allow depth imgs with pixel values zero (=invalid)
allow_zero_range = False  # allow img range [0,0]

#######

# search candidates
depth_candidate_paths = glob.glob(
    join(ground_truth_depth_folder, "*" + ground_truth_depth_pattern)
)
print(f"Found {len(depth_candidate_paths)} candidates")

# find pairs
imgs = []
depths = []
i = 0
for depth_candidate_path in depth_candidate_paths:

    # check if depth map is valid, if not then skip
    if not allow_zero or not allow_zero_range:
        depth_candidate = cv2.imread(depth_candidate_path, cv2.IMREAD_UNCHANGED)
        if not allow_zero_range and (depth_candidate.min() == depth_candidate.max()):
            continue
        if not allow_zero and (np.any(depth_candidate <= 0.0)):
            continue

    # get img name
    img_name = basename(depth_candidate_path).split(ground_truth_depth_pattern)[0]

    # append pair to imgs and depths list
    imgs.append(join(images_folder, img_name + images_pattern))
    depths.append(depth_candidate_path)

    if i % 100 == 0:
        print(
            f"Checking depth_candidates for invalid values: {i}/{len(depth_candidate_paths)}"
        )
    i += 1
print(f"Found {len(depths)} valid depth maps.")

# shuffle
pairs = list(zip(imgs, depths))
random.shuffle(pairs)

# get splits ranges
n_pairs = len(imgs)
n_splits = len(split_sizes)
split_ranges = []
idx = 0
for i in range(n_splits):
    split_ranges.append((idx, idx + int(split_sizes[i] * n_pairs)))
    idx += int(split_sizes[i] * n_pairs)
# print(f"n_pairs: {n_pairs}, split_ranges: {split_ranges}")

# create splits, each split is a tuple with (imgs_list, depths_list)
splits = []
for split_range in split_ranges:
    splits.append((sorted(pairs[split_range[0] : split_range[1]])))

# validation check
for split in splits:  # for all splits
    for img, depth in split:

        if (not exists(img)) or (not exists(depth)):
            print("Validation failed, file is missing!")
            print(f"Missing pair: {img}, {depth}")
            exit(1)

# Summary
print(f"Created {n_splits} splits:")
for i in range(n_splits):
    split_name = split_names[i]
    print(f"    - {split_name} split [{len(splits[i])} pairs]")

# write file
print(f"Writing csv files to {images_folder} ...")
for i in range(n_splits):
    split_name = split_names[i]
    split_imgs = [img for img, _ in splits[i]]
    split_depths = [depth for _, depth in splits[i]]
    split_features = [
        join(features_folder, splitext(basename(img))[0] + features_pattern)
        for img in split_imgs
    ]
    d = {"img": split_imgs, "depth": split_depths, "features": split_features}
    df = pd.DataFrame.from_dict(d)
    df.to_csv(join(images_folder, split_name + ".csv"), index=False, header=False)

print("Done.")


# if __name__ == "__main__":

#     locations = 92
#     for i in range(locations):
#         location = f"{i:04}"
#         main(location)
