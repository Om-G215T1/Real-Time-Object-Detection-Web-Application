import os
import random
import shutil

# Paths
base_path = "dataset/coco128"
images_path = os.path.join(base_path, "images/train2017")
labels_path = os.path.join(base_path, "labels/train2017")

# Output paths
output_base = "dataset"
splits = ["train", "val", "test"]

for split in splits:
    os.makedirs(f"{output_base}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_base}/labels/{split}", exist_ok=True)

# Get all images
images = [f for f in os.listdir(images_path) if f.endswith(".jpg")]
random.shuffle(images)

# Split ratios
train_split = 0.7
val_split = 0.2

train_count = int(len(images) * train_split)
val_count = int(len(images) * val_split)

train_files = images[:train_count]
val_files = images[train_count:train_count + val_count]
test_files = images[train_count + val_count:]

def move_files(file_list, split):
    for file in file_list:
        img_src = os.path.join(images_path, file)
        lbl_src = os.path.join(labels_path, file.replace(".jpg", ".txt"))

        img_dst = f"{output_base}/images/{split}/{file}"
        lbl_dst = f"{output_base}/labels/{split}/{file.replace('.jpg', '.txt')}"

        shutil.copy(img_src, img_dst)
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, lbl_dst)

move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print("✅ Dataset split completed!")