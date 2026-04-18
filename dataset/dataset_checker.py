# # dataset/dataset_checker.py
# # Verifies dataset structure is correct before training

# import os
# import yaml

# DATASET_PATH = 'dataset'
# SPLITS = ['train', 'val', 'test']

# def check_dataset():
#     print("\n--- Dataset Health Check ---")

#     # Check data.yaml exists
#     yaml_path = os.path.join(DATASET_PATH, 'data.yaml')
#     if os.path.exists(yaml_path):
#         print(f"  data.yaml         Found")
#         with open(yaml_path) as f:
#             config = yaml.safe_load(f)
#         print(f"  Classes           {config['nc']} classes")
#     else:
#         print(f"  data.yaml          Missing!")
#         return

#     # Check each split
#     for split in SPLITS:
#         img_dir = os.path.join(DATASET_PATH, 'images', split)
#         lbl_dir = os.path.join(DATASET_PATH, 'labels', split)

#         img_count = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
#         lbl_count = len(os.listdir(lbl_dir)) if os.path.exists(lbl_dir) else 0

#         status = "✅" if img_count > 0 else "⚠️ Empty"
#         print(f"  {split:<8} images: {img_count:<6} labels: {lbl_count:<6} {status}")

#     print("----------------------------\n")

# if __name__ == '__main__':
#     check_dataset()


# dataset/dataset_checker.py
# Verifies dataset structure is correct before training

import os
import yaml

DATASET_PATH = 'dataset'
SPLITS = ['train', 'val', 'test']

def check_dataset():
    print("\n--- Dataset Health Check ---")

    # Check data.yaml exists
    yaml_path = os.path.join(DATASET_PATH, 'data.yaml')
    if os.path.exists(yaml_path):
        print(f"  data.yaml         ✅ Found")
        with open(yaml_path) as f:
            config = yaml.safe_load(f)
        print(f"  Classes           ✅ {config['nc']} classes")
    else:
        print(f"  data.yaml         ❌ Missing!")
        return

    # Check each split
    for split in SPLITS:
        img_dir = os.path.join(DATASET_PATH, 'images', split)
        lbl_dir = os.path.join(DATASET_PATH, 'labels', split)

        img_count = len(os.listdir(img_dir)) if os.path.exists(img_dir) else 0
        lbl_count = len(os.listdir(lbl_dir)) if os.path.exists(lbl_dir) else 0

        status = "✅" if img_count > 0 else "⚠️ Empty"
        print(f"  {split:<8} images: {img_count:<6} labels: {lbl_count:<6} {status}")

    print("----------------------------\n")

if __name__ == '__main__':
    check_dataset()