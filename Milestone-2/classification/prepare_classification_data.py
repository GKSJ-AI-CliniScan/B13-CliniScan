import os
import shutil

images_path = r"D:\Module_2\dataset\images"
labels_path = r"D:\Module_2\dataset\labels"

normal_dir = r"D:\Module_2\classification\data\normal"
abnormal_dir = r"D:\Module_2\classification\data\abnormal"

os.makedirs(normal_dir, exist_ok=True)
os.makedirs(abnormal_dir, exist_ok=True)

for img in os.listdir(images_path):

    img_path = os.path.join(images_path, img)

    label_file = img.replace(".png", ".txt")
    label_path = os.path.join(labels_path, label_file)

    if os.path.exists(label_path):

        if os.path.getsize(label_path) == 0:
            shutil.copy(img_path, normal_dir)

        else:
            shutil.copy(img_path, abnormal_dir)

print("Dataset prepared successfully!")