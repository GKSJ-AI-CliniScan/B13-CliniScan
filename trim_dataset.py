import os
import random

def trim_data(img_dir, label_dir, keep_count):
    images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    if len(images) > keep_count:
        to_remove = random.sample(images, len(images) - keep_count)
        for img in to_remove:
            os.remove(os.path.join(img_dir, img))
            label_path = os.path.join(label_dir, img.replace('.png', '.txt'))
            if os.path.exists(label_path):
                os.remove(label_path)
        print(f"✅ Kept exactly {keep_count} images in {img_dir}.")
    else:
        print(f"⚠️ {img_dir} already has {len(images)} images.")

print("Trimming dataset down to 2,000 total images...")
# 1,600 Train + 400 Val = 2,000 Total Images
trim_data('vinbigdata_yolo_preprocessed/train/images', 'vinbigdata_yolo_preprocessed/train/labels', 1600)
trim_data('vinbigdata_yolo_preprocessed/val/images', 'vinbigdata_yolo_preprocessed/val/labels', 400)
print("🎉 Done! You now have exactly 2,000 images.")