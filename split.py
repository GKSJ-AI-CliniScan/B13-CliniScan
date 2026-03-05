import random
import shutil
from pathlib import Path

IMAGES_SRC = Path(r"C:\Users\vibes\Downloads\images3000_processed\images_processed")
LABELS_SRC = Path(r"C:\Users\vibes\Downloads\images3000_processed\labels")

OUTPUT_DIR = Path("dataset")

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

random.seed(42)  

def create_dirs():
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)


def split_dataset():
    images = list(IMAGES_SRC.glob("*.[jp][pn]g"))
    random.shuffle(images)

    n = len(images)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_imgs = images[:n_train]
    val_imgs   = images[n_train:n_train+n_val]
    test_imgs  = images[n_train+n_val:]

    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs,
    }

    for split, img_list in splits.items():
        print(f"{split}: {len(img_list)} images")

        for img_path in img_list:
            label_path = LABELS_SRC / (img_path.stem + ".txt")

            # copy image
            shutil.copy(
                img_path,
                OUTPUT_DIR / "images" / split / img_path.name
            )

            # copy label (if exists)
            if label_path.exists():
                shutil.copy(
                    label_path,
                    OUTPUT_DIR / "labels" / split / label_path.name
                )


if __name__ == "__main__":
    create_dirs()
    split_dataset()
    print("Dataset split complete ✔")