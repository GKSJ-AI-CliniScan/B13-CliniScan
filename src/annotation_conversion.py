import pandas as pd
import os

CSV_PATH = "data/raw/train.csv"
LABEL_OUTPUT = "data/processed/labels"

os.makedirs(LABEL_OUTPUT, exist_ok=True)

df = pd.read_csv(CSV_PATH)

for image_id in df['image_id'].unique():
    image_data = df[df['image_id'] == image_id]

    with open(os.path.join(LABEL_OUTPUT, image_id + ".txt"), "w") as f:
        for _, row in image_data.iterrows():

            x_center = (row['x_min'] + row['x_max']) / 2
            y_center = (row['y_min'] + row['y_max']) / 2
            width = row['x_max'] - row['x_min']
            height = row['y_max'] - row['y_min']

            f.write(f"0 {x_center} {y_center} {width} {height}\n")

print("Annotation conversion completed.")
