import pandas as pd
from tqdm import tqdm
import argparse
from PIL import Image
from io import BytesIO
import requests
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Pokemon Card Set')

    parser.add_argument('--image_size', type=int, default=64, help='Size of the image')
    parser.add_argument('--color', action='store_true', help='Use color images')
    parser.add_argument('--resume_from', type=int, default=0, help='Resume from index')

    return parser.parse_args()

def sanitize_filename(filename, max_length=100):
    sanitized = "".join([c if c.isalnum() or c in (' ', '.', '_') else '_' for c in filename])
    return sanitized[:max_length]

if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()

    # Load the data
    poke_cards_df = pd.read_csv("hf://datasets/TheFusion21/PokemonCards/train.csv")

    dir_name = f'data/cards_{args.image_size}_{"RGB" if args.color else "G"}'

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    existing_files = {f.split('.')[0] for f in os.listdir(dir_name)}

    for index, row in tqdm(poke_cards_df.iterrows(), total=len(poke_cards_df)):
        if index < args.resume_from:
            continue
        # print(f"Processing {row['name']}")
        
        try:
            img = Image.open(BytesIO(requests.get(row['image_url']).content))
        except Exception as e:
            print(f"Failed to download image for {row['name']}: {e}")
            continue
        
        if not args.color:
            img = img.convert('L')
        img = img.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
        
        sanitized_name = sanitize_filename(row["name"])
        file_index = 0
        file_path = os.path.join(dir_name, f"{sanitized_name}_{file_index}.png")
        
        while os.path.exists(file_path) or f"{sanitized_name}_{file_index}" in existing_files:
            file_index += 1
            file_path = os.path.join(dir_name, f"{sanitized_name}_{file_index}.png")
        
        existing_files.add(f"{sanitized_name}_{file_index}")

        try:
            img.save(file_path)
        except Exception as e:
            print(f"Failed to save image for {row['name']}: {e}")
