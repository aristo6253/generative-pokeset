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

    return parser.parse_args()

def sanitize_filename(filename):
    # Replace invalid characters with an underscore or any other valid character
    return "".join([c if c.isalnum() or c in (' ', '.', '_') else '_' for c in filename])


if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()

    # Load the data
    poke_cards_df = pd.read_csv("hf://datasets/TheFusion21/PokemonCards/train.csv")

    dir_name = f'data/cards_{args.image_size}_{"RGB" if args.color else "G"}'

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for index, row in tqdm(poke_cards_df.iterrows(), total=len(poke_cards_df)):
        img = Image.open(BytesIO(requests.get(row['image_url']).content))
        if not args.color:
            img = img.convert('L')
        img = img.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
        sanitized_name = sanitize_filename(row["name"])
        count = 0
        while os.path.exists(os.path.join(dir_name, f"{sanitized_name}.png")):
            count += 1
            sanitized_name = f"{sanitized_name}_{count}"

        img.save(os.path.join(dir_name, f"{sanitized_name}.png"))
        