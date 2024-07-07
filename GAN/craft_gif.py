import imageio
import os
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser(description='Create GIF from images')
    parser.add_argument('--image_folder', type=str, required=True, help='Directory containing the .bmp images')
    parser.add_argument('--output_file', type=str, required=True, help='Output GIF file')
    parser.add_argument('--duration', type=float, default=0.25, help='Duration of each frame in the GIF')
    parser.add_argument('--up_to', type=int, default=None, help='Number of images to include in the GIF')
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    # Directory containing the .bmp images
    image_folder = args.image_folder

    # Output GIF file
    output_file = args.output_file

    # Regular expression to extract the number
    def extract_number(file_name):
        match = re.search(r'epoch_(\d+)', file_name)
        if match:
            return int(match.group(1))
        else:
            return float('inf')  # Place any non-matching file names at the end

    # Read images from the directory, sorted by the extracted number
    images = []
    for i, file_name in enumerate(sorted(os.listdir(image_folder), key=extract_number)):
        if file_name.startswith('epoch_') and file_name.endswith('.bmp') and (args.up_to is None or i + 1 <= args.up_to):
            print(file_name)
            file_path = os.path.join(image_folder, file_name)
            images.append(imageio.imread(file_path))

    # Create the GIF
    print('Creating GIF...')
    imageio.mimsave(output_file, images, duration=args.duration)
    print('GIF created at', output_file)
