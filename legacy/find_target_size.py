import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def find_max_image_size(img_root):
    max_width, max_height = 0, 0
    for root, _, files in os.walk(img_root):
        root_pretty = root.split("\\")
        root_pretty = root_pretty[-2] + "\\" + root_pretty[-1]
        print(f"Root: {root_pretty}")
        for file in tqdm(files):
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                with Image.open(img_path) as img:
                    width, height = img.size
                    max_width = max(max_width, width)
                    max_height = max(max_height, height)
    return max_width, max_height

def nearest_power_of_2(n):
    return 2 ** int(np.ceil(np.log2(n)))

def save_target_size(target_width, target_height, output_file='target_size.txt'):
    with open(output_file, 'w') as f:
        f.write(f"Target Width: {target_width}\n")
        f.write(f"Target Height: {target_height}\n")
    print(f"Target size saved to {output_file}")


def main():
    img_root = "D:\\Downloads\\маниока" # Root directory containing image folders
    max_width, max_height = find_max_image_size(img_root)
    print(f"Maximum image size: {max_width}x{max_height}")
    target_width = nearest_power_of_2(max_width)
    target_height = nearest_power_of_2(max_height)
    print(f"Target size: {target_width}x{target_height}")
    save_target_size(target_width, target_height)

if __name__ == '__main__':
    main()