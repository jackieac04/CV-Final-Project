import os
from PIL import Image

def find_problematic_image(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                file_path = os.path.join(root, filename)
                try:
                    with Image.open(file_path) as img:
                        img.verify()  # Attempt to open and verify the image
                except (Image.UnidentifiedImageError) as e:
                    print("Error processing image:", file_path)
                    print("Error message:", str(e))

folder_path = "./"
find_problematic_image(folder_path)
