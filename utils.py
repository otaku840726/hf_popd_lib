from PIL import Image
from datetime import datetime
import os
from numpy import uint8

def array_to_image_path(image_array):
    if image_array is None:
        raise ValueError("No image provided. Please upload an image before submitting.")
    img = Image.fromarray(uint8(image_array))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"image_{timestamp}.png"
    img.save(filename)
    full_path = os.path.abspath(filename)
    return full_path

def load_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()