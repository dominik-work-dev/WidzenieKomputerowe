import numpy as np
from PIL import Image

def load_rgb(path: str, size: tuple[int, int] = (512, 512)) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.resize(size, Image.Resampling.LANCZOS)

def to_array(image: Image.Image) -> np.ndarray:
    return np.array(image)

def to_image(array: np.ndarray) -> Image.Image:
    return Image.fromarray(array)