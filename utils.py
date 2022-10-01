import numpy as np
from PIL import Image
from io import BytesIO


def read_and_resize(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    image = np.resize(image, (160,160,3))
    return image
