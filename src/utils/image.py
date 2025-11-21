import numpy as np

def white_pixel_percentage(img):
    white = (img == 255).all(-1)
    return 100 * white.mean()