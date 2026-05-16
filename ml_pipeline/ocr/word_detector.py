"""
Word Detector — extracts individual word bounding boxes from a line strip image.
Uses connected component analysis on binarized input. Pure OpenCV, no ML required.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple


def detect_word_boxes(
    line_image: Image.Image,
    min_w: int = 8,
    min_h: int = 6,
    dilation_w: int = 18,   # merge character strokes into word blobs
) -> List[Tuple[int, int, int, int]]:
    """
    Find (x, y, w, h) bounding boxes of words in a single line strip.

    Args:
        line_image:  PIL image of one text line
        min_w:       minimum bounding box width to keep
        min_h:       minimum bounding box height to keep
        dilation_w:  horizontal dilation width — increase to merge more characters

    Returns:
        List of (x, y, w, h) sorted left-to-right
    """
    gray = np.array(line_image.convert("L"))

    # Binarize: ink pixels become 255
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal dilation merges individual character strokes into word blobs
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_w, 1))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [(x, y, w, h) for x, y, w, h in boxes if w >= min_w and h >= min_h]
    boxes.sort(key=lambda b: b[0])   # left → right
    return boxes


def crop_words(
    line_image: Image.Image,
    padding: int = 2,
    **kwargs,
) -> List[Image.Image]:
    """
    Return word-crop PIL images from a line strip, sorted left-to-right.

    Args:
        line_image: PIL image of one text line
        padding:    extra pixels added around each crop
        **kwargs:   forwarded to detect_word_boxes

    Returns:
        List of PIL word-crop images
    """
    boxes  = detect_word_boxes(line_image, **kwargs)
    w_img, h_img = line_image.size
    crops  = []

    for (x, y, w, h) in boxes:
        x0 = max(0, x - padding)
        y0 = max(0, y - padding)
        x1 = min(w_img, x + w + padding)
        y1 = min(h_img, y + h + padding)
        crops.append(line_image.crop((x0, y0, x1, y1)))

    return crops
