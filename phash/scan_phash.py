import cv2 as cv
from ultralytics import YOLO
import imagehash
from PIL import Image
import pandas as pd
import numpy as np

def difference(hash1: str, hash2: str):
    """
    Computes the Hamming distance (number of differing bits) between two perceptual hashes.

    Args:
        hash1 (str): Hexadecimal string of the first hash.
        hash2 (str): Hexadecimal string of the second hash.

    Returns:
        int: Number of differing bits between the two hashes.
    """
    diff = int(hash1, 16) ^ int(hash2, 16)
    return diff.bit_count()


def find_card(card_hash):
    """
    Finds the closest matching card in the database using perceptual hash comparison.

    Args:
        card_hash (str): Hexadecimal string of the perceptual hash of the detected card.

    Returns:
        str: Name of the closest matching card.
    """
    min_name = df['name'][0]
    min_hash = df['phash'][0]
    min_diff = difference(min_hash, card_hash)

    for row in df.itertuples():
        diff = difference(row.phash, card_hash)
        if (diff < min_diff):
            min_name = row.name
            min_hash = row.phash
            min_diff = diff
    if (min_diff < 8):
        print(min_diff)
    return min_name


def get_corners(coords, grid_x, grid_y): # the yolo segmentation model sometimes halucinates protrusions from the card so this filters them out
    """
    Finds the closest detected points to the four image corners.

    Args:
        coords (numpy.ndarray): Array of (x, y) points defining the detected card contour.
        grid_x (int): Image width.
        grid_y (int): Image height.

    Returns:
        list: Four points (top-left, top-right, bottom-right, bottom-left) 
              corresponding to the closest detected points to each image corner.
    """
    corners = np.array([
        [0,0],
        [grid_x, 0],
        [grid_x, grid_y],
        [0, grid_y]
    ])
    mask_corners = []
    for corner in corners:
        min_distance = np.linalg.norm(corner - coords[0])
        closest = coords[0]
        for coord in coords:
                distance = np.linalg.norm(corner - coord)
                if (distance < min_distance):
                     min_distance = distance
                     closest = coord
        mask_corners.append(closest)
    return mask_corners


def isolate_card(img, card_pts):
    """
    Performs a perspective transform to extract a card from an image.

    Args:
        img (numpy.ndarray): Source image.
        card_pts (numpy.ndarray): Four points (in float32) defining the card's corners in the image.

    Returns:
        numpy.ndarray: Warped image of the card with fixed dimensions (672x936).
    """
    width, height = 672, 936
    pts2 = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])
    matrix = cv.getPerspectiveTransform(card_pts, pts2)
    return cv.warpPerspective(img, matrix, (width, height))


def draw_bbox(img, contour, text=None):
    """
    Draws a bounding box around a contour and optionally labels it with text.

    Args:
        img (numpy.ndarray): Image to draw on.
        contour (numpy.ndarray): Contour points (Nx1x2 array) to compute the bounding box.
        text (str, optional): Text label to display above the bounding box.
    """
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if text is not None:
        draw_text(img, text, x, y, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1, (0,255,0))


def draw_text(img, text, x, y, font, font_scale, text_thickness, text_color):
    """
    Draws text on an image at a given position.
    """
    (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, text_thickness)
    y = max(text_height, y-10)
    cv.putText(img, text, (x, y), font, font_scale, text_color, text_thickness)


# Load Precalculated pHashes
df = pd.read_csv('phash/phashes.csv')

# Load Segmentation Model
MODEL_SEGMENT_PATH = 'instance_segmentation/models/v1.pt'
model_segment = YOLO(MODEL_SEGMENT_PATH)

# Camera Setup
VID_WIDTH, VID_HEIGHT = 1920, 1080
capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, VID_WIDTH)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, VID_HEIGHT)


while True:
    ret, frame = capture.read()
    if not ret:
        continue

    results = model_segment(frame, verbose=False)

    for result in results:
        if result.masks is not None:
            for poly in result.masks.xy:
                pts = get_corners(poly, VID_WIDTH, VID_HEIGHT)

                card_pts = np.array(pts, dtype=np.float32)
                card = isolate_card(frame, card_pts)
                card = cv.resize(card, (300, 420))
                x1, y1 = 23, 40
                x2, y2 = 277, 246
                art = card[y1:y2, x1:x2]
                art = cv.cvtColor(art, cv.COLOR_BGR2GRAY)
                art = cv.GaussianBlur(art, (3,3), cv.BORDER_DEFAULT)
                art = cv.equalizeHist(art)
                art = Image.fromarray(art).convert('RGB')
                card_hash = imagehash.phash(art, hash_size=16)
                prediction = find_card(str(card_hash))

                contour = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                draw_bbox(frame, contour, prediction)


    cv.imshow('Webcam (type q to exit)', frame)

    if cv.waitKey(20) & 0xFF==ord('q'):
        break

capture.release()
cv.destroyAllWindows()