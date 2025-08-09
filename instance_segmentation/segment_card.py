import cv2 as cv
from ultralytics import YOLO
import numpy as np

def get_corners(coords, grid_x, grid_y):
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


MODEL_PATH = 'instance_segmentation/models/v1.pt'
model = YOLO(MODEL_PATH)

VID_WIDTH, VID_HEIGHT = 1920, 1080
capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, VID_WIDTH)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, VID_HEIGHT)

while True:
    ret, frame = capture.read()
    if not ret:
        continue

    results = model(frame, verbose=False)

    for result in results:
        if result.masks is not None:
            for poly in result.masks.xy:
                pts = get_corners(poly, VID_WIDTH, VID_HEIGHT)
                pts1 = np.array(pts, dtype=np.float32)
                pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv.polylines(frame, [pts], isClosed=True, color=(255,0,0), thickness = 2)
                width, height = 672, 936
                pts2 = np.float32([
                    [0, 0],
                    [width, 0],
                    [width, height],
                    [0, height]
                ])
                matrix = cv.getPerspectiveTransform(pts1, pts2)
                card = cv.warpPerspective(frame, matrix, (width, height))
                cv.imshow('card', card)

    cv.imshow('video', frame)

    if cv.waitKey(20) & 0xFF==ord('q'):
        break


cv.waitKey(20)
cv.destroyAllWindows()