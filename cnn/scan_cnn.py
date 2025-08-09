import cv2 as cv
import numpy as np
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from torch import nn
import torch
import json
from ultralytics import YOLO


def classify_card(img):
    model.eval()
    img = Image.fromarray(img).convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        outputs = model(img)
    probs = torch.softmax(outputs, dim=1)
    pred_prob = probs.max().item()
    if (pred_prob > 0.8):
        pred_idx = torch.argmax(outputs, dim=1).item()
        # print(f"Top class: " + idx_2_label[str(pred_idx)] + ", Prob: {pred_prob:.4f}")
        id = idx_2_label[str(pred_idx)]
        return id_2_name[id]
    return None


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


def isolate_card(img, card_pts):
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
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if text is not None:
        draw_text(img, text, x, y, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1, (0,255,0))


def draw_text(img, text, x, y, font, font_scale, text_thickness, text_color):
    (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, text_thickness)
    y = max(text_height, y-10)
    cv.putText(img, text, (x, y), font, font_scale, text_color, text_thickness)



with open('cnn/idx_2_label.json', 'r') as f:
    idx_2_label = json.load(f)
with open('cnn/id_2_name.json', 'r') as f:
    id_2_name = json.load(f)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

checkpoint = torch.load('cnn/models/v1.pth')
model = resnet18()
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.5),
    nn.Linear(512, 528)
)
model.load_state_dict(checkpoint['model_state_dict'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


MODEL_SEGMENT_PATH = 'instance_segmentation/models/v1.pt'
model_segment = YOLO(MODEL_SEGMENT_PATH)

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
                prediction = classify_card(card)

                contour = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                draw_bbox(frame, contour, prediction)
            

    cv.imshow('Webcam (type q to exit)', frame)

    if cv.waitKey(20) & 0xFF==ord('q'):
        break

capture.release()
cv.destroyAllWindows()