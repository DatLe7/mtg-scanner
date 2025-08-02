import cv2 as cv
import numpy as np
from torch import nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from torch import nn
import torch
import json

    

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

with open('idx_2_label.json', 'r') as f:
    idx_2_label = json.load(f)
with open('id_2_name.json', 'r') as f:
    id_2_name = json.load(f)

checkpoint = torch.load('checkpoint/v2checkpoint#9.pth')
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


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


capture = cv.VideoCapture(0)
capture.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
capture.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)

text = None
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 0.7
text_thickness = 1
text_color = (0, 255, 0)

card_cnt = None


while True:
    istrue, frame = capture.read()
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    frame_copy = frame.copy()
    gray = cv.cvtColor(frame_copy, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT) 
    equalized = cv.equalizeHist(blur)
    background = cv.morphologyEx(equalized, cv.MORPH_DILATE, kernel)
    no_shadow = cv.subtract(background, equalized)
    edges = cv.Canny(no_shadow, 150, 175)
    dilated = cv.dilate(edges, kernel, iterations=1)
    closed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel)

    contours, hierarchy = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    blank = np.zeros(frame.shape, dtype='uint8')
    filtered_contours = []

    for cnt in contours:
        area = cv.contourArea(cnt)
        perimeter = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) == 4 and area > 10000:
            filtered_contours.append(cnt)
    
    if filtered_contours:
        card_cnt = filtered_contours[0]

    if card_cnt is not None:
        x, y, w, h = cv.boundingRect(card_cnt)
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        perimeter = cv.arcLength(card_cnt, True)
        corners = cv.approxPolyDP(card_cnt, 0.02 * perimeter, True)
        pts1 = corners.reshape(4, 2).astype(np.float32)
        pts1 = order_points(pts1)
        width, height = 500, 700
        pts2 = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])
        matrix = cv.getPerspectiveTransform(pts1, pts2)
        result = cv.warpPerspective(frame, matrix, (width, height))
        prediction = classify_card(result)
        if prediction is not None:
            print(prediction)
            text = prediction
            
    if text is not None:
        (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, text_thickness)
        text_x = x
        text_y = y - 10
        text_y = max(text_height, text_y)
        cv.putText(frame, text, (text_x, text_y), font, font_scale, text_color, text_thickness)

    cv.imshow('Webcam (type q to exit)', frame)

    if cv.waitKey(20) & 0xFF==ord('q'):
        break

capture.release()
cv.destroyAllWindows()