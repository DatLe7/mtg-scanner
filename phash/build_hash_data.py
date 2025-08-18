import imagehash
import cv2 as cv
from PIL import Image
import pandas as pd
import json
import csv

df = pd.read_csv('mtgdb.csv')

fields = ['phash', 'name']
csv_data = []

with open('phash/id_2_name.json', 'r') as f:
    id_2_name = json.load(f)

for row in df.itertuples():
    name = id_2_name[row.id]
    img = cv.imread(row.path)
    img = cv.resize(img, (300, 420))
    x1, y1 = 23, 40
    x2, y2 = 277, 246
    art = img[y1:y2, x1:x2]
    gray = cv.cvtColor(art, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (3,3), cv.BORDER_DEFAULT)
    equalized = cv.equalizeHist(blurred)
    input_img = Image.fromarray(equalized).convert('RGB')
    phash = imagehash.phash(input_img, hash_size=16)
    csv_data.append([phash, name])

with open('phash/phashes.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(csv_data)

