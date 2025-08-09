from PIL import Image, ImageDraw
import math
import random

'''REFERENCE: https://math.stackexchange.com/questions/270194/how-to-find-the-vertices-angle-after-rotation'''
def rotate(x, y, rot_angle, center_x, center_y):
    x -= center_x
    y -= center_y
    x_new = x * math.cos(rot_angle) - y * math.sin(rot_angle)
    y_new = x * math.sin(rot_angle) + y * math.cos(rot_angle)
    return x_new + center_x, y_new + center_y


def create_labeled_image(background_path, card_path, img_save_path, txt_save_path):
    scale_factor = random.randint(2, 5)
    angle = random.randint(-15,15)
    angle_rad = math.radians(-angle)
    BACKGROUND_SIZE = (640, 640)
    no_instance = random.randint(1, 100)
    if no_instance <= 10:
        background = Image.open(background_path).convert('RGB').resize(BACKGROUND_SIZE, resample=Image.LANCZOS)
        background.save(img_save_path)
        with open(txt_save_path, 'w') as f:
            f.write('')
        return None

    background = Image.open(background_path).convert('RGBA').resize(BACKGROUND_SIZE, resample=Image.LANCZOS)
    img = Image.open(card_path).convert('RGBA')

    w, h = img.size
    rescaled_w = w // scale_factor
    rescaled_h = h // scale_factor
    img = img.resize((rescaled_w, rescaled_h), resample=Image.LANCZOS)
    img = img.rotate(angle, expand=True)
    center_x, center_y = img.size

    match (scale_factor):
        case 2:
            place_x, place_y = 150, 80
        case 3:
            offset_x, offset_y = random.randint(-100,100), random.randint(-100,100)
            place_x, place_y = 200 + offset_x, 150 + offset_y
        case 4:
            offset_x, offset_y = random.randint(-150,150), random.randint(-150,150)
            place_x, place_y = 250 + offset_x, 200 + offset_y
        case 5:
            offset_x, offset_y = random.randint(-150,150), random.randint(-150,150)
            place_x, place_y = 275 + offset_x, 225 + offset_y

    pts = [
        (place_x, place_y),
        (place_x+rescaled_w, place_y),
        (place_x+rescaled_w, place_y+rescaled_h),
        (place_x, place_y+rescaled_h)
    ]
    rot_pts = [rotate(pt[0], pt[1], angle_rad, center_x, center_y) for pt in pts]
    norm_coords = []
    for pt in rot_pts:
        for coord in pt:
            norm_coords.append(round(coord / BACKGROUND_SIZE[0], 6))
            

    if (angle < 0):
        point = (int(rot_pts[0][0] - (rot_pts[1][0] - rot_pts[2][0])), int(rot_pts[0][1]))
    elif (angle > 0):
        point = (int(rot_pts[0][0]), int(rot_pts[0][1] - (rot_pts[0][1] - rot_pts[1][1])))
    else:
        point = (int(rot_pts[0][0]), int(rot_pts[0][1]))
    Image.Image.paste(background, img, point, mask=img)

    background.convert('RGB').save(img_save_path)
    with open(txt_save_path, 'w') as f:
        f.write(' '.join(map(str, [0] + norm_coords)))
    

# 706 total backgrounds 1 index
# 398 total cards 0 index
for i in range(0, 1800): # test
    background = random.randint(1, 706)
    card = random.randint(0, 397)
    create_labeled_image(
        background_path=f'instance_segmentation/backgrounds/image-{background}.jpg', 
        card_path=f'images/card{card}.jpg', 
        img_save_path=f'instance_segmentation/card_dataset/images/train/card{i}.jpg', 
        txt_save_path=f'instance_segmentation/card_dataset/labels/train/card{i}.txt')
for i in range(1800, 2000): # val
    background = random.randint(1, 706)
    card = random.randint(0, 397)
    create_labeled_image(
        background_path=f'instance_segmentation/backgrounds/image-{background}.jpg', 
        card_path=f'images/card{card}.jpg', 
        img_save_path=f'instance_segmentation/card_dataset/images/val/card{i}.jpg', 
        txt_save_path=f'instance_segmentation/card_dataset/labels/val/card{i}.txt')