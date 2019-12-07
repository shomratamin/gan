import qrcode
from PIL import ImageFont
from PIL import ImageDraw, Image, ImageColor, ImageFilter, ImageEnhance
import numpy as np
import cv2
from tqdm import tqdm
import random
import csv

def hex_to_rgb(hex):
    return '#{}'.format(hex) 

def read_colors():
    colors = []
    with open('colors.csv', 'r', encoding='utf-8') as f:
        color_csv = csv.reader(f)
        
        i = 0
        for color in color_csv:
            if i > 0:
                colors.append([color[2],color[6]])
                colors.append([color[3],color[6]])
                colors.append([color[4],color[6]])
                colors.append([color[5],color[6]])
            i += 1
    for i, color in enumerate(colors):
        colors[i] = [hex_to_rgb(color[0]), hex_to_rgb(color[1])]
    print(colors)
    return colors

def shuffle_string(text):
    l = list(text)
    random.shuffle(l)
    result = ''.join(l)
    return result

# def get_qr_segments(img_orig, img_mod):

def random_distort(image,i=0, prob = 5):
    if i % prob == 0:
        image = image.filter(ImageFilter.GaussianBlur(random.randint(1,2)))
    # elif i % prob == 1:
    #     wh = random.randint(150,200)
    #     image = image.resize((wh,wh))
    #     image = image.resize((256,256))
    elif i % prob == 2:
        # wh = random.randint(100,200)
        # image = image.resize((wh,wh))
        # image = image.resize((256,256))
        image = image.filter(ImageFilter.GaussianBlur(random.randint(1,2)))
    elif i % prob == 3:
        image = image.filter(ImageFilter.GaussianBlur(random.randint(1,2)))
        # wh = random.randint(100,200)
        # image = image.resize((wh,wh))
        # image = image.resize((256,256))
    elif i % prob == 4:
        image = image.filter(ImageFilter.BoxBlur(random.randint(1,5)))


    return image

qr = qrcode.QRCode(
    version=4,
    error_correction=qrcode.constants.ERROR_CORRECT_Q,
    box_size=8,
    border=4,
)

colors = read_colors()
total_colors = len(colors)
images = []
counter = 1
img_wh = 650
text = '<?xml version="1.0" encoding="UTF-8"?>\
<PrintLetterBarcodeData uid="310977181643" name="Mohan Paswan" gender="M" yob="1967" gname="Ramdev Paswan" house="-" street="-" loc="Marwatoli" vtc="Basatpur Marua Toli" po="Chakalaghat" dist="Kishanganj" subdist="Dighalbank" state="Bihar" pc="855107" dob="15/01/1967"/>'
counter = 1
for i in tqdm(range(1,1000,1), desc="generating qrcode"):
    qr.clear()
    text = shuffle_string(text)
    qr.add_data(text)
    qr.make(fit=True)
    color_pair = colors[i % total_colors]
    img_mod = qr.make_image(fill_color=color_pair[0], back_color=color_pair[1])
    img_orig = qr.make_image(fill_color='#000000', back_color='#ffffff')

    img_orig = img_orig.resize((512,512))
    img_mod = img_mod.resize((512,512))
    img_mod = random_distort(img_mod,i)
    img_orig.save(f'datasets/gt/{counter}.jpg')
    img_mod.save(f'datasets/gen/{counter}.jpg')
    counter += 1

# print('total qr codes', len(images))
# col = 0
# row = 0
# max_row = 5
# max_col = 4
# full_image_width = img_wh * max_col
# full_image_height = img_wh * max_row
# new_tile_image = np.ones((full_image_height,full_image_width,3),dtype=np.int8)
# new_tile_image = new_tile_image * 255
# for i in range(len(images)):
#     if i > 0 and i % 20 == 0:
#         cv2.imwrite(f'tmp/{counter}.jpg', new_tile_image)
#         counter += 1
#         new_tile_image = np.ones((full_image_height,full_image_width,3),dtype=np.int8)
#         new_tile_image = new_tile_image * 255
#         col = 0
#         row = 0
#     col_s = col * img_wh
#     col_e = (col + 1) * img_wh
#     row_s = row * img_wh
#     row_e = (row + 1) * img_wh
#     print(row_s,row_e,':',col_s,col_e)
#     new_tile_image[row_s:row_e,col_s:col_e,:] = images[i]
#     col += 1

#     if col > max_col - 1:
#         col = 0
#         row += 1
#     if row > max_row - 1:
#         row = 0
#         col = 0
#     if i == len(images) - 1:
#         cv2.imwrite(f'tmp/{counter}.jpg', new_tile_image)
#         counter += 1
