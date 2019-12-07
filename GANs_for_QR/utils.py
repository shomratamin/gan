import cv2
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol



def get_base_file_name_with_ext(filename):
    filename = filename.replace('\\','/')
    return filename.split('/')[-1]

def find_square_contours(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for pts in contours:
        rotated_rect = cv2.minAreaRect(pts)

        center, size, angle = rotated_rect[0], rotated_rect[1], rotated_rect[2]
        if (size[0] * size[1]) > 5000 and size[0] / size[1] < 1.10 and size[0] / size[1] > 0.9:
            h, w = image.shape[:2]
            if w // 2 <= center[0]:
                filtered_contours.append(pts)

    return filtered_contours

def find_largest_contour(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    return cnt

def crop_minAreaRect(img, rect):
    # get the parameter of the small rectangle
    center, size, angle = rect[0], rect[1], rect[2]
    
    if angle < -45:
        angle = angle + 90
        size = (size[1], size[0])
    size = (size[0] * 1.05, size[1] * 1.05)
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]

    # calculate the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1)
    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (width, height),borderValue=(255,255,255))

    # now rotated rectangle becomes vertical and we crop it
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop

def auto_rotate_text_line(line_image, _file=None):
    global counter
    image = line_image.copy()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = illumination_correction(image)
    image = threshold(image)
    _structure = cv2.getStructuringElement(cv2.MORPH_RECT,(25,1))
    image = cv2.morphologyEx(image,cv2.MORPH_ERODE,_structure)
    image = cv2.morphologyEx(image,cv2.MORPH_DILATE,_structure)
    image = cv2.bitwise_not(image)

    pts = find_largest_contour(image)
    rotated_rect = cv2.minAreaRect(pts)
    output_roi = crop_minAreaRect(line_image.copy(), rotated_rect)
    # output_roi = output_roi[:,:]
    # output_roi = resize_for_ocr(output_roi)

    # cv2.drawContours(line_image,[pts],-1,(147,20,255),1)
    # cv2.imwrite('tmp/{}.jpg'.format(counter), line_image)
    # counter += 1
    # cv2.imwrite('tmp/{}.jpg'.format(counter), output_roi)
    # counter += 1
    # to replace the original file with cropped and angle corrected line
    # cv2.imwrite(_file, output_roi)

    # output_roi = illumination_correction(output_roi,False)
    # output_roi = cv2.equalizeHist(output_roi)
    # output_roi = threshold(output_roi)
    return output_roi

def threshold(image, keep_channels=False):
    up_convert = False
    if len(image.shape) > 2:
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        up_convert = True
    image = cv2.threshold(image,128,255,cv2.THRESH_OTSU)[1]
    if up_convert and keep_channels:
        image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)

    return image


def illumination_correction(image, kernel=(115,115)):
    structure = cv2.getStructuringElement(cv2.MORPH_RECT,kernel)
    image = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT,structure)
    image = cv2.bitwise_not(image)
    return image


def dilate(image, kernel=(5,5)):
    structure = cv2.getStructuringElement(cv2.MORPH_RECT,kernel)
    image = cv2.morphologyEx(image,cv2.MORPH_DILATE,structure)
    return image


def erode(image, kernel=(5,5)):
    structure = cv2.getStructuringElement(cv2.MORPH_RECT,kernel)
    image = cv2.morphologyEx(image,cv2.MORPH_ERODE,structure)
    return image



def decode_qr(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    qrs = decode(gray_img)
    if len(qrs) < 1:
        return None

    data = qrs[0].data.decode("utf-8")
    return data