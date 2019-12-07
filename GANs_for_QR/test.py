from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import sys
import numpy as np
import os
import keras.backend as K
from glob import glob
import cv2
from time import time
import utils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def resize_scaled(image, height = 32):
    h, w = image.shape[:2]
    new_width = int((height * w)/h)
    image = cv2.resize(image,(new_width,height),interpolation=cv2.INTER_CUBIC)
    return image

def chop_or_pad(image, width = 256, height = 256):
    h, w = image.shape[:2]
    pad_w = width - w
    pad_h = height - h

    image = cv2.copyMakeBorder(image,0,pad_h,0,pad_w, cv2.BORDER_CONSTANT,value=(255, 255, 255))


    image = image.astype(np.float)

    return image, w

def build_generator():
    """U-Net Generator"""
    gf = 64
    channels = 3
    img_shape = (512,512,channels)

    def conv2d(layer_input, filters, f_size=4, bn=True):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = BatchNormalization(momentum=0.8)(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=img_shape)

    # Downsampling
    d1 = conv2d(d0, gf, bn=False)
    d2 = conv2d(d1, gf*2)
    d3 = conv2d(d2, gf*4)
    d4 = conv2d(d3, gf*8)
    d5 = conv2d(d4, gf*8)
    d6 = conv2d(d5, gf*8)
    d7 = conv2d(d6, gf*8)

    # Upsampling
    u1 = deconv2d(d7, d6, gf*8)
    u2 = deconv2d(u1, d5, gf*8)
    u3 = deconv2d(u2, d4, gf*8)
    u4 = deconv2d(u3, d3, gf*4)
    u5 = deconv2d(u4, d2, gf*2)
    u6 = deconv2d(u5, d1, gf)

    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

    return Model(d0, output_img)


weights_path = 'saved_model/gen_2.h5'
generator = build_generator()
generator.load_weights(weights_path)


images = glob('test/*.png')
# images = glob('test/*.png')
# images.extend(glob('E:/Projects/ocr-eng/clean_dataset/*.png'))
print('total images', len(images))


def detect_qr_code(image):
    orig_image = image.copy()
    image = utils.illumination_correction(image)
    image = utils.threshold(image)
    image = utils.erode(image)
    squares = utils.find_square_contours(image)
    if len(squares) > 0:
        rotated_rect = cv2.minAreaRect(squares[0])
        qr_code = utils.crop_minAreaRect(orig_image, rotated_rect)
        qr_code = cv2.resize(qr_code,(512,512), interpolation=cv2.INTER_CUBIC)
        return qr_code
    else:
        return None
    # cv2.drawContours(orig_image,squares,-1,(0,255,0), 2, cv2.LINE_8)

    # return orig_image





def enhance_qr():
    t1 = time()
    qr_before = 0
    qr_after = 0
    qr_none = 0
    total_image = 0
    no_qr_found = 0
    for i, _image in enumerate(images):
        if i % 100 == 0:
            print('processed', i)
        __image = cv2.imread(_image)
        if __image is None:
            continue
        tt = time()
        image = detect_qr_code(__image.copy())
        if image is None:
            no_qr_found += 1
            continue
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # image = utils.illumination_correction(image)
        # image = cv2.GaussianBlur(image,(9,9),0)
        # image = cv2.blur(image,(3,3))
        # image, orig_width = chop_or_pad(image)
        input_images = []
        input_images.append(image)
        input_images = np.array(input_images)/127.5 - 1.

        ting = time()
        output_image = generator.predict(input_images)
        ting2 = time() - ting
        print("printing ting2 *****************************************", ting2)


        output_image = output_image + 1.0
        output_image = 127.5 * output_image
        output_image = output_image[0]
        try:
            output_qr_b = utils.decode_qr(__image)
            print("printing second stage -----------------", output_qr_b)
        except:
            continue
        if output_qr_b is not None:
            qr_before += 1
        try:
            output_qr_a = utils.decode_qr(output_image)
            print("printing final stage -----------------", output_qr_a)
        except:
            continue
        if output_qr_a is not None and output_qr_b is None:
            qr_after += 1
        if output_qr_a is None and output_qr_b is None:
            qr_none += 1

        total_image += 1
        print('time taken', time()-tt)
        # output_image = cv2.cvtColor(output_image,cv2.COLOR_RGB2BGR)
        # # output_image = output_image[:,:orig_width,:]

        # output_tile = np.zeros((1024,512,3))
        # output_tile[:512,:,:] = image
        # output_tile[512:,:,:] = output_image
        # print(output_image)
        # cv2.imwrite(_image, output_image)
        # cv2.imwrite(f'out/{i}.jpg', output_tile)
        # cv2.imwrite(f'out/gt/{i}.jpg', __image)
        # cv2.imwrite(f'out/gen/{i}.jpg', output_image)
        print(f'normal: {qr_before} gan : {qr_after} none : {qr_none} no_qr: {no_qr_found} total : {total_image}')


    print('time taken',time() - t1)


if __name__ == "__main__":
    enhance_qr()
    # for i, _image in enumerate(images):
    #     image = cv2.imread(_image)
    #     image = detect_qr_code(image)
    #     image_file_name = utils.get_base_file_name_with_ext(_image)
    #     print(image_file_name)
    #     if image is not None:
    #         cv2.imwrite(f'out/{image_file_name}', image)
