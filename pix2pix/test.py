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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def resize_scaled(image, height = 32):
    h, w = image.shape[:2]
    new_width = int((height * w)/h)
    image = cv2.resize(image,(new_width,height),interpolation=cv2.INTER_CUBIC)
    return image

def chop_or_pad(image, width = 256, height = 32):
    h, w = image.shape[:2]
    if h != height:
        image = resize_scaled(image,height)
    h, w = image.shape[:2]

    if w < width:
        pad = width - w
        image = cv2.copyMakeBorder(image,0,0,0,pad, cv2.BORDER_CONSTANT,value=(255, 255, 255))
    elif w > width:
        image = image[:,:width,:]

    image = image.astype(np.float)

    return image

def build_generator():
    """U-Net Generator"""
    gf = 64
    channels = 3
    img_shape = (32,256,channels)


    def conv2d(layer_input, filters, f_size=4, bn=True,stride=2,dropout_rate=0):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=stride, padding='same')(layer_input)
        if dropout_rate:
            u = Dropout(dropout_rate)(d)
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
        K.int_shape(u)
        K.int_shape(skip_input)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=img_shape)

    # Downsampling
    d1 = conv2d(d0, gf, bn=False, stride=2)
    d2 = conv2d(d1, gf*2,stride=2)
    d3 = conv2d(d2, gf*4,stride=2)
    d4 = conv2d(d3, gf*8,stride=2)
    d5 = conv2d(d4, gf*8,stride=2)
    # d6 = conv2d(d5, gf*8,stride=2)
    # d7 = conv2d(d6, gf*8,stride=2)

    # Upsampling
    u1 = deconv2d(d5, d4, gf*8)
    u2 = deconv2d(u1, d3, gf*8)
    u3 = deconv2d(u2, d2, gf*8)
    u4 = deconv2d(u3, d1, gf*4)
    # u5 = deconv2d(u4, d1, gf*2)
    # u6 = deconv2d(u5, d1, gf)
    u6 = u4

    u7 = UpSampling2D(size=2)(u6)
    output_img = Conv2D(channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
    # print('generator output', output_img.shape)
    return Model(d0, output_img)


weights_path = 'saved_model/gen_10.h5'
generator = build_generator()
generator.load_weights(weights_path)


images = glob('test/*.jpg')
images.extend(glob('test/*.png'))

t1 = time()
for i, _image in enumerate(images):
    print(_image)
    image = cv2.imread(_image)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # image = cv2.GaussianBlur(image,(9,9),0)
    image = cv2.blur(image,(7,7))
    image = chop_or_pad(image)
    input_images = []
    input_images.append(image)
    input_images = np.array(input_images)/127.5 - 1.
    tt = time()
    output_image = generator.predict(input_images)
    print('time taken', time()-tt)
    output_image = 255.0 * output_image
    output_image = output_image[0]
    output_image = cv2.cvtColor(output_image,cv2.COLOR_RGB2BGR)

    output_tile = np.zeros((64,256,3))
    output_tile[:32,:,:] = image
    output_tile[32:,:,:] = output_image
    # print(output_image)
    cv2.imwrite(f'out/{i}.jpg', output_tile)

print('time taken',time() - t1)