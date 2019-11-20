import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random




class DataLoader():
    def __init__(self, dataset_name, img_res=(32, 512)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def chop_or_pad(self,image):
        h, w = image.shape[:2]
        if w < self.img_res[1]:
            pad = self.img_res[1] - w
            image = cv2.copyMakeBorder(image,0,0,0,pad, cv2.BORDER_CONSTANT,value=(255, 255, 255))
        elif w > self.img_res[1]:
            image = image[:,:self.img_res[1],:]

        image = image.astype(np.float)

        return image

    def load_data(self, batch_size=1, is_testing=False):
        gt_path = glob('{}/test/gt/*.png'.format(self.dataset_name))
        gt_path.extend(glob('{}/test/gt/*.jpg'.format(self.dataset_name)))
        random.shuffle(gt_path)
        pairs = []
        for f in gt_path[:500]:
            f = f.replace('\\','/')
            base_name = f.split('/')[-1]
            # base_name = base_name.replace('_GT_','_GT_IN_')
            ch_file = '{}/test/gen/{}'.format(self.dataset_name,base_name)
            if os.path.exists(ch_file):
                pairs.append([f,ch_file])


        batch_images = np.random.choice(len(pairs) - 1, size=batch_size)

        imgs_A = []
        imgs_B = []
        for i in batch_images:
            img_a = cv2.imread(pairs[i][0])
            img_a = cv2.cvtColor(img_a,cv2.COLOR_BGR2RGB)
            img_a = self.chop_or_pad(img_a)
            img_b = cv2.imread(pairs[i][1])
            img_b = cv2.cvtColor(img_b,cv2.COLOR_BGR2RGB)
            img_b = self.chop_or_pad(img_b)


            img_A = img_a
            img_B = img_b

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A)/127.5 - 1.
        imgs_B = np.array(imgs_B)/127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        gt_path = glob('{}/train/gt/*.png'.format(self.dataset_name))
        gt_path.extend(glob('{}/train/gt/*.jpg'.format(self.dataset_name)))
        random.shuffle(gt_path)
        pairs = []
        for f in gt_path:
            f = f.replace('\\','/')
            base_name = f.split('/')[-1]
            # base_name = base_name.replace('_GT_','_GT_IN_')
            ch_file = '{}/train/gen/{}'.format(self.dataset_name,base_name)
            if os.path.exists(ch_file):
                pairs.append([f,ch_file])

        self.n_batches = int(len(pairs) / batch_size)

        for i in range(self.n_batches-1):
            batch = pairs[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img_path in batch:
                img_a = cv2.imread(img_path[0])
                img_a = cv2.cvtColor(img_a,cv2.COLOR_BGR2RGB)
                img_a = self.chop_or_pad(img_a)
                img_b = cv2.imread(img_path[1])
                img_b = cv2.cvtColor(img_b,cv2.COLOR_BGR2RGB)
                img_b = self.chop_or_pad(img_b)


                img_A = img_a
                img_B = img_b

                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.

            yield imgs_A, imgs_B


    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
