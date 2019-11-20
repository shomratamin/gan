import cv2
from glob import glob


# gt = glob('Ground_Truth/*.jpg')
# rep1 = glob('484848_1_2/*.jpg')
# rep2 = glob('484848_1_4/*.jpg')
# rep3 = glob('484848_1_4_T/*.jpg')

folder_maps = dict()
folder_maps['GT_arial'] = ['arial_12_686868_pic','arial_14_282828_gauss']
folder_maps['GT_arial_i'] = ['arial_i_13_484848_pic','arial_i_13_585858_gauss']
folder_maps['GT_calibri'] = ['calibri_12_383838_pic','calibri_14_282828_gauss']
folder_maps['GT_calibri_b'] = ['calibri_b_13_484848_pic','calibri_b_14_484848_gauss']
folder_maps['GT_times_i'] = ['times_i_11_383838_pic','times_i_13_383838_gass']

for k, folder in enumerate(folder_maps):
    gt = glob('{}/*.jpg'.format(folder))
    folder_map = folder_maps[folder]
    gen_data_list = []

    for f in folder_map:
        gen = glob('{}/*.jpg'.format(f))
        gen_data_list.append(gen)
    gen_list_len = len(folder_map)

    for i,img in enumerate(gt):
        img_a = img
        img_b = img
        key_word = img.split('_')[-1]
        key_word = '_' + key_word
        gen_files = gen_data_list[i % gen_list_len]
        for _gen_file in gen_files:
            if _gen_file.endswith(key_word):
                img_b = _gen_file
                break

        print(img_a, img_b)

        img_a = cv2.imread(img_a)
        img_b = cv2.imread(img_b)
        train_test = i % 10
        if train_test == 0:
            cv2.imwrite('test/gt/{}_{}.jpg'.format(k,i),img_a)
            cv2.imwrite('test/gen/{}_{}.jpg'.format(k,i),img_b)
        else:
            cv2.imwrite('train/gt/{}_{}.jpg'.format(k,i),img_a)
            cv2.imwrite('train/gen/{}_{}.jpg'.format(k,i),img_b)