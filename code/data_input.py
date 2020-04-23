"""
读取图像数据，切割训练测试集合
"""
from sklearn.model_selection import train_test_split
from random import randint
from keras.utils import np_utils
from tensorflow.python.keras import backend as K
import os
import numpy as np
import glob
from PIL import Image
from configure import *


class Dataset:
    def __init__(self):
        # 全部数据
        self.all_images = None
        self.all_labels = None
        # 训练集
        self.train_images = None
        self.train_labels = None
        # 验证集
        self.valid_images = None
        self.valid_labels = None
        # 测试集
        self.test_images = None
        self.test_labels = None
        # 数据集加载路径
        self.path_name = DATA_SOURCE_PATH
        # 当前库采用的维度顺序
        self.input_shape = INPUT_SHAPE
        self.nb_classes = NB_CLASS

    # 加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
    def load(self, img_rows=INPUT_HEIGHT, img_cols=INPUT_WEIGHT,
             img_channels=IMAGE_CHANNELS):
        # 加载数据集到内存
        images, labels, face_num = read_img(self.path_name)
        self.nb_classes = face_num

        all_images = images
        all_labels = labels

        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.2,
                                                                                  random_state=randint(0, 100))
        valid_images, test_images, valid_labels, test_labels = train_test_split(valid_images, valid_labels,
                                                                                test_size=0.5,
                                                                                random_state=randint(0, 100))
        # train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2,
        #                                                                         random_state=randint(0, 100))

        # 当前的维度顺序如果为'th'，则输入图片数据时的顺序为：channels,rows,cols，否则:rows,cols,channels
        # 这部分代码就是根据keras库要求的维度顺序重组训练数据集
        if K.image_data_format() == 'channels_first':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            self.input_shape = (img_channels, img_rows, img_cols)
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            self.input_shape = (img_rows, img_cols, img_channels)

        all_images = all_images.reshape(all_images.shape[0],  img_rows, img_cols, img_channels)

        # 输出训练集、验证集、测试集的数量
        print(all_images.shape[0], 'all samples')
        print(train_images.shape[0], 'train samples')
        print(valid_images.shape[0], 'valid samples')
        print(test_images.shape[0], 'test samples')

        '''
        我们的模型使用categorical_crossentropy作为损失函数，因此需要根据类别数量nb_classes将
        类别标签进行one-hot编码使其向量化
        '''

        all_labels = np_utils.to_categorical(all_labels, self.nb_classes)
        train_labels = np_utils.to_categorical(train_labels, self.nb_classes)
        valid_labels = np_utils.to_categorical(valid_labels, self.nb_classes)
        test_labels = np_utils.to_categorical(test_labels, self.nb_classes)

        # 像素数据浮点化以便归一化
        all_images = all_images.astype('float32')
        train_images = train_images.astype('float32')
        valid_images = valid_images.astype('float32')
        test_images = test_images.astype('float32')

        # 将其归一化,图像的各像素值归一化到0~1区间
        all_images /= 255
        train_images /= 255
        valid_images /= 255
        test_images /= 255

        self.all_images = all_images
        self.all_labels = all_labels
        print(self.all_images.shape, images.shape, valid_images.shape)
        self.train_images = train_images
        self.valid_images = valid_images
        self.test_images = test_images
        self.train_labels = train_labels
        self.valid_labels = valid_labels
        self.test_labels = test_labels

    def get_data(self, size=cut_size, sample_size=total_sample_size, photos_source=location,
                 nb_class=NB_CLASSES, per_pic=PIC_NO_EACH):

        imgae_shape, image = read_image(photos_source + str(1) + '/' + str(0) + '.bmp')
        print(imgae_shape)
        image = image[::size, ::size]

        dim1 = image.shape[0]
        dim2 = image.shape[1]

        print('在正在载入正样例')
        count = 0
        x_geuine_pair = np.zeros([sample_size, 2, 1, dim1, dim2])  # 2 is for pairs
        y_genuine = np.zeros([sample_size, 1])

        for i in range(nb_class):
            for j in range(int(sample_size / nb_class)):
                ind1 = 0
                ind2 = 0

                while ind1 == ind2:
                    ind1 = np.random.randint(per_pic)
                    ind2 = np.random.randint(per_pic)

                img1 = read_image(photos_source + str(i) + '/' + str(ind1) + '.bmp')[1]
                img2 = read_image(photos_source + str(i) + '/' + str(ind2) + '.bmp')[1]

                img1 = img1[::size, ::size]
                img2 = img2[::size, ::size]

                x_geuine_pair[count, 0, 0, :, :] = img1
                x_geuine_pair[count, 1, 0, :, :] = img2

                y_genuine[count] = 1
                count += 1
        print('正样例载入完毕')
        print(count)
        print('在正在载入负样例')
        count = 0
        x_imposite_pair = np.zeros([sample_size, 2, 1, dim1, dim2])
        y_imposite = np.zeros([sample_size, 1])

        for i in range(int(sample_size / per_pic)):
            for j in range(per_pic):
                while True:
                    ind1 = np.random.randint(nb_class)
                    ind2 = np.random.randint(nb_class)
                    if ind1 != ind2:
                        break

                img1 = read_image(photos_source + str(ind1) + '/' + str(j) + '.bmp')[1]
                img2 = read_image(photos_source + str(ind2) + '/' + str(j) + '.bmp')[1]

                img1 = img1[::size, ::size]
                img2 = img2[::size, ::size]

                x_imposite_pair[count, 0, 0, :, :] = img1
                x_imposite_pair[count, 1, 0, :, :] = img2

                y_imposite[count] = 0
                count += 1
        print('负样例载入完毕')
        print(count)
        all_images = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0) / 255
        all_labels = np.concatenate([y_genuine, y_imposite], axis=0)
        self.all_images = all_images
        self.all_labels = all_labels

        x_train, x_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=randint(0, 100))
        self.train_images = x_train
        self.train_labels = y_train
        self.test_images = x_test
        self.test_labels = y_test

        return imgae_shape


def read_img(photo_path):
    images = []
    labels = []
    dirs = os.listdir(photo_path)
    dirs.sort(key=int)
    face_num = len(dirs)
    for item in dirs:  # loop all directory
        #  + '\\frontal'
        print(item)
        # for pic in glob.glob(photo_path + '\\' + item + '\\frontal' + '\\*.jpg'):
        # for pic in glob.glob(location + '\\' + item + '\\*.tiff'):
        for pic in glob.glob(photo_path + '\\' + item + '\\*.jpg'):
            im = Image.open(pic).convert('L')  # open data
            im = im.resize((INPUT_HEIGHT, INPUT_WEIGHT), Image.ANTIALIAS)
            im = np.array(im)
            # if len(im.shape) == 3:
            #     r = im[:, :, 0]
            #     g = im[:, :, 1]
            #     b = im[:, :, 2]
            # images.append([r, g, b])  # save in x_test
            images.append(im)  # save in x_test
            labels.append(item)
    print(np.array(images).shape, np.array(labels).shape, face_num)
    return np.array(images), np.array(labels), face_num


def read_image(filename):
    im = Image.open(filename).convert('L')
    shape = [im.size[0], im.size[1]]
    return shape, np.array(im.resize((input_size_height, input_size_weight), Image.ANTIALIAS))
    # .reshape((input_size_height, input_size_weight))
    # reshape((128, 128))
    # resize((128, 128), Image.ANTIALIAS)
