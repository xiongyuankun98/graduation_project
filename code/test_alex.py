import os
import csv
import random
import time
from functools import wraps
import pandas as pd
from configure import *
import itertools
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from data_input import *
from keras.models import load_model
from model_train_2 import SiameseNet
from keras.utils.vis_utils import plot_model
import streamlit as st
import math
import cv2 as cv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('max_colwidth', 200)
from insightface import amsoftmax_loss, AMSoftmax
from data_input import Dataset


class Alextest(object):
    def __init__(self):
        self.model = None

    def load_model(self):
        print('正在载入人脸验证模型及参数')
        self.model = load_model(alexnet_model_path, custom_objects={'AMSoftmax': AMSoftmax,'amsoftmax_loss': amsoftmax_loss})
        # custom_objects={'AMSoftmax': AMSoftmax,'amsoftmax_loss': amsoftmax_loss}
        self.model.load_weights(alexnet_weight_model)
        print('载入完毕')
        # print(self.model.metrics_names)

    def rand_test(self, photo_source=webface_path, face_num=3, nb_class=NB_CLASS):
        pic_list = []
        pic_info = []
        prob_list = []
        prediction = []
        rand_index_1 = random.sample(range(0, nb_class), face_num)
        rand_index_2 = np.random.randint(0, 150, 50)
        iter_rand_index_2 = iter(rand_index_2)
        # print(rand_index_1)
        for item in rand_index_1:
            while True:
                sub = next(iter_rand_index_2)
                #path = photo_source + '\\' + str(item).zfill(2) + '\\' + str(sub) + '.jpg'
                path = photo_source + '\\' + str(item).zfill(4) + '\\' + str(sub).zfill(3) + '.jpg'
                if os.path.exists(path):
                    # print(path)
                    pic_list.append(path)
                    pic_info.append([item, sub])
                    break
        for item in pic_list:
            prob_singel = []
            im = Image.open(item).convert('L')
            im = im.resize((INPUT_HEIGHT, INPUT_WEIGHT), Image.ANTIALIAS)
            im = np.array(im)
            im = im.astype('float32')
            im /= 255
            result = im.reshape(1, 227, 227, 1)
            result = self.model.predict(result)
            temp = result.copy()
            biu = np.argsort(temp)
            tt = []
            biu = biu.tolist()[0][::-1][0:5]  # 概率值从大到小排前五序号index
            temp = temp.tolist()[0]
            for it in biu:
                tt.append(temp[int(it)])
            summ = sum(tt[0:5])
            prob_singel.append([[tt[0]/summ, tt[1]/summ, tt[2]/summ, tt[3]/summ, tt[4]/summ], biu])
            prob_list.append(prob_singel)

        for item in prob_list:
            temp_list = []
            for it in range(5):
                temp_list.append(['{}: {}%'.format(item[0][1][it], round(100*item[0][0][it], 2))])
            prediction.append(temp_list)
        max_columns_num = math.ceil(face_num / 2)
        min_margin = min(max_columns_num * 4.5, 18)
        if face_num != 1:
            figure = plt.figure(figsize=(5, min_margin))
            plt.subplots_adjust(left=0.05, bottom=0.3, right=1, wspace=0.25, hspace=0, top=0.97)
        else:
            figure = plt.figure(figsize=(3, 2.5))
            plt.subplots_adjust(left=0.05, bottom=0.1, right=1, wspace=0.25, hspace=0, top=0.9)
        # figure.canvas.set_window_title('单人 ' + str(face_num) + ' 组测试, 左下为Top-5概率值')
        for i in range(face_num):
            print('正在绘制···第' + str(i + 1) + '张图像/共计' + str(face_num) + '张')
            img = np.array(Image.open(pic_list[i]).resize((512, 512), Image.ANTIALIAS))
            plt.subplot(max_columns_num, 2, i + 1)
            plt.xlabel('Name:' + str(pic_info[i][0]) + '\nsub:' + str(pic_info[i][1]))
            margin = 15
            cv.rectangle(img, (0, 385), (150, 512), (255, 255, 255), -1)
            for item in range(5):
                textLabel = str(prediction[i][item]).replace("'", '').replace('[', '').replace(']', '')
                cv.putText(img, textLabel, (0, 388+margin), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
                # cv.FONT_HERSHEY_COMPLEX_SMALL
                margin += 25

            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.axis('on')  # 关掉坐标轴为 off
            plt.title('image_' + str(i + 1))  # 图像题目

        # plt.show()
        return figure, prediction

    def confusion_matrix(self):
        dataset = Dataset()
        dataset.load()
        y_pred = self.model.predict(dataset.all_images)
        rounded_labels = np.argmax(dataset.all_labels, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        from sklearn.metrics import confusion_matrix
        cfm = confusion_matrix(rounded_labels, y_pred)
        print(cfm)
        plt.figure(figsize=(6, 6))
        plt.matshow(cfm, cmap=plt.cm.gray)
        plt.matshow(cfm, cmap=plt.cm.Blues)
        row_sum = np.sum(cfm, axis=1)
        err_matrix = cfm / row_sum
        np.fill_diagonal(err_matrix, 0)
        plt.matshow(err_matrix, cmap=plt.cm.gray)
        plt.matshow(err_matrix, cmap=plt.cm.Blues)
        plt.show()


    # def show_confusion_matrix(self, dataset_name):
    #     figure = plt.figure(figsize=(6, 6))
    #     plt.subplots_adjust(left=0.05, bottom=0.2, right=0.95, wspace=0.25, hspace=0.25)
    #
    #     if dataset_name == 'cisia-webface':
    #         plt.subplot(2, 2, 1)
    #         img = Image.open(pic_list[i]).convert('LA')
    #         plt.imshow(img)

    def one_test(self,uploadfile):
        prob_list = []
        prob_singel = []
        prediction = []
        im = uploadfile.convert('L')
        im = np.array(im)
        im = im.astype('float32')
        im /= 255
        result = im.reshape(1, 227, 227, 1)
        result = self.model.predict(result)
        print(np.argmax(result, axis=1))
        temp = result.copy()
        biu = np.argsort(temp)
        tt = []
        biu = biu.tolist()[0][::-1][0:5]  # 概率值从大到小排前五序号index
        temp = temp.tolist()[0]
        for it in biu:
            tt.append(temp[int(it)])
        summ = sum(tt[0:5])
        prob_singel.append([[tt[0] / summ, tt[1] / summ, tt[2] / summ, tt[3] / summ, tt[4] / summ], biu])
        prob_list.append(prob_singel)

        for item in prob_list:
            temp_list = []
            for it in range(5):
                temp_list.append(['{}: {}%'.format(item[0][1][it], round(100*item[0][0][it], 2))])
            prediction.append(temp_list)

        figure = plt.figure(figsize=(3, 2.5))
        plt.subplots_adjust(left=0.05, bottom=0.1, right=1, wspace=0.25, hspace=0, top=0.9)
            # figure.canvas.set_window_title('单人 ' + str(face_num) + ' 组测试, 左下为Top-5概率值')
        img = np.array(uploadfile.resize((512, 512), Image.ANTIALIAS))
        margin = 15
        cv.rectangle(img, (0, 385), (150, 512), (255, 255, 255), -1)
        for item in range(5):
            textLabel = str(prediction[0][item]).replace("'", '').replace('[', '').replace(']', '')
            cv.putText(img, textLabel, (0, 388 + margin), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
            # cv.FONT_HERSHEY_COMPLEX_SMALL
            margin += 25
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.axis('on')  # 关掉坐标轴为 off
        plt.title('image')  # 图像题目
        # plt.show()
        return figure, prediction

# test = Alextest()
# test.load_model()
# # # test.confusion_matrix()
# dataset = Dataset()
# dataset.load()
# # print(dataset.test_labels.shape)
# score = test.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
# print("{}:{:.2f}".format(test.model.metrics_names[0], score[0]))
# print("%s: %.2f%%" % (test.model.metrics_names[1], score[1] * 100))
# print("%s: %.2f%%" % (test.model.metrics_names[2], score[2] * 100))

# # test.confusion_matrix()
# plot_model(test.model, to_file="model.png", show_shapes=True)
# test.rand_test(face_num=6)

# dirs = os.listdir(webface_path)
# index = 0
# totalnum = 0
# all_predict = []
# for item in dirs:  # loop all directory
#     temp = []
#     temp_list = []
#     for pic in glob.glob(webface_path + '\\' + item + '\\*.jpg'):
#         im = Image.open(pic).convert('L')
#         im = im.resize((INPUT_HEIGHT, INPUT_WEIGHT), Image.ANTIALIAS)
#         im = np.array(im)
#         temp.append(im)
#     print(item)
#     number = len(temp)
#     for it in temp:
#         dataset = np.array(it)
#         train_images = dataset.astype('float32')
#         train_images /= 255
#         result = train_images.reshape(number, 227, 227, 1)
#         result = np.argmax(test.model.predict(result), axis=1)
#         tt = result.tolist()
#         temp_list.append(tt)
#     all_predict.append()
#
# model = load_model(r'D:\For_Python\SomethingINT\holiday\mytry\normal_try\result\webface\1585491382.721277\model.h5',
#                    custom_objects={'AMSoftmax': AMSoftmax, 'amsoftmax_loss': amsoftmax_loss}
#                    )
# model.load_weights(r'D:\For_Python\SomethingINT\holiday\mytry\normal_try\result\webface\1585491382.721277\weights.best.hdf5')


# with open('temp.txt','a') as f:
#     f.writelines(str(cfm.tolist()))


# result = np.argmax(result, axis=1)
# print(result)
# # dataset = Dataset()
# # dataset.load()
# # print('数据加载完毕')
# # score = model.evaluate(x=dataset.test_images,y=dataset.test_labels, verbose=1)
# # print("{}:{:.2f}".format(model.metrics_names[0], score[0]))
# # print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
# # print("%s: %.2f%%" % (model.metrics_names[2], score[2] * 100))
