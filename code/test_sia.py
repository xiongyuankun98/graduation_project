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
import keras.backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('max_colwidth', 200)


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        r = func(*args, **kwargs)
        end = time.time()
        print(end - start)
        return r
    return wrapper


class SampleTest(object):

    def __init__(self):
        self.model = load_model(siamese_model_path, custom_objects={'contrastive_loss': SiameseNet.contrastive_loss})
        self.model2 = load_model(siamese_model_path2, custom_objects={'contrastive_loss': SiameseNet.contrastive_loss})
        print('SampleTest initial')

    def figure_model(self):
        plot_model(self.model, to_file="model.png", show_shapes=True)

    def pair_test(self, pairs):
        """
        :param self:
        :param pairs: 要对比的一对图片路径
        :return: 对比两照片在当前distance of model now that，返回建议
        """
        if pairs[0].__class__.__name__ == 'Image':
            img_1 = pairs[0].convert('L')
            img_1 = np.array(img_1)
        else:
            print(pairs[0].__class__.__name__)
            img_1 = read_image(pairs[0])[1]

        img_shape, img_2 = read_image(pairs[1])

        img_1 = img_1[::cut_size, ::cut_size] / 255
        img_2 = img_2[::cut_size, ::cut_size] / 255
        img_1 = img_1.reshape(1, 1, int(input_size_height/cut_size), int(input_size_weight/cut_size))
        img_2 = img_2.reshape(1, 1, int(input_size_height/cut_size), int(input_size_weight/cut_size))
        print(img_shape)
        if img_shape[0] < 128:
            pred = self.model.predict([img_1, img_2])[0][0]
            print(img_shape[0])
            if pred < 0.47:
                return str(pred), 'The same one'
            elif pred < 0.6:
                return str(pred), 'Similar, unfortunately'
            else:
                return str(pred), 'Too far away'
        else:
            pred2 = self.model2.predict([img_1, img_2])[0][0]
            if pred2 < 0.56:
                return str(pred2), 'The same one'
            elif pred2 < 0.6:
                return str(pred2), 'Similar, unfortunately'
            else:
                return str(pred2), 'Too far away'

    def rand_test(self, pair_num=3, face_dataset=location, nb_class=NB_CLASSES, per_pic=PIC_NO_EACH):
        """
        :param per_pic: 人脸均张数
        :param nb_class: 人脸类别数
        :param face_dataset: 默认打开的人脸数据存放位置
        :param pair_num: 默认打开的对比照片对数
        :return: 随机对比pair_num组图片，并展示
        """
        print('此为随机测试, 旨在对比不同类别的差异')

        pic_list = []
        prediction_distance = []

        rand_index_1 = random.sample(range(0, nb_class), pair_num * 2)
        rand_index_2 = np.random.randint(0, per_pic, pair_num * 2)

        index_dict = zip(rand_index_1, rand_index_2)

        print('正在随机生成验证序列')
        for i, x in enumerate(index_dict):
            pic_list.append(face_dataset + str(x[0]) + '/' + str(x[1]) + '.bmp')

        print('正在计算图像对内距离')
        for i in range(0, len(pic_list), 2):
            prediction_distance.append(self.pair_test(pairs=pic_list[i:i + 2]))

        max_columns_num = math.ceil(pair_num/3)
        min_margin = min(max_columns_num * 6 - 0.5, 17.5)
        figure = plt.figure(figsize=(7, min_margin))
        figure.canvas.set_window_title('随机 ' + str(pair_num) + ' 组测试, 上下互为一组')
        plt.subplots_adjust(left=0.05, bottom=0.13, right=1, wspace=0.25, hspace=0, top=0.97)

        index = 0

        for i in range(pair_num):
            now_columns_num = math.ceil((i+1)/3)
            print(now_columns_num)
            print('正在绘制···第' + str(i + 1) + '对图像/共计' + str(pair_num) + '对')
            img = Image.open(pic_list[i * 2]) if nb_class == 500 else Image.open(pic_list[i * 2]).convert('LA')
            plt.subplot(max_columns_num * 2, 3, i + 1 + (now_columns_num-1) * 3)
            plt.ylabel('Name:' + str(rand_index_1[i * 2]) + '\nsub:' + str(rand_index_2[i * 2]), fontsize=8)
            if float(prediction_distance[i][0]) < 0.55:
                plt.xlabel(r"Wrong demo, But may close", fontsize=8)
            else:
                plt.xlabel('Result:' + prediction_distance[i][1], fontsize=10)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.axis('on')  # 关掉坐标轴为 off
            plt.title('image_pair_' + str(i + 1))  # 图像题目

            img = Image.open(pic_list[i * 2 + 1])if nb_class == 500 else Image.open(pic_list[i * 2+1]).convert('LA')
            plt.subplot(max_columns_num * 2, 3, i + 1 + (now_columns_num-1) * 3 + 3)
            plt.ylabel('Name:' + str(rand_index_1[i * 2 + 1]) + '\nsub:' + str(rand_index_2[i * 2 + 1]), fontsize=8)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.axis('on')  # 关掉坐标轴为 off
            plt.title('distance: ' + prediction_distance[i][0], fontsize=10)  # 图像题目
            index += 1

        plt.show()

    def unit_test(self, pair_num=3, face_dataset=location, nb_class=NB_CLASSES, per_pic=PIC_NO_EACH):
        """
        :param per_pic: 人脸均张数
        :param nb_class: 人脸类别数
        :param face_dataset: 默认打开的人脸数据存放位置
        :param pair_num: 默认打开的对比照片对数
        :return: 对特定pair_num个人, 随机对比2张图片，并展示
        """
        print('此为单人测试, 选取的图像对为同一个人')

        pic_list = []
        prediction_distance = []

        rand_index_1 = random.sample(range(0, nb_class), pair_num)
        rand_index_1.extend(rand_index_1)
        rand_index_2 = np.random.randint(0, per_pic, pair_num * 2)

        index_dict = zip(rand_index_1, rand_index_2)

        print('正在随机生成验证序列')
        for i, x in enumerate(index_dict):
            pic_list.append(face_dataset + '/' + str(x[0]) + '/' + str(x[1]) + '.bmp')

        print('正在计算图像对内距离')
        for i in range(0, pair_num):
            prediction_distance.append(self.pair_test(pairs=[pic_list[i], pic_list[i + pair_num]]))

        max_columns_num = math.ceil(pair_num / 3)
        min_margin = min(max_columns_num * 6 - 0.5, 17.5)
        figure = plt.figure(figsize=(7, min_margin))
        figure.canvas.set_window_title('单人 ' + str(pair_num) + ' 组测试, 上下互为一组')
        plt.subplots_adjust(left=0.05, bottom=0.13, right=1, wspace=0.25, hspace=0, top=0.97)

        for i in range(pair_num):
            now_columns_num = math.ceil((i + 1) / 3)
            print('正在绘制···第' + str(i + 1) + '对图像/共计' + str(pair_num) + '对')
            img = Image.open(pic_list[i]) if nb_class == 500 else Image.open(pic_list[i]).convert('LA')
            # plt.subplot(2, pair_num, i + 1)
            plt.subplot(max_columns_num * 2, 3, i + 1 + (now_columns_num - 1) * 3)
            plt.ylabel('Name:' + str(rand_index_1[i]) + '\nsub:' + str(rand_index_2[i]), fontsize=8)
            if float(prediction_distance[i][0]) > 0.5:
                plt.xlabel(r"Wrong demo, But not clear", fontsize=8)
            else:
                plt.xlabel(prediction_distance[i][1], fontsize=10)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.axis('on')  # 关掉坐标轴为 off
            plt.title('image_pair_' + str(i + 1))  # 图像题目

            img = Image.open(pic_list[i + pair_num]) if nb_class == 500 else Image.open(pic_list[i + pair_num]).convert('LA')
            # plt.subplot(2, pair_num, i + 1 + pair_num)
            plt.subplot(max_columns_num * 2, 3, i + 1 + (now_columns_num - 1) * 3 + 3)
            plt.ylabel('Name:' + str(rand_index_1[i]) + '\nsub:' + str(rand_index_2[i + pair_num]), fontsize=8)
            if rand_index_2[i] == rand_index_2[i + pair_num]:
                plt.xlabel('Happen to be the same photo', fontsize=8)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.axis('on')  # 关掉坐标轴为 off
            plt.title('distance: ' + prediction_distance[i][0], fontsize=10)  # 图像题目

        plt.show()

    def newpair_test(self):
        """
        :return: 对比指定两张脸的距离
        """
        paths = ['D:/For_Python/SomethingINT/holiday/mytry/some-dates/XYK/0.bmp',
                 'D:/For_Python/SomethingINT/holiday/mytry/some-dates/XYK/1.bmp']
        return self.pair_test(pairs=paths)

    @timeit
    def multi_single_test(self, pair_num=500, margin=0.5, source=location,
                          nb_class=NB_CLASSES, per_pic=PIC_NO_EACH, load_data=None):
        """
        :param self:
        :param load_data: 如果已经有现成数据，则不用每次新载入数据
        :param per_pic: 人脸均张数
        :param nb_class: 人脸类别数
        :param source: 选择测试的图片来源，不一定是来自训练样本
        :param pair_num: 测试2*pair_num组图片
        :param margin: 间距,认定的可信距离
        :return: 对比2*pari_num组图片,返回测试指标及混淆矩阵
        """
        if load_data is None:
            dataset = Dataset()
            image_shape = dataset.get_data(sample_size=pair_num, photos_source=source,
                                           nb_class=nb_class, per_pic=per_pic)
            print(image_shape)
            x, y = dataset.all_images, dataset.all_labels
        else:
            x, y = load_data.all_images, load_data.all_labels
            image_shape, image = read_image(source + str(0) + '/' + str(0) + '.bmp')
            print(image_shape)
        index = [i for i in range(len(y))]
        random.shuffle(index)
        data = x[index]
        label = y[index]

        if image_shape[0] < 128:
            pred = self.model.predict([data[:, 0], data[:, 1]])
        else:
            print(image_shape[0])
            pred = self.model2.predict([data[:, 0], data[:, 1]])

        result = list(SiameseNet.compute_accuracy(pred, label, margin=margin))
        cmx = result.pop()

        return result, cmx

    def multi_test(self, pair_num=2000, store_recording=True):
        """
        margins:用于测试最佳'margin'
        :param pair_num: 测试2*pair_num组图片
        :param store_recording: 是否存储测试指标
        :return:
        """
        margins = np.linspace(0.01, 1.0, 100)
        dataset = Dataset()
        dataset.get_data(sample_size=pair_num, photos_source=location, nb_class=200, per_pic=7)
        if store_recording:
            with open('judge.csv', 'a+', newline='') as filewriter:
                csvwriter = csv.writer(filewriter, dialect="excel")
                csvwriter.writerow(['margin', 'accurace', 'precision', 'recall(tpr)', 'fdr', 'f1score'])
                for item in margins:
                    print(item)
                    init = list(self.multi_single_test(pair_num, margin=item, load_data=dataset,nb_class=500, per_pic=5))[0]
                    init.insert(0, str(round(item, 2)))
                    csvwriter.writerow(init)
        else:
            print(['margin', 'accurace', 'precision', 'recall', 'f1score'])
            for item in margins:
                init = list(self.multi_single_test(pair_num, margin=item))[0]
                init.insert(0, str(round(item, 2)))
                print(init)

    @timeit
    def confusion_metricx(self, photo_source=location, pair_num=2000, margin=0.5,nb_class=NB_CLASSES, per_pic=PIC_NO_EACH):
        """
        :return: 根据multi_single_test测试结果，绘制混淆矩阵
        """
        index_rate, cmx = self.multi_single_test(source=photo_source, pair_num=pair_num, margin=margin,nb_class=nb_class,per_pic=per_pic)
        print(index_rate)
        classes = [0, 1]
        plt.figure(figsize=(4, 4))
        plt.imshow(cmx, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0)
        plt.yticks(tick_marks, classes)
        thresh = cmx.max() / 2.0
        for i, j in itertools.product(range(cmx.shape[0]), range(cmx.shape[1])):
            plt.text(j, i, cmx[i, j],
                     horizontalalignment="center",
                     color="white" if cmx[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label', fontsize=8)
        plt.show()
        return index_rate

    @staticmethod
    @timeit
    def statics_roc_auc(path='judge.csv',show_roc=True):
        """
        :param show_roc: 展现roc曲线绘制结果
        :return: 根据multi_test测试结果指标，绘制roc曲线，计算auc值
        """
        with open(path, 'r', newline='') as filereader:
            csvreader = csv.reader(filereader, delimiter=',')
            next(csvreader)
            tpr_fpr = []
            for row in csvreader:
                tpr_fpr.append(tuple((float(row[3]), float(row[4]))))

        sorted_tpr_fdr = sorted(tpr_fpr, key=lambda t: t[1])

        init_limit_tpr = round(sorted_tpr_fdr[-1][0], 4)
        init_limit_fpr = round(sorted_tpr_fdr[-1][1], 4)
        num_extend = int((1-init_limit_fpr) / 0.01 + 1)
        extend_array = np.linspace(init_limit_fpr, 1, num_extend)

        temple = sorted_tpr_fdr[:]

        for item in extend_array:
            sorted_tpr_fdr.append(tuple((init_limit_tpr, round(item, 4))))

        df = pd.DataFrame(sorted_tpr_fdr, columns=['tpr', 'fpr'])
        df2 = pd.DataFrame(temple, columns=['tpr', 'fpr'])

        auc_value = auc(df2['fpr'], df2['tpr']) + init_limit_tpr * (1-init_limit_fpr)

        if show_roc:
            plt.figure(figsize=(9, 9))

            plt.scatter(x=df['fpr'], y=df['tpr'], label='(FPR,TPR)', color='c', marker='.')
            plt.plot(df['fpr'], df['tpr'], 'm', label='AUC = %0.4f' % auc_value)
            plt.plot([(0, 0), (1, 1)], 'r--')

            up_range = np.ones((1, len(df)))[0]
            plt.fill_between(x=df['fpr'], y1=df['tpr'], y2=up_range, facecolor='green')

            plt.legend(loc='lower right')
            plt.title('Receiver Operating Characteristic')
            plt.grid()

            xticks_range = np.linspace(0, 1, 21)
            yticks_range = np.linspace(0, 1, 21)
            plt.xticks(xticks_range)
            plt.yticks(yticks_range)

            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 01.01])

            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')

            # 设置坐标轴交点为(0,0)
            # ax = plt.gca()
            # ax.spines["bottom"].set_position(("data", 0))
            # ax.spines["left"].set_position(("data", 0))

            plt.show()

        return auc_value
