"""
图像获取、图像预处理等并存储
CASIA-FaceV5：500人,2500张图片(640*480)
"""
import cv2 as cv
import os
import numpy as np
from configure import *
import re
import glob
# from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array, array_to_img


def face_position_detection_with_opencv(imgfile):
    # 人脸位置检测
    face_cascade = cv.CascadeClassifier(face_detection_path)
    # print(img.shape)
    gray = cv.cvtColor(imgfile, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3)  # 一般1.2-1.3
    print(len(faces))
    for (x, y, w, h) in faces:
        img = cv.rectangle(imgfile, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # img = cv.rectangle(imgfile, (x, y), (0, 0), (0, 0, 255), 2)
    cv.imshow('image', img)
    cv.imwrite('000_0_1.bmp',img)
    cv.waitKey(0)
    return faces[0]


def face_cropping_and_scaling(filepath):
    # 人脸剪裁并放缩
    ori_img = cv.imread(filepath)
    (x, y, w, h) = face_position_detection_with_opencv(ori_img)
    first_step_img = ori_img[y:y + w, x:x + w]  # 裁剪坐标为[y0:y1, x0:x1]
    cv.imshow('image', first_step_img)
    cv.imwrite('000_0_2.bmp', first_step_img)
    secend_step_img = cv.resize(first_step_img, (INPUT_HEIGHT, INPUT_WEIGHT), interpolation=cv.INTER_CUBIC)
    cv.imwrite('000_0_3.bmp', secend_step_img)
    cv.imshow('reSize1', secend_step_img)
    cv.waitKey(0)
    return secend_step_img


def cut_and_store_iamge():
    for root, dirs, files in os.walk(pic_ori_path):
        for directory in dirs:
            for item in os.listdir(pic_ori_path + '\\' + directory):
                ori_path = pic_ori_path + '\\' + directory + '\\' + item
                out_path = pic_store_path + '\\' + directory + '\\' + item
                try:
                    image_result = face_cropping_and_scaling(ori_path)
                    cv.imwrite(out_path, image_result)
                except IndexError:
                    pass
            print(directory)


def verification_quantity(pic_init_path=pic_ori_path, pic_deal_path=pic_store_path):
    # 出现问题，人工处理
    errorlist = []
    for root, dirs, files in os.walk(pic_deal_path):
        for directory in dirs:
            if len(os.listdir(pic_deal_path + '\\' + directory)) != 5:
                list1 = os.listdir(pic_deal_path + '\\' + directory)
                list2 = os.listdir(pic_init_path + '\\' + directory)
                if len(list1) != 5:
                    print(len(list1), list1)
                errorlist.extend(list(set(list2).difference(set(list1))))
    for item in errorlist:
        image = cv.imread(pic_init_path + '\\' + re.findall(PrefixRegex, item)[0] + '\\' + item)
        cv.imwrite(manual_store_path + '\\' + item, image)


def store_image_after_handwork(pic_deal_path=pic_store_path):
    for pic in glob.glob(manual_store_path + '\\' + r'\*.bmp'):
        file_name = re.findall(PicName, pic)[0]
        directory_name = re.findall(PrefixRegex, file_name)[0]
        image = cv.imread(pic)
        image = cv.resize(image, (INPUT_HEIGHT, INPUT_WEIGHT), interpolation=cv.INTER_CUBIC)
        cv.imwrite(pic_deal_path + '\\' + directory_name + '\\' + file_name, image)


def image_process(filepath, gray=False, he=True, z_score=True, blur=False):
    # 数据预处理
    file = cv.imread(filepath)

    if blur is True:
        # 开启中值滤波、高斯滤波等
        dst = cv.medianBlur(file, 5)
        file = cv.GaussianBlur(dst, (5, 5), 0)

    if gray is False and he is True:
        # 彩色图像均衡化,需要分解通道 对每一个通道均衡化
        (b, g, r) = cv.split(file)
        bh = cv.equalizeHist(b)
        gh = cv.equalizeHist(g)
        rh = cv.equalizeHist(r)
        # 合并每一个通道
        histogram_equalization = cv.merge((bh, gh, rh))
        # print(histogram_equalization.shape)
        # cv.imshow('histogram_equalization', histogram_equalization)
        # cv.waitKey(0)
        file = histogram_equalization
    elif gray is True and he is True:
        # 灰度图像均衡化
        gray = cv.cvtColor(file, cv.COLOR_BGR2GRAY)
        dst = cv.equalizeHist(gray)
        file = dst
    elif gray is True and he is False:
        # 仅保存灰度图像，不做均衡化
        gray = cv.cvtColor(file, cv.COLOR_BGR2GRAY)
        file = gray

    if z_score is True:
        # 归一化，均值为 0，方差为 1
        file2 = file.copy()
        z_score_normalized = (file - np.average(file)) / np.std(file)
        # z_score_normalized = file / np.std(file)
        file = z_score_normalized
        file2 = file2 / 255.0
    # cv.imshow('test',file)
    # cv.waitKey(0)
    # print(file)
    # cv.imwrite('000_0_6.bmp', file)
    return file, file2



def process_and_store_iamge():
    for root, dirs, files in os.walk(pic_ori_path):
        for directory in dirs:
            for item in os.listdir(pic_ori_path + '\\' + directory):
                ori_path = pic_ori_path + '\\' + directory + '\\' + item
                out_path = pic_store_path + '\\' + directory + '\\' + item
                try:
                    image_result = image_process(ori_path)
                    # 默认：彩图，无滤波，均衡化，归一化
                    image_result = image_result * 255
                    # 归一化之后，数值应还原至0-255方可显现图像
                    cv.imwrite(out_path, image_result)
                except IndexError:
                    pass
            print(directory)


