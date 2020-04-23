import glob
from PIL import Image
from configure import *
import numpy as np
import os
from model_train import NetModel
import shutil
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2

from test_alex import Alextest
def static_prb_list(storepath=problems_path):
    """
    :param storepath: 存储prb_list_path
    :param weights: 1:N，分类模型，权重
    :return: 返还错误识别样例路径至prblem_index.txt
    """
    model = Alextest()
    model.load_model()
    # model.model.load_weights(weights)
    dirs = os.listdir(webface_path)
    index = 0
    totalnum = 0
    for item in dirs:  # loop all directory
        prb_list = []
        temp = []
        for pic in glob.glob(webface_path + '\\' + item + '\\*.jpg'):
            im = Image.open(pic).convert('L')
            im = im.resize((INPUT_HEIGHT, INPUT_WEIGHT), Image.ANTIALIAS)
            im = np.array(im)
            temp.append(im)
        number = len(temp)
        print(number)
        # for it in temp:
        dataset = np.array(temp)
        train_images = dataset.astype('float32')
        train_images /= 255
        result = train_images.reshape(number, 227, 227, 1)
        result = np.argmax(model.model.predict(result), axis=1)
        print(result)
        tt = result.tolist()
        indexs = str(index).zfill(4) + '\\'
        for i, x in enumerate(tt):
            if x != index:
                totalnum += 1
                prb_list.append(webface_path + '\\' + indexs + str(i).zfill(3) + '.jpg')
        #print(str(totalnum))
        index += 1
        with open(storepath, 'a+') as filewriter:
            for problem in prb_list:
                filewriter.write(problem)
                filewriter.write('\n')
static_prb_list()

def cross_problems(prb_path=problems_path, new_prb_path=new_problems_path, inter_path=intersection_path):
    """
    :param prb_path: 前一个问题名单
    :param new_prb_path: 新的问题名单
    :param inter_path: 交叉对比结果
    :return: 返还前后两次模型都不易识别的照片paths
    """
    new_list = []
    with open(prb_path,'r') as filereader:
        lines = filereader.readlines()
        for line in lines:
            new_list.append(line.replace('\n', ''))
    old_list =[]
    with open(new_prb_path, 'r') as filereader:
        lines = filereader.readlines()
        for line in lines:
            old_list.append(line.replace('\n', ''))
    with open(inter_path, 'a') as filewriter:
        for item in list(set(new_list).intersection(set(old_list))):
            filewriter.write(item)
            filewriter.write('\n')


def del_prb(inter_path=intersection_path):
    """
    :param inter_path: 前后两次模型都不易识别的照片paths
    :return: 删除交叉对比--两次都无法识别的照片
    """
    temp_list = []
    with open(inter_path, 'r') as filereader:
        lines = filereader.readlines()
        for line in lines:
            temp_list.append(line.replace('\n', ''))

    import random
    k=int(len(temp_list))
    indexs = random.sample(temp_list,k)
    print(len(indexs)/len(temp_list))
    for item in indexs:
        if os.path.exists(item):  # 如果文件存在
            # 删除文件，可使用以下两种方法。
            os.remove(item)
            # os.unlink(path)
        else:
            print('no such file:%s' % item)  # 则返回文件不存在


def rename_index(path):
    """
    :param path: 需要重命名的照片根目录(二次上级目录)
    :return: 重命名路径下照片id，从零开始
    """
    ori_photo_type = input('键入原始照片后缀(e.g. jpg): ')
    out_photo_type = input('键入原始照片后缀(e.g. bmp): ')
    dirs = os.listdir(path)
    index = 0
    for item in dirs:
        iindex = 0
        for pic_name in glob.glob(path + '\\' + item + '\\*.' + ori_photo_type):
            direction = path + '\\' \
                        + item + '\\' + str(iindex) + '.' + out_photo_type
            os.rename(pic_name, direction)
            iindex += 1
        index += 1
        print(index)
    print('done')


def subprefix_subfolders(path=None, substr=None, index_from_zero=True):
    """
    :param path: 需要去除目录开头的路径地址
    :param substr: 需要去除的目录开头表达
    :param index_from_zero: 目录是否要重新编号，从零开始
    :return:去除目录开头(可含重新编号)
    """
    old_names = os.listdir(path)
    temp_list = []
    temp =old_names[:]
    temp.sort(key=int)
    GG = temp[::-1]
    for item in GG:
        temp_list.append(str(int(item)).zfill(4))
    names_dict = zip(temp_list, GG)
    for i, x in enumerate(names_dict):
        print(x)
        os.rename(os.path.join(path, x[1]), os.path.join(path, str(x[0])))
    # for item in temp[::-1]:
    #     if index_from_zero:
    #         temp_list.append(str(int(item.replace(substr, ''))+1))
    #     else:
    #         # temp_list.append(item.replace(substr, ''))
    #         temp_list.append(str(int(item.replace(substr, ''))))
    # names_dict = zip(temp_list, old_names)
    # for i, x in enumerate(names_dict):
    #     os.rename(os.path.join(path, x[1]), os.path.join(path, str(int(x[0]))))


def mycopyfile(srcfile, dstfile):
    """
    :param srcfile: 复制文件源路径
    :param dstfile: 复制文件目标路径
    :return:复制文件，源-->目标
    """
    # path = r'D:\For_Python\SomethingINT\holiday\mytry\some-dates\Feret(副本)\test'
    if not os.path.isfile(srcfile):
        print("{} not exist!".format(srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.copyfile(srcfile, dstfile)  # 复制文件
        print("copy {} -> {}".format(srcfile, dstfile))


def stastic_face_num(path=r'D:\facerec\CASIA-WebFace'):
    # 统计各文件夹下人脸数目并写入文件
    old_names = os.listdir(path)
    out = open('stastic.csv', 'a+', newline='\n')
    filewriter = csv.writer(out, dialect='excel')
    print('start')
    for item in old_names:
        info = [item]
        face_num = len(os.listdir(path + '\\' + item))
        info.append(str(face_num))
        filewriter.writerow(info)


def copy_move_files(path=r'D:\facerec\CASIA-WebFace'):
    # 从文件中统计指定数目范围内的人脸路径
    with open("stastic.csv", newline='') as f:
        reader = csv.reader(f,delimiter = ',')
        for line in reader:
            if 110 <= int(line[1]) <= 200:
                file_name = line[0].zfill(7)
                print(file_name)
                source_path = path + '\\' + file_name
                direction_path = r'D:\For_Python\SomethingINT\holiday\mytry\some-dates\middle-cisia-webface\\' + file_name
                shutil.copytree(source_path, direction_path)
                print('ok')


# TODO 去掉照片集合中的黑白照
def read_img(photo_path=DATA_SOURCE_PATH):
    images = []
    labels = []
    dirs = os.listdir(photo_path)
    face_num = len(dirs)
    for item in dirs:
        for pic in glob.glob(photo_path + '\\' + item + '\\*.jpg'):
            im = Image.open(pic)
            # .convert('L')  # open data
            im = im.resize((INPUT_HEIGHT, INPUT_WEIGHT), Image.ANTIALIAS)
            im = np.array(im)
            print(pic + str(im.shape))
            r = im[:, :, 0]
            g = im[:, :, 1]
            b = im[:, :, 2]
            images.append([r, g, b])  # save in x_test
            # images.append(im)  # save in x_test
            labels.append(item)
    print(np.array(images).shape, np.array(labels).shape, face_num)
    return np.array(images), np.array(labels), face_num
