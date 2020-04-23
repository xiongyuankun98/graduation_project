import time
# OpenCV人脸位置检测xml文件路径
face_detection_path = r'D:\\For_Python\\opencv\\opencv-3.4.4\\data\\haarcascades\\haarcascade_frontalface_default.xml'
# 原始数据集500*5存储路径
pic_ori_path = r'D:\For_Python\SomethingINT\holiday\mytry\some-dates\64_CASIA-FaceV5\CASIA-FaceV5'
# 存放剪裁头像后存储路径
pic_store_path = r'D:\For_Python\SomethingINT\holiday\mytry\some-dates\64_CASIA-FaceV5\CASIA-FaceV5 1st'

# 小规模测试数据集jaffedbase-10人213张图片存放路径
# DATA_SOURCE_PATH, DATA_NAME = r'D:\For_Python\SomethingINT\holiday\mytry\cnn_try_jp\japan', 'JAFFE'
# 中等规模测试数据集CISIA-FaceV5-500人2500张图片+1人10张存放路径
# DATA_SOURCE_PATH = r'D:\For_Python\SomethingINT\holiday\mytry\some-dates\64_CASIA-FaceV5\CASIA-FaceV5'
# DATA_SOURCE_PATH = r'D:\For_Python\SomethingINT\holiday\mytry\some-dates\cfp-dataset\Data\Images'
# DATA_SOURCE_PATH = r'D:\For_Python\SomethingINT\holiday\mytry\some-dates\middle-cisia-webface'
# DATA_SOURCE_PATH = r'D:\For_Python\SomethingINT\holiday\mytry\some-dates\small-50-cisia-webface'
DATA_SOURCE_PATH = r'D:\For_Python\SomethingINT\holiday\mytry\some-dates\small-webface'


# OpenCV检测人脸位置失败/异常，需要人工筛选剪裁的文件存放路径
manual_store_path = r'D:\For_Python\SomethingINT\holiday\mytry\some-dates\64_CASIA-FaceV5\handwork'


# # 存放网络模型存放路径，含模型结构与训练好的参数
# MODEL_PATH = r'D:\For_Python\SomethingINT\holiday\mytry\\normal_try\modelstore\\'
# # 学习曲线保存图片地址
# learning_curve_path = 'D:\For_Python\SomethingINT\holiday\mytry\\normal_try\curvestore\\'

all_store_path = r'D:\For_Python\SomethingINT\holiday\mytry\normal_try\result\{}'.format(time.time())

# log_path = all_store_path
MODEL_PATH = all_store_path
learning_curve_path = all_store_path

# 照片HEIGHT--Y--ROWS
INPUT_HEIGHT = 227
# 照片WEIGHT--X--COLS
INPUT_WEIGHT = 227
# 照片输入通道--(1--黑白,3--彩照)
IMAGE_CHANNELS = 1
# 人脸类别数目
NB_CLASS = 50
# 照片输入尺寸(HEIGHT--WEIGHT--CHANNELS)
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WEIGHT, IMAGE_CHANNELS)

# 每次输入网络训练数量
BATCH_SIZE = 16
# 训练轮次
NB_EPOCH = 300

# SGD参数
SGD_LR = 0.00003  # LearningRate 学习率
SGD_DECAY = 1e-6  # 动量学习率衰减量
SGD_MOMENTUM = 0.9  # 动量，Parameter updates momentum.

# keras.optimizers.Adagrad
# 考虑这个优化器

PrefixRegex = r'^\d{1,3}'
PicName = r'.+\\(.+)$'


# 换数据集说明
# 考虑data_input.py:
# read_image:
# 彩色?黑白?
# 图片格式?

# 考虑configure.py:
# INPUT_HEIGHT,
# INPUT_WEIGHT,
# IMAGE_CHANNELS,
# NB_CLASSES,
# BATCH_SIZE,
# NB_EPOCH
# 文件前缀名|正则表达式

# cut_size = 2
# input_size_height = 80
# input_size_weight = 80
# total_sample_size = 2000
# NB_CLASSES = 200
# PIC_NO_EACH = 7
# # siamese_epochs = 30
#
cut_size = 2
input_size_height = 256
input_size_weight = 256
total_sample_size = 9800
NB_CLASSES = 200
PIC_NO_EACH = 7
siamese_epochs = 50
INPUT_SHAPE2 = (IMAGE_CHANNELS, int(input_size_height/cut_size), int(input_size_weight/cut_size))

problems_path = 'prblem_index.txt'
intersection_path = 'intersection.txt'
new_problems_path = r'D:\For_Python\SomethingINT\holiday\mytry\normal_try\result\webface\1585491382.721277\new.txt'

# alexnet_weight_model = 'D:/For_Python/SomethingINT/holiday/mytry/normal_try/result/1585632951.3588202/weights.best.hdf5'
alexnet_weight_model = 'D:/For_Python/SomethingINT/holiday/mytry/normal_try/result/webface/1585491382.721277/weights.best.hdf5'
alexnet_model_path = 'D:/For_Python/SomethingINT/holiday/mytry/normal_try/result/webface/1585491382.721277/model.h5'
# alexnet_model_path = 'D:/For_Python/SomethingINT/holiday\mytry/normal_try/result/jpaee/ten/model_1.h5'

siamese_model_path2 = 'D:/For_Python/SomethingINT/holiday/mytry/normal_try/result/siamese/cisiaface_v5/model.h5'
location2 = 'D:/For_Python/SomethingINT/holiday/mytry/some-dates/64_CASIA-FaceV5/CASIA-FaceV5 1st/'

siamese_model_path = 'D:/For_Python/SomethingINT/holiday/mytry/normal_try/result/siamese/feret_256_new/model.h5'
location = 'D:/For_Python/SomethingINT/holiday/mytry/some-dates/Feret(副本)/FERET_80_80-人脸数据库/'
webface_path = r'D:\For_Python\SomethingINT\holiday\mytry\some-dates\small-webface'

base_css = "<p style='text-align:{}; font-size:{}px; color: black;'>"