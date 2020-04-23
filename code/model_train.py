"""
网络模型构建，训练及保存模型
"""

from keras.models import Model, load_model
from keras.layers import Convolution2D, MaxPooling2D, Input
from keras.layers import Dense, Dropout, Activation, Flatten, concatenate, BatchNormalization,regularizers
from keras.optimizers import SGD,Adam, Adadelta
from data_input import Dataset
from configure import *
from keras.preprocessing.image import ImageDataGenerator
from Auxiliary import LossHistory
from sklearn.model_selection import KFold, StratifiedKFold
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import os
from keras import backend as K
from amsoftmax import amsoftmax_loss,AMSoftmax
from keras.metrics import top_k_categorical_accuracy
from keras.losses import categorical_crossentropy


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.80
set_session(tf.Session(config=config))


class NetModel(object):
    def __init__(self):
        # 建立模型
        self.model = None
        self.selected_model = None
        self.csv_logger = CSVLogger(all_store_path+r'\logger.csv')
        self.history = LossHistory()  # 实例化一个记录学习曲线的类
        self.tensorboard = TensorBoard(log_dir=all_store_path)
        self.checkpoint = ModelCheckpoint(all_store_path+r'\weights.best.hdf5',
                                          monitor='val_acc', verbose=1,
                                          save_best_only=True, mode='max')
        self.earlystopping = EarlyStopping(monitor='val_acc', patience=15, verbose=2, mode='auto')

    def build_alexnet_deformation_model(self):
        self.selected_model = 'alexnet'
        # 构建一个空的网络模型，函数式API，自由组合，需要指定输入输出
        inputs = Input(shape=INPUT_SHAPE, name='input')  # 227*227*3

        # layer1：两个卷积操作、两个pooling操作、两个normaliziton操作，
        # conv1_1 = Convolution2D(48, (11, 11), strides=(4, 4), activation='relu',
        #                         name='conv1_1', kernel_initializer='he_normal')(inputs)  # 55*55*48
        # conv1_2 = Convolution2D(48, (11, 11), strides=(4, 4), activation='relu',
        #                         name='conv1_2', kernel_initializer='he_normal')(inputs)  # 55*55*48
        conv1_1 = Convolution2D(96, (7, 7), strides=(2, 2), activation='relu',
                                name='conv1_1', kernel_initializer='he_normal')(inputs)  # 55*55*48
        conv1_2 = Convolution2D(96, (5, 5), strides=(2, 2), activation='relu',padding='same',
                                name='conv1_2', kernel_initializer='he_normal')(conv1_1)  # 55*55*48
        normal1_1 = BatchNormalization(name='normal1_1',axis=1)(conv1_2)  # 27*27*48
        # normal1_2 = BatchNormalization(name='normal1_2',axis=1)(conv1_2)  # 27*27*48
        pool1_1 = MaxPooling2D((3, 3), strides=(2, 2), name='pool1_1')(normal1_1)  # 27*27*48
        # pool1_2 = MaxPooling2D((3, 3), strides=(2, 2), name='pool1_2')(normal1_2)  # 27*27*48
        # normal标准化层：解决内部协变量转变问题，更加健壮
        # normal1_1 = BatchNormalization(name='normal1_1')(pool1_1)  # 27*27*48
        # normal1_2 = BatchNormalization(name='normal1_2')(pool1_2)  # 27*27*48

        # layer2:两个卷积，将前边的池化得到的数据，进行卷积，再继续池化
        conv2_1 = Convolution2D(128, (5, 5), strides=(1, 1), activation='relu',
                                padding='same', kernel_initializer='he_normal')(pool1_1)  # 27*27*128
        conv2_2 = Convolution2D(128, (5, 5), strides=(1, 1), activation='relu',
                                padding='same', kernel_initializer='he_normal')(pool1_1)  # 27*27*128
        normal2_1 = BatchNormalization(name='normal2_1',axis=1)(conv2_1)  # 13*13*128
        normal2_2 = BatchNormalization(name='normal2_2',axis=1)(conv2_2)  # 13*13*128
        pool2_1 = MaxPooling2D((3, 3), strides=(2, 2), name='pool2_1')(normal2_1)  # 13*13*128
        pool2_2 = MaxPooling2D((3, 3), strides=(2, 2), name='pool2_2')(normal2_2)  # 13*13*128
        # normal标准化层：解决内部协变量转变问题，更加健壮
        # normal2_1 = BatchNormalization(name='normal2_1')(pool2_1)  # 13*13*128
        # normal2_2 = BatchNormalization(name='normal2_2')(pool2_2)  # 13*13*128

        # layer3:两个卷积操作
        conv3_1 = Convolution2D(192, (3, 3), strides=(1, 1), activation='relu',
                                name='conv3_1', padding='same')(pool2_1)  # 13*13*192
        conv3_2 = Convolution2D(192, (3, 3), strides=(1, 1), activation='relu',
                                name='conv3_2', padding='same')(pool2_2)  # 13*13*192

        # latyer4:两个卷积操作
        conv4_1 = Convolution2D(192, (3, 3), strides=(1, 1), activation='relu',
                                name='conv4_1', padding='same')(conv3_1)    # 13*13*192
        conv4_2 = Convolution2D(192, (3, 3), strides=(1, 1), activation='relu',
                                name='conv4_2', padding='same')(conv3_2)    # 13*13*192

        # layer5:两个卷积操作和两个pooling操作
        conv5_1 = Convolution2D(128, (3, 3), strides=(1, 1), activation='relu',
                                name='conv5_1', padding='same')(conv4_1)  # 13*13*192
        conv5_2 = Convolution2D(128, (3, 3), strides=(1, 1), activation='relu',
                                name='conv5_2', padding='same')(conv4_2)  # 13*13*192
        pool5_1 = MaxPooling2D((3, 3), strides=(2, 2), name='pool5_1')(conv5_1)  # 6*6*128
        pool5_2 = MaxPooling2D((3, 3), strides=(2, 2), name='pool5_2')(conv5_2)  # 6*6*128

        # merge合并层：第五层进入全连接之前，要将分开的合并
        merge = concatenate([pool5_1, pool5_2], axis=-1)  # 6*6*256
        # 通过flatten将多维输入一维化
        dense1 = Flatten(name='flatten')(merge)  # 9216

        # layer6：进行4096维的全连接，中间加dropout避免过拟合
        dense2_1 = Dense(4096, activation='relu', name='dense2_1')(dense1)  # 4096
        dense2_2 = Dropout(0.5)(dense2_1)  # 4096

        # # layer7：再次进行4096维的全连接，中间加dropout避免过拟合
        # dense3_1 = Dense(4096, activation='relu', name='dense3_1')(dense2_2)  # 4096
        # dense3_2 = Dropout(0.5)(dense3_1)  # 4096

        # 输出层：输出类别，分类函数使用softmax
        # dense3_3 = Dense(NB_CLASS, name='dense3_3')(dense2_2)  # NB_CLASSES
        # prediction = Activation('softmax', name='softmax')(dense3_3)  # NB_CLASSES

        prediction = AMSoftmax(NB_CLASSES)(dense2_2)
        # logit = arcface_loss(embedding=dense2_2, labels=labels_s[i], w_init=keras.initializers.glorot_uniform, out_num=NB_CLASS)

        # output = NormDense(NB_CLASS, name='arcface_dense')(dense2_2)

        # 最后定义模型输出
        alexnet = Model(inputs=[inputs], outputs=[prediction])
        self.model = alexnet

    def train(self, dataset, data_augmentation=True):

        # inputs = self.model.inputs[0]
        # embedding = self.model.outputs[0]
        # output = NormDense(50, name='norm_dense')(embedding)
        # self.model = Model(inputs, output)
        # self.model.summary()

        sgd = SGD(lr=SGD_LR, decay=SGD_DECAY,
                  momentum=SGD_MOMENTUM, nesterov=True)  # 采用SGD+momentum的优化器进行训练，首先生成一个优化器对象
        # adam = Adam(lr=0.0012, beta_1=0.9, beta_2=0.9, epsilon=1e-08, amsgrad=True)
        adad = Adadelta(epsilon=1e-7)  # lr=0.01? epsilon=1e-8 小数据集， epsilon=1e-7 大数据集
        # 浮点数，一个模糊的数值常量，在某些操作中用于避免被零除。
        self.model.compile(loss=amsoftmax_loss,
                           optimizer=adad,
                           metrics=['accuracy', top_k_categorical_accuracy])  # 完成实际的模型配置工作
        # ,ins.precisionnnn,ins.recallll
        # 不使用数据提升
        # 训练数据，有意识的提升训练数据规模，增加模型训练量
        if not data_augmentation:
            self.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size=BATCH_SIZE,
                           epochs=NB_EPOCH, verbose=1,
                           validation_data=(dataset.valid_images, dataset.valid_labels),
                           shuffle=True,
                           callbacks=[self.history, self.csv_logger,self.tensorboard,self.checkpoint,self.earlystopping])
            # ''' Train for a few epoch only to fit the bottleneck layer '''
            # self.model.trainable = False
            # self.model.compile(optimizer='adamax', loss=arcface_loss, metrics=["accuracy"])
            # self.model.fit(dataset.train_images,
            #                       dataset.train_labels,
            #                       epochs=2, verbose=1,
            #                       steps_per_epoch=1,
            #                       validation_data=(dataset.valid_images, dataset.valid_labels),
            #                       shuffle=True,
            #                       callbacks=[self.history, self.csv_logger, self.tensorboard],
            #                       validation_steps=1)
            #
            # ''' Train the whole model '''
            # self.model.trainable = True
            # self.model.compile(optimizer='adamax', loss=arcface_loss,
            #                    metrics=["accuracy"])  # MUST run compile after changing trainable value
            # self.model.fit(dataset.train_images,
            #                       dataset.train_labels,
            #                       epochs=20, verbose=1,
            #                       steps_per_epoch=1,
            #                       validation_data=(dataset.valid_images, dataset.valid_labels),
            #                       validation_steps=1,
            #                       shuffle=True,
            #                       callbacks=[self.history, self.csv_logger, self.tensorboard])
        # 使用实时数据提升
        else:
            # 提升就是从我们提供的训练数据中利用旋转、翻转、加噪声等方法创造新的
            # 定义数据生成器用于数据提升，其返回一个生成器对象datagen，datagen每被调用一
            # 次其生成一组数据（顺序生成），节省内存，其实就是python的数据生成器
            datagen = ImageDataGenerator(
                featurewise_center=False,  # 是否使输入数据去中心化（均值为0），
                samplewise_center=False,  # 是否使输入数据的每个样本均值为0
                featurewise_std_normalization=False,  # 是否数据标准化（输入数据除以数据集的标准差）
                samplewise_std_normalization=False,  # 是否将每个样本数据除以自身的标准差
                zca_whitening=True,  # 是否对输入数据施以ZCA白化
                rotation_range=20,  # 数据提升时图片随机转动的角度(范围为0～180)
                width_shift_range=0.2,  # 数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
                height_shift_range=0.2,  # 同上，只不过这里是垂直
                horizontal_flip=True,  # 是否进行随机水平翻转
                vertical_flip=False)  # 是否进行随机垂直翻转

            # 计算整个训练样本集的数量以用于特征值归一化、ZCA白化等处理

            datagen.fit(dataset.train_images)

            # 利用生成器开始训练模型
            self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,
                                                  batch_size=BATCH_SIZE),
                                     # samples_per_epoch=dataset.train_images.shape[0],
                                     steps_per_epoch=dataset.train_images.shape[0]//BATCH_SIZE,
                                     epochs=NB_EPOCH,shuffle=True,
                                     validation_data=(dataset.valid_images, dataset.valid_labels),
                                     validation_steps=dataset.valid_images.shape[0]//BATCH_SIZE,
                                     callbacks=[self.history, self.csv_logger,self.tensorboard,self.checkpoint,self.earlystopping])
        self.history.loss_plot('epoch')

    def cross_validation(self, dataset):
        import numpy as np
        from keras.utils import np_utils
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        data = dataset.all_images
        label = dataset.all_labels
        label = [int(float(i)) for i in label]
        index = 1
        for train, test in kfold.split(data, label):
            # from numba import cuda
            # cuda.select_device(0)
            # cuda.close()
            self.build_alexnet_deformation_model()
            mymodel = self.model
            # sgd = SGD(lr=0.001, decay=1e-8, momentum=0.9, nesterov=True)
            adad = Adadelta()
            self.model.compile(loss=amsoftmax_loss,
                               optimizer=adad,
                               metrics=['accuracy', top_k_categorical_accuracy])
            mymodel.fit(data[train], np.array(label)[train], validation_data=(data[test], np.array(label)[test]),
                        epochs=300, batch_size=4,
                        shuffle=True, callbacks=[self.history, CSVLogger(all_store_path+r'\logger_'+str(index)+'.csv'),
                                                 ModelCheckpoint(all_store_path+r'\weights.best_'+str(index)+'.hdf5',
                                                                 monitor='val_acc', verbose=1,
                                                                 save_best_only=True, mode='max'),
                                                 self.tensorboard,self.earlystopping], verbose=1)
            # np.array(label)
            # train_data, train_label = prepare_data()
            # model.fit(train_data, train_label, batch_size=64, epochs=20, shuffle=True, validation_split=0.2)
            mymodel.save(MODEL_PATH + r'\model_' + str(index) + '.h5')
            self.history.loss_plot('epoch', note=index)
            index = index + 1
            K.clear_session()

    def save_model(self, file_path=MODEL_PATH):
        self.model.save(file_path+r'\model.h5')

    # def load_model(self, file_path=MODEL_PATH):
    #     self.model = load_model(file_path, custom_objects={'amsoftmax_loss': amsoftmax_loss})

    def evaluates(self, dataset):
        score = self.model.evaluate(dataset.test_images, dataset.test_labels, verbose=1)
        print("{}:{:.2f}".format(self.model.metrics_names[0], score[0]))
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))
        print("%s: %.2f%%" % (self.model.metrics_names[2], score[2] * 100))
        # readmodel = load_model(all_store_path+r'\model.h5',
        #                        custom_objects={'amsoftmax_loss': amsoftmax_loss,
        #                                        'top':top_k_categorical_accuracy,
        #                                        'AttLayer': AMSoftmax
        #                                        })
        #
        # loss,acc = self.model.evaluate(datasets)
        # print('\naccuracy', accuracy)
        # print(np.argmax(readmodel.predict(dataset.test_images), axis=1))

    def scheduler(self, epoch):
        if epoch % 100 == 0 and epoch != 0:
            lr = K.get_value(self.model.optimizer.lr)
            K.set_value(self.model.optimizer.lr, lr * 0.1)
            print("lr changed to {}".format(lr * 0.1))
        return K.get_value(self.model.optimizer.lr)

    reduce_lr = LearningRateScheduler(scheduler)


# if __name__ == '__main__':
#
#     datasets = Dataset()
#     datasets.load()
#
#     model = NetModel()
#     model.model = load_model(alexnet_model_path, custom_objects={'AMSoftmax': AMSoftmax,'amsoftmax_loss': amsoftmax_loss})
#     # model.build_alexnet_deformation_model()
#     # model.train(datasets, data_augmentation=False)
#     # model.save_model()
#     model.evaluates(datasets)
#     # model.cross_validation(datasets)


# TODO 尝试保存学习曲线，尝试换模型，尝试模型融合，
# TODO 尝试不同优化器，尝试添加自己图片、已有图片变形及预测，尝试CISIA，
# TODO 尝试Django尝试vue尝试pyechart，尝试获取特征图(倒数第二层输出)，尝试'判定距离'
