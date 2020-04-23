import numpy as np
from random import randint
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Convolution2D, MaxPooling2D, Input, Lambda
from keras.layers import Dense, Dropout, Activation, Flatten, concatenate, BatchNormalization, regularizers
from keras.optimizers import SGD, Adam, Adadelta, RMSprop
from data_input import Dataset
from configure import *
from sklearn.model_selection import KFold, StratifiedKFold
from keras.callbacks import CSVLogger, ModelCheckpoint,TensorBoard, EarlyStopping
import tensorflow as tf
import os
from keras import backend as K
from PIL import Image
import numpy as np
from keras.utils import np_utils
from keras.backend.tensorflow_backend import set_session
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.80
set_session(tf.Session(config=config))


class SiameseNet(object):
    def __init__(self):
        self.model = None
        self.tensorboard = None
        self.checkpoint = None
        self.earlystopping = None
        self.csv_logger = None

    def build_callbacks(self, storepath=all_store_path):
        self.tensorboard = TensorBoard(log_dir=storepath)
        self.checkpoint = ModelCheckpoint(storepath+r'\weights.best.hdf5',
                                          monitor='val_loss', verbose=1,
                                          save_best_only=True, mode='min')
        self.earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=2, mode='auto')
        self.csv_logger = CSVLogger(storepath + r'\logger.csv')

    @staticmethod
    def build_base_network(input_shape):
        seq = Sequential()
        nb_filter = [48, 64]
        # convolutional layer 1
        seq.add(Convolution2D(nb_filter[0], (5, 5), input_shape=input_shape,
                              padding='valid', data_format="channels_first"))
        seq.add(Activation('relu'))
        seq.add(MaxPooling2D(pool_size=(2, 2)))
        seq.add(Dropout(0.5))
        # convolutional layer 2
        seq.add(Convolution2D(nb_filter[1], (5, 5), padding='valid', data_format="channels_first"))
        seq.add(Activation('relu'))
        seq.add(MaxPooling2D((2, 2), data_format="channels_first"))
        seq.add(Dropout(0.5))
        # convolutional layer 2
        seq.add(Convolution2D(nb_filter[1], (3, 3), padding='valid', data_format="channels_first"))
        seq.add(Activation('relu'))
        # flatten
        seq.add(Flatten())
        seq.add(Dense(256, activation='relu'))
        seq.add(Dropout(0.1))
        seq.add(Dense(100, activation='relu'))
        return seq

    def build_real_model(self):
        base_model = self.build_base_network(INPUT_SHAPE2)

        img_a = Input(shape=INPUT_SHAPE2)
        img_b = Input(shape=INPUT_SHAPE2)

        feat_vecs_a = base_model(img_a)
        feat_vecs_b = base_model(img_b)

        distance = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])

        self.model = Model(inputs=[img_a, img_b], outputs=[distance])

    def train_and_save(self, datasets):
        rms = RMSprop()
        self.model.compile(loss=self.contrastive_loss, optimizer=rms)
        img_1 = datasets.train_images[:, 0]  # shape:num_samples, channels, height, weitht
        img_2 = datasets.train_images[:, 1]
        self.model.fit([img_1, img_2], datasets.train_labels, validation_split=.25, batch_size=64,
                       verbose=1, epochs=siamese_epochs,
                       callbacks=[self.tensorboard, self.csv_logger, self.earlystopping, self.checkpoint])
        pred = self.model.predict([datasets.test_images[:, 0], datasets.test_images[:, 1]])
        print(self.compute_accuracy(pred, datasets.test_labels, 0.47))
        self.model.save(all_store_path + '\\model.h5')

    @staticmethod
    def euclidean_distance(vects):
        x, y = vects
        return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

    @staticmethod
    def eucl_dist_output_shape(shapes):
        shape1, shape2 = shapes
        return shape1[0], 1

    @staticmethod
    def contrastive_loss(y_true, y_pred):
        margin = 1
        return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

    @staticmethod
    def compute_accuracy(predictions, labels, margin):

        labels_true = np.where(labels == 1)
        labels_false = np.where(labels != 1)

        predictions_true = predictions[labels_true]
        predictions_false = predictions[labels_false]

        tp_count = labels_true[0][predictions_true.ravel() < margin].size
        fn_count = labels_true[0][predictions_true.ravel() > margin].size
        fp_count = labels_false[0][predictions_false.ravel() < margin].size
        tn_count = labels_false[0][predictions_false.ravel() > margin].size

        cm = np.arange(4).reshape(2, 2)
        cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1] = tn_count, fp_count, fn_count, tp_count

        print(tp_count, fn_count, fp_count, tn_count)
        accurace = (tp_count + tn_count) / (tp_count + fn_count + fp_count + tn_count)
        precision = tp_count/(tp_count + fp_count)
        recall = tp_count/(tp_count + fn_count)  # TPR = TP/ (TP + FN)
        fdr = fp_count / (tn_count + fp_count)  # fp/ (TP + fp)
        f1score = 2 * recall * precision / (recall + precision)
        return str(round(accurace, 4)), str(round(precision, 4)), \
               str(round(recall, 4)), str(round(fdr, 4)),\
               str(round(f1score, 4)), cm


# if __name__ == '__main__':
#     dataset = Dataset()
#     dataset.get_data(nb_class=200, per_pic=7)
#
    # model = SiameseNet()
    # model.build_callbacks()
    # model.build_real_model()
    # print(model.model.summary())

    # model.model.load_weights(r'D:\For_Python\SomethingINT\holiday\mytry\normal_try\result\1586004511.0160203\weights.best.hdf5')
    # pred = model.model.predict([dataset.all_images[:, 0], dataset.all_images[:, 1]])
    # print(compute_accuracy(pred, dataset.all_labels, 0.47))
    # model.train_and_save(dataset)




