"""
展示学习曲线,并绘制图像,或其他评估方法如召回率准确率ROCF1Score等
"""
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback
from configure import *
from keras.callbacks import CSVLogger, ModelCheckpoint,TensorBoard, EarlyStopping

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type, note=None):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')

        plt.gcf().set_facecolor(np.ones(3) * 240 / 255)

        plt.grid(linestyle='--', linewidth=1, color='gray', alpha=0.4)

        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig(all_store_path + r'\learning_curve' + str(note) + '.png')


def build_callbacks(storepath=all_store_path):
    tensorboard = TensorBoard(log_dir=storepath)
    checkpoint = ModelCheckpoint(storepath + r'\weights.best.hdf5',
                                      monitor='val_loss', verbose=1,
                                      save_best_only=True, mode='min')
    earlystopping = EarlyStopping(monitor='val_acc', patience=15, verbose=2, mode='auto')
    csv_logger = CSVLogger(storepath + r'\logger.csv')
    return tensorboard, checkpoint, earlystopping, csv_logger
