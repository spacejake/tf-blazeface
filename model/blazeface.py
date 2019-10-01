import tensorflow as tf
import numpy as np
import cv2
import pickle
import glob
import os
import time

from model.network import network
from model.loss import loss
from utils.utils import get_iou
from dataset.dataloader import dataloader, create_dataset, create_generator
from utils.det_utils import get_anchors

class BlazeFace():

    def __init__(self, config):

        self.input_shape = (config.input_shape, config.input_shape, config.input_channels)
        self.feature_extractor = network(self.input_shape)

        self.n_boxes = [2, 6]  # 2 for 16x16, 6 for 8x8

        self.model = self.build_model()

        if config.train:
            self.batch_size = config.batch_size
            self.nb_epoch = config.nb_epoch

        self.checkpoint_path = config.checkpoint_path
        self.numdata = config.numdata
        self.config = config

    def build_model(self):

        model = self.feature_extractor

        # 16x16 bounding box - Confidence, [batch_size, 16, 16, 2]
        bb_16_conf = tf.keras.layers.Conv2D(filters=self.n_boxes[0] * 1,
                                            kernel_size=3,
                                            padding='same',
                                            activation='sigmoid')(model.output[0])
        # reshape [batch_size, 16**2 * #bbox(2), 1]
        bb_16_conf_reshaped = tf.keras.layers.Reshape((16 ** 2 * self.n_boxes[0], 1))(bb_16_conf)

        # 8 x 8 bounding box - Confindece, [batch_size, 8, 8, 6]
        bb_8_conf = tf.keras.layers.Conv2D(filters=self.n_boxes[1] * 1,
                                           kernel_size=3,
                                           padding='same',
                                           activation='sigmoid')(model.output[1])
        # reshape [batch_size, 8**2 * #bbox(6), 1]
        bb_8_conf_reshaped = tf.keras.layers.Reshape((8 ** 2 * self.n_boxes[1], 1))(bb_8_conf)
        # Concatenate confidence prediction

        # shape : [batch_size, 896, 1]
        conf_of_bb = tf.keras.layers.Concatenate(axis=1)([bb_16_conf_reshaped, bb_8_conf_reshaped])

        # 16x16 bounding box - loc [x, y, w, h]
        bb_16_loc = tf.keras.layers.Conv2D(filters=self.n_boxes[0] * 4,
                                           kernel_size=3,
                                           padding='same')(model.output[0])
        # [batch_size, 16**2 * #bbox(2), 4]
        bb_16_loc_reshaped = tf.keras.layers.Reshape((16 ** 2 * self.n_boxes[0], 4))(bb_16_loc)

        # 8x8 bounding box - loc [x, y, w, h]
        bb_8_loc = tf.keras.layers.Conv2D(filters=self.n_boxes[1] * 4,
                                          kernel_size=3,
                                          padding='same')(model.output[1])
        bb_8_loc_reshaped = tf.keras.layers.Reshape((8 ** 2 * self.n_boxes[1], 4))(bb_8_loc)
        # Concatenate  location prediction

        loc_of_bb = tf.keras.layers.Concatenate(axis=1)([bb_16_loc_reshaped, bb_8_loc_reshaped])

        output = tf.keras.layers.Concatenate(axis=-1)([conf_of_bb, loc_of_bb])

        # Detectors model
        # return tf.keras.models.Model(model.input, [conf_of_bb, output])
        return tf.keras.models.Model(model.input, output)

    def train(self):

        opt = tf.keras.optimizers.Adam(amsgrad=True)
        model = self.model
        model.compile(loss=[loss], optimizer=opt)

        """ Callback """
        monitor = 'loss'
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor, patience=4)

        """Callback for Tensorboard"""
        tb = tf.keras.callbacks.TensorBoard(log_dir="./logs/", update_freq='batch')

        """ Training loop """

        STEP_SIZE_TRAIN = self.numdata // self.batch_size

        t0 = time.time()

        # # data_gen = dataloader(config.dataset_dir, config.label_path, self.batch_size)
        # anchors = tf.concat([get_anchors(feat_size=16, numAnchors=2),
        #                      get_anchors(feat_size=8, numAnchors=6)], axis=0)


        data_gen = create_generator(self.config)

        for epoch in range(self.nb_epoch):
            t1 = time.time()
            res = model.fit_generator(generator=data_gen,
                                      steps_per_epoch=STEP_SIZE_TRAIN,
                                      initial_epoch=epoch,
                                      epochs=epoch + 1,
                                      callbacks=[reduce_lr, tb],
                                      verbose=1,
                                      shuffle=True)
            t2 = time.time()

            print(res.history)

            print('Training time for one epoch : %.1f' % ((t2 - t1)))

            if epoch % 100 == 0:
                model.save_weights(os.path.join(self.config.checkpoint_path, str(epoch)))

        print('Total training time : %.1f' % (time.time() - t0))