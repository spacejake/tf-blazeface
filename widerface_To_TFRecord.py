from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy
import cv2
import os
import hashlib
import argparse

from dataset import widerface_parser

# lineno = 0

def main(unused_argv):
    args = argparse.ArgumentParser()
    args.add_argument('--dataset_dir', type=str, default="./dataset/data/widerface/WIDER_train")
    args.add_argument('--output_path', type=str, default="./dataset/data/widerface/train.tfrecord")

    config = args.parse_args()

    """
    Instructions:
    1) Rename dataset split annotaion file for training to 'wider_face_train_anno.txt'
    2) move annotaion splits into WIDER_train and WIDER_val 
    """
    # lineno = 0
    f = open(os.path.join(config.dataset_dir, "wider_face_train_anno.txt"))
    writer = tf.python_io.TFRecordWriter(config.output_path)

    # WIDER FACE DATASET ANNOTATED 12880 IMAGES
    valid_image_num = 0
    invalid_image_num = 0
    for image_idx in range(12880):
        print("image idx is %d" % image_idx)
        valid_face_number, tf_example = widerface_parser.parse_example(f, config.dataset_dir)
        if (valid_face_number != 0):
            writer.write(tf_example.SerializeToString())
            valid_image_num += 1
        else:
            invalid_image_num += 1
            print("Pass!")
    writer.close()

    print("Valid image number is %d" % valid_image_num)
    print("Invalid image number is %d" % invalid_image_num)


if __name__ == '__main__':
    tf.app.run()
