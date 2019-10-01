import tensorflow as tf
import numpy as np
import argparse

from model.network import network
from model.blazeface import BlazeFace


def test_network_build():
    """
    check whether the network is built sucessfully or not 
    """
    x = np.float32(np.random.random((3, 128, 128, 3)))

    blazeface_extractor = network((128, 128, 3))
    feature = blazeface_extractor(x)
    print(feature)
    assert feature[0].shape == (
        3, 16, 16, 96) or feature[1].shape == (3, 8, 8, 96)

def test_blazeface():
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--input_shape', type=int, default=128)
    args.add_argument('--input_channels', type=int, default=3)
    args.add_argument('--batch_size', type=int, default=128)
    args.add_argument('--nb_epoch', type=int, default=1000)
    args.add_argument('--numdata', type=int, default=2625)
    args.add_argument('--train', type=bool, default=True)
    args.add_argument('--shuffle', type=bool, default=True)
    args.add_argument('--checkpoint_path', type=str, default="./")
    args.add_argument('--dataset_dir', type=str, default="./")
    args.add_argument('--label_path', type=str, default="./")

    config = args.parse_args()

    blazeface = BlazeFace(config)
    x = np.float32(np.random.random((1, 128, 128, 3)))

    predictions = blazeface.model.predict(x)
    print('predictions shape:', predictions)


if __name__ == "__main__":
    test_network_build()
    test_blazeface()
