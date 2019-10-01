import argparse

from model.blazeface import BlazeFace

if __name__ == "__main__":

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

    if config.train:
        blazeface.train()


