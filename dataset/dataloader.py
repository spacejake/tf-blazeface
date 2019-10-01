
import tensorflow as tf
import cv2
import pickle
import glob 
import os 
import numpy as np

slim = tf.contrib.slim
import tensorflow.keras.backend as K
import utils.det_utils as det_utils
from utils.det_utils import encode_annos

_SPLITS_TO_SIZES = {
    'train': 16102,
    'validation': 12881,
}

IM_EXTENSIONS = ['png', 'jpg', 'bmp']

_NUM_CLASSES = 2

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, integer between 0 and 999',
    'label_text': 'The text of the label.',
    'object/bbox': 'A list of bounding boxes.',
    'object/label': 'A list of labels, one per each object.',
}

def read_img(img_path, img_shape=(128,128)):
    """
    load image file and divide by 255.
    """
    img = cv2.imread(img_path)
    img = cv2.resize(img, img_shape)
    img /= 255.

    return img


def dataloader(dataset_dir, label_path,  batch_size=32, img_shape=(128, 128)):

    """
    data loader

    return image, [class_label, class_and_location_label]
    """
    
    img_files = glob.glob(dataset_dir)
    img_files = [f for f in img_files if f[-3:] in IM_EXTENSIONS]

    with open(label_path, "rb") as f:
        labels = pickle.load(f)
    
    numofData = len(img_files)# endwiths(png,jpg ...)
    data_idx = np.arange(numofData)
    
    while True:
        batch_idx = np.random.choice(data_idx, size=batch_size, replace=False)
        
        batch_img = []
        batch_label = []
        batch_label_cls = []
        
        for i in batch_idx:
            
            img = read_img(img_files[i], img_shape=img_shape)
            label = labels[i]
            
            batch_img.append(img)
            batch_label.append(label)
            batch_label_cls.append(label[0:1])
            
        yield np.array(batch_img, dtype=np.float32), 
        [np.array(batch_label_cls, dtype=np.float32), np.array(batch_label, dtype=np.float32)]




def _parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {
                'image/height': tf.io.FixedLenFeature((), tf.int64, 1),
                'image/width': tf.io.FixedLenFeature((), tf.int64, 1),
                'image/filename': tf.io.FixedLenFeature((), tf.string, default_value=''),
                'image/source_id': tf.io.FixedLenFeature((), tf.string, default_value=''),
                'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
                'image/format': tf.io.FixedLenFeature((), tf.string, default_value='jpg'),
                'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
                'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
                'image/object/class/text': tf.io.VarLenFeature(tf.string),
                'image/object/class/label': tf.io.VarLenFeature(tf.int64)
                }

    # Load one example
    parsed_features = tf.parse_single_example(proto, keys_to_features)

    # Turn your saved image string into an array
    # parsed_features['image'] = tf.decode_raw(
    #     parsed_features['image'], tf.uint8)

    image = tf.image.resize_images(tf.image.decode_jpeg(parsed_features['image/encoded']), [128,128])
    label_cls = tf.sparse.to_dense(tf.cast(parsed_features['image/object/class/label'], tf.float32))
    # w = tf.cast(parsed_features['image/width'], tf.float32)
    # h = tf.cast(parsed_features['image/height'], tf.float32)

    # Dataset pre-processing has already Normalized bounding boxes
    xmin = tf.sparse.to_dense(parsed_features['image/object/bbox/xmin'])
    xmax = tf.sparse.to_dense(parsed_features['image/object/bbox/xmax'])
    ymin = tf.sparse.to_dense(parsed_features['image/object/bbox/ymin'])
    ymax = tf.sparse.to_dense(parsed_features['image/object/bbox/ymax'])

    label = tf.stack([label_cls, xmin, xmax, ymin, ymax], axis=1)

    labels_input, box_delta_input = encode_annos(label[..., :1], label[..., 1:], det_utils.ANCHORS)

    # Network learns gt bboxes of (d_cx, d_cy, d_w, d_h) in image normalzed coordinates
    delta_lable = tf.concat([labels_input, box_delta_input], axis=1)

    return image, delta_lable

def create_dataset(config):
    # This works with arrays as well
    dataset = tf.data.TFRecordDataset(config.dataset_dir)

    # Maps the parser on every filepath in the array. You can set the number of parallel loaders here
    dataset = dataset.map(_parse_function, num_parallel_calls=8)

    # This dataset will go on forever
    dataset = dataset.repeat()

    # Set the number of datapoints you want to load and shuffle
    dataset = dataset.shuffle(config.shuffle)

    # Set the batchsize
    dataset = dataset.batch(config.batch_size)

    # dataset = dataset.padded_batch(config.batch_size,
    #                                drop_remainder=False,
    #                                padded_shapes=([128, 128, 3],
    #                                               [896, 5])
    #                                )

    return dataset

def create_generator(config):
    dataset = create_dataset(config)
    iter = dataset.make_one_shot_iterator()
    batch = iter.get_next()

    while True:
        yield K.batch_get_value(batch)


def get_split(split_name, dataset_dir, file_pattern='%s.tfrecord', reader=None):
    # define your tfrecord again. Remember that you saved your image as a string.


    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/height': tf.io.FixedLenFeature((), tf.int64, 1),
        'image/width': tf.io.FixedLenFeature((), tf.int64, 1),
        'image/filename': tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/source_id': tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.io.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
        # 'image/object/class/text': tf.io.VarLenFeature(tf.string), # Bug in Current tensorflow
        'image/object/class/label': tf.io.VarLenFeature(tf.int64)
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = {
        0: 'background',
        1: 'face'
    }

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=_SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        labels_to_names=labels_to_names)

    # Load one example
    # parsed_features = tf.parse_single_example(proto, keys_to_features)

    # Turn your saved image string into an array
    # parsed_features['image'] = tf.decode_raw(
    #     parsed_features['image'], tf.uint8)

    # image = tf.image.decode_jpeg(parsed_features['image/encoded'])
    # label_cls = tf.cast(parsed_features['image/object/class/label'], tf.float32)
    # xmin = parsed_features['image/object/bbox/xmin']
    # xmax = parsed_features['image/object/bbox/xmax']
    # ymin = parsed_features['image/object/bbox/ymin']
    # ymax = parsed_features['image/object/bbox/ymax']

    # label = [np.array(label_cls),
    #           np.array([label_cls,xmin,xmax,ymin,ymax])]
    #
    # return image, label


if __name__ == "__main__":
    pass

