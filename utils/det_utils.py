import tensorflow as tf
import numpy as np
import cv2

def get_anchor_scale(a=0):
    return a * 0.5 + 1.0


def get_anchors(feat_size, numAnchors, image_size=128):
    anchor_size = feat_size / image_size

    # anchors = tf.zeros([feat_size, feat_size, numAnchors, 4],
    #                    dtype=tf.float32,
    #                    name=None)

    # center coords of each anchor
    cx = cy = np.linspace(anchor_size / 2, 1.0 - anchor_size / 2, feat_size)
    CX, CY = np.meshgrid(cx, cy)
    CX = CX.reshape((feat_size, feat_size))
    CY = CY.reshape((feat_size, feat_size))
    # anchors_c = tf.convert_to_tensor(np.stack((CX, CY), axis=-1), dtype=tf.float32)
    np_anchors = np.zeros((feat_size, feat_size, numAnchors, 4), dtype=np.float32)
    np_anchors[..., :2] = np.expand_dims(np.stack((CX, CY), axis=-1), 2)

    for anchor_i in range(0, numAnchors):
        # Get normalized default bbox (cx, cy, w, h)
        anchor_dim = get_anchor_scale(anchor_i) * anchor_size
        np_anchors[..., anchor_i, 2:] = anchor_dim

    return tf.reshape(tf.convert_to_tensor(np_anchors, dtype=tf.float32),
                      [feat_size * feat_size * numAnchors, 4])  # , np_anchors


ANCHORS = tf.concat([get_anchors(feat_size=16, numAnchors=2),
                     get_anchors(feat_size=8, numAnchors=6)], axis=0)

def xywh_to_yxyx(bbox):
    shape = bbox.get_shape().as_list()
    _axis = 1 if len(shape) > 1 else 0
    [x, y, w, h] = tf.unstack(bbox, axis=_axis)
    y_min = y - 0.5 * h
    x_min = x - 0.5 * w
    y_max = y + 0.5 * h
    x_max = x + 0.5 * w
    return tf.stack([y_min, x_min, y_max, x_max], axis=_axis)


def yxyx_to_xywh(bbox):
    y_min = bbox[:, 0]
    x_min = bbox[:, 1]
    y_max = bbox[:, 2]
    x_max = bbox[:, 3]
    x = (x_min + x_max) * 0.5
    y = (y_min + y_max) * 0.5
    w = x_max - x_min
    h = y_max - y_min
    return tf.stack([x, y, w, h], axis=1)


def batch_iou_fast(anchors, bboxes):
    """ Compute iou of two batch of boxes. Box format '[y_min, x_min, y_max, x_max]'.
    Args:
      anchors: know shape
      bboxes: dynamic shape
    Return:
      ious: 2-D with shape '[num_bboxes, num_anchors]'
    """
    num_anchors = anchors.get_shape().as_list()[0]
    num_bboxes = tf.shape(bboxes)[0]

    box_indices = tf.reshape(tf.range(num_bboxes), shape=[-1, 1])
    box_indices = tf.reshape(tf.stack([box_indices] * num_anchors, axis=0), shape=[-1, 1])
    # box_indices = tf.Print(box_indices, [box_indices], "box_indices", summarize=100)
    # box_indices = tf.concat([box_indices] * num_anchors, axis=0)
    # box_indices = tf.Print(box_indices, [box_indices], "box_indices", summarize=100)
    bboxes_m = tf.gather_nd(bboxes, box_indices)

    anchors_m = tf.tile(anchors, [num_bboxes, 1])

    lr = tf.maximum(
        tf.minimum(bboxes_m[:, 3], anchors_m[:, 3]) -
        tf.maximum(bboxes_m[:, 1], anchors_m[:, 1]),
        0
    )
    tb = tf.maximum(
        tf.minimum(bboxes_m[:, 2], anchors_m[:, 2]) -
        tf.maximum(bboxes_m[:, 0], anchors_m[:, 0]),
        0
    )

    intersection = tf.multiply(tb, lr)

    union = tf.subtract(
        tf.multiply((bboxes_m[:, 3] - bboxes_m[:, 1]), (bboxes_m[:, 2] - bboxes_m[:, 0])) +
        tf.multiply((anchors_m[:, 3] - anchors_m[:, 1]), (anchors_m[:, 2] - anchors_m[:, 0])),
        intersection
    )

    ious = tf.truediv(intersection, union)

    ious = tf.reshape(ious, shape=[num_bboxes, num_anchors])

    return ious


def arg_closest_anchor(bboxes, anchors):
    """Find the closest anchor. Box Format [ymin, xmin, ymax, xmax]
    """
    num_anchors = anchors.get_shape().as_list()[0]
    num_bboxes = tf.shape(bboxes)[0]

    _indices = tf.reshape(tf.range(num_bboxes), shape=[-1, 1])
    _indices = tf.reshape(tf.stack([_indices] * num_anchors, axis=1), shape=[-1, 1])
    bboxes_m = tf.gather_nd(bboxes, _indices)
    # bboxes_m = tf.Print(bboxes_m, [bboxes_m], "bboxes_m", summarize=100)

    anchors_m = tf.tile(anchors, [num_bboxes, 1])
    # anchors_m = tf.Print(anchors_m, [anchors_m], "anchors_m", summarize=100)

    square_dist = tf.squared_difference(bboxes_m[:, 0], anchors_m[:, 0]) + \
                  tf.squared_difference(bboxes_m[:, 1], anchors_m[:, 1]) + \
                  tf.squared_difference(bboxes_m[:, 2], anchors_m[:, 2]) + \
                  tf.squared_difference(bboxes_m[:, 3], anchors_m[:, 3])

    square_dist = tf.reshape(square_dist, shape=[num_bboxes, num_anchors])

    # square_dist = tf.Print(square_dist, [square_dist], "square_dist", summarize=100)

    indices = tf.arg_min(square_dist, dimension=1)

    return indices


def batch_delta(bboxes, anchors):
    """
    Args:
       bboxes: [num_bboxes, 4]
       anchors: [num_bboxes, 4]
    Return:
      deltas: [num_bboxes, 4]
    """
    bbox_x, bbox_y, bbox_w, bbox_h = tf.unstack(bboxes, axis=1)
    anchor_x, anchor_y, anchor_w, anchor_h = tf.unstack(anchors, axis=1)
    delta_x = (bbox_x - anchor_x) / bbox_w
    delta_y = (bbox_y - anchor_y) / bbox_h
    delta_w = tf.log(bbox_w / anchor_w)
    delta_h = tf.log(bbox_h / anchor_h)
    return tf.stack([delta_x, delta_y, delta_w, delta_h], axis=1)


def encode_annos(labels, bboxes, anchors):
    """Encode annotations for losses computations.
    All the output tensors have a fix shape(none dynamic dimention).

    Args:
      labels: 1-D with shape `[num_bounding_boxes]`.
      bboxes: 2-D with shape `[num_bounding_boxes, 4]`. Format [ymin, xmin, ymax, xmax]
      anchors: 4-D tensor with shape `[num_anchors, 4]`. Format [cx, cy, w, h]

    Returns:
      input_mask: 2-D with shape `[num_anchors, 1]`, indicate which anchor to be used to cal loss.
      labels_input: 2-D with shape `[num_anchors, 1]`, 0 for background, 1 for face.
      box_delta_input: 2-D with shape `[num_anchors, 4]`. Format [dcx, dcy, dw, dh]
      box_input: 2-D with shape '[num_anchors, 4]'. Format [ymin, xmin, ymax, xmax]
    """
    with tf.name_scope("Encode_annotations") as scope:
        num_anchors = anchors.get_shape().as_list()[0]
        num_bboxes = tf.shape(bboxes)[0]

        # Cal iou, find the target anchor
        with tf.name_scope("Matching") as subscope:
            ious = batch_iou_fast(xywh_to_yxyx(anchors), bboxes)
            anchor_indices = tf.reshape(tf.arg_max(ious, dimension=1), shape=[-1, 1])  # target anchor indices
            # anchor_indices = tf.Print(anchor_indices, [anchor_indices], summarize=100)

            with tf.name_scope("Deal_with_noneoverlap"):
                # find the none-overlap bbox
                bbox_indices = tf.reshape(tf.range(num_bboxes), shape=[-1, 1])
                iou_indices = tf.concat([bbox_indices, tf.cast(anchor_indices, dtype=tf.int32)], axis=1)
                target_iou = tf.gather_nd(ious, iou_indices)
                none_overlap_bbox_indices = tf.where(target_iou <= 0)  # 1-D
                # find it's corresponding anchor
                closest_anchor_indices = arg_closest_anchor(tf.gather_nd(bboxes, none_overlap_bbox_indices),
                                                            anchors)  # 1-D

        with tf.name_scope("Delta") as subscope:
            target_anchors = tf.gather_nd(anchors, anchor_indices)
            bboxes = yxyx_to_xywh(bboxes)
            delta = batch_delta(bboxes, target_anchors)

        with tf.name_scope("Scattering") as subscope:
            # tf.scatter_nd initilizes tensor as zeros

            # bbox
            # box_input = tf.scatter_nd(anchor_indices,
            #                           bboxes,
            #                           shape=[num_anchors, 4]
            #                           )

            # label
            labels_input = tf.scatter_nd(anchor_indices,
                                         labels,
                                         shape=[num_anchors, 1]
                                         )

            # anchor mask
            # input_mask = tf.scatter_nd(anchor_indices,
            #                            tf.ones([num_bboxes]),
            #                            shape=[num_anchors])
            # input_mask = tf.reshape(input_mask, shape=[-1, 1])

            # delta
            box_delta_input = tf.scatter_nd(anchor_indices,
                                            delta,
                                            shape=[num_anchors, 4]
                                            )

    # return input_mask, labels_input, box_delta_input, box_input
    return labels_input, box_delta_input

    # def encode_annos(gt_bbox, anchors):
    '''
    arguments must be normalized.
    The dataloader and anchor generation preprocesses the objects to be in their normalized state
    '''


def main():
    print(get_anchors(feat_size=4, numAnchors=2, image_size=12))


if __name__ == "__main__":
    main()
