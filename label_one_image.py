# -*- coding: utf-8 -*-
"""
 The code derived from:  https://github.com/vietdoan/fcn-mobilenet
 Which is derivation from: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py
 The script is an excerpt from Fcn.py to perform semantic segmentation.
 All routines for training and validation are stripped away.
 
"""

from __future__ import print_function
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import Utils as utils
from collections import namedtuple
import BatchDatsetReader as dataset
import time
import SceneParsingData as scene_parsing

NUM_OF_CLASSES = 12
learning_rate =1e-4   #  "Learning rate for Adam Optimizer")
batch_size=1   # batch size for training")
logs_dir= "logs/"  # "path to logs directory")
data_dir= "Data_zoo/camvid/"  # "path to dataset")
model_dir = "Model_zoo/" # , "Path to mobile model mat")
chk_pt="model.ckpt-99500"
debug=False   #", "Debug mode: True/ False")
mode = "visualize"  # , "Mode train/ validate/ visualize")

#MODEL_URL = 'http://download.tensorflow.org/models/mobilenet_v1_1.0_224_2017_06_14.tar.gz'
MODEL_URL = 'http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_0.25_128.tgz'

IMAGE_NET_MEAN = [103.939, 116.779, 123.68]
IMAGE_SIZE = (320, 480)
MAX_ITERATION = int(1e5 + 1)
# Conv and DepthSepConv namedtuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# DepthSepConv defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=8),
    DepthSepConv(kernel=[3, 3], stride=1, depth=16),
    DepthSepConv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=32),
    DepthSepConv(kernel=[3, 3], stride=2, depth=64),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256)
]


def mobile_net(image, final_endpoint=None, num_classes=NUM_OF_CLASSES):
    with tf.variable_scope('MobilenetV1',reuse=tf.AUTO_REUSE):
        net = image
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
            for i, conv_def in enumerate(_CONV_DEFS):
                end_point_base = 'Conv2d_%d' % i
                if isinstance(conv_def, Conv):
                    net = slim.conv2d(net, conv_def.depth, conv_def.kernel,
                                      stride=conv_def.stride,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point_base)
                elif isinstance(conv_def, DepthSepConv):
                    end_point = end_point_base + '_depthwise'
                    net = slim.separable_conv2d(net, None, conv_def.kernel,
                                                depth_multiplier=1,
                                                stride=conv_def.stride,
                                                rate=1,
                                                normalizer_fn=slim.batch_norm,
                                                scope=end_point)

                    end_point = end_point_base + '_pointwise'
                    net = slim.conv2d(net, conv_def.depth, [1, 1],
                                      stride=1,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point)
                if final_endpoint and final_endpoint == end_point_base:
                    break
    return net


def inference(image, dropout_keep_prob, num_classes=NUM_OF_CLASSES):
    print("setting up mobile initialized conv layers ...")
    mean = tf.constant(IMAGE_NET_MEAN)
    image -= mean
    net = mobile_net(image, num_classes=num_classes)

    with tf.variable_scope('inference',reuse=tf.AUTO_REUSE):
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout')
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                          normalizer_fn=None, scope='Conv2d_1x1')
        net = slim.convolution2d_transpose(net, num_classes, 64, 32)

        annotation_pred = tf.argmax(net, axis=3, name="prediction")
        
    return tf.expand_dims(annotation_pred, dim=3), net




def main(argv=None):
    
    utils.get_model_data(model_dir, MODEL_URL)
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder( tf.float32, shape=[None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3], name="input_image")
    annot = tf.placeholder( tf.int32  , shape=[None, IMAGE_SIZE[0], IMAGE_SIZE[1], 1], name="annotation" )

    pred_annotation, logits = inference(image, dropout_keep_prob=keep_probability)
        
    # Dataset_reader reads and transforms WHOLE dataset into memory so for 1 image job it is crazy overkill
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    train_records, _, _ = scene_parsing.read_dataset(data_dir)
    train_dataset_reader = dataset.BatchDatset( train_records, image_options)
    # Dataset_reader reads and transforms WHOLE dataset into memory so for 1 image job it is crazy overkill

    sess = tf.Session()
    
    
    if mode == "visualize":
        saver = tf.train.Saver()
        # print_tensors_in_checkpoint_file(file_name='logs/model.ckpt-5500', tensor_name='', all_tensors=False)
        saver.restore(sess, logs_dir+chk_pt)
        #to read a batch of images USE next line a dataset_reader.
        # images, _ = train_dataset_reader.get_random_batch(batch_size)
        # To read a specific image use next line READ_ONE
        images = train_dataset_reader.read_one_image(data_dir + 'train/20180429_081127.png')
        t1 = time.time()
        pred = sess.run(pred_annotation, feed_dict={image: images, keep_probability: 1.0})
        t2 = time.time()
        print('time sec. elapsed:',t2 - t1)
        #annotations = np.squeeze(annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(batch_size):
            utils.save_image(images[itr].astype(np.uint8), logs_dir, name="1inp_" + str(5 + itr))
           # utils.save_image(utils.decode_segmap(annotations[itr]), logs_dir, name="1gt_" + str(5 + itr))
            utils.save_image(utils.decode_segmap(pred[itr]), logs_dir, name="1pred_" + str(5 + itr))

    sess.close()
if __name__ == "__main__":
    tf.app.run()
