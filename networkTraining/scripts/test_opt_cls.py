#
# Project Sketch2CAD
#
#   Author: Changjian Li (chjili2011@gmail.com),
#   Copyright (c) 2020. All Rights Reserved.
#
# ==============================================================================
"""Bevel operator parameters regression network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
import argparse
import time

import tensorflow as tf
from Sketch2CADNet.scripts.network import SKETCH2CADNET
from Sketch2CADNet.scripts.loader import SketchReader
from Sketch2CADNet.utils.util_funcs import slice_tensor, make_dir, dump_params

import cv2
import numpy as np

# Hyper Parameters
hyper_params = {
    'dbTest': '',
    'outDir': '',
    'device': '0',
    'ckpt': '',
    'nbThreads': 1,
    'nb_cls': 3,
}


# Loss def
def loss(logit_prob, gt_label, scope='loss'):
    with tf.name_scope(scope) as _:
        # convert to onehot vector
        onehot_labels = tf.one_hot(gt_label, hyper_params['nb_cls'])

        # Extrude: 50000, addSub: 200000, bevel: 50000, sweep: 100000 - sum: 400000
        # Weight: normalize(400000/50000, 400000/200000, 400000/50000, 400000/100000)
        class_weights = tf.constant([[0.3636, 0.0909, 0.3636, 0.1819]])
        # deduce weights for batch samples based on their true label
        weights = tf.reduce_sum(class_weights * onehot_labels, axis=1)
        # compute your (unweighted) softmax cross entropy loss
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logit_prob)
        # apply the weights, relying on broadcasting of the multiplication
        weighted_losses = unweighted_losses * weights
        # reduce the result to get your final loss
        cost = tf.reduce_mean(weighted_losses)

    return cost


# Accuracy
def cal_accuracy(logit_prob, gt_labels, scope='accuracy'):
    with tf.name_scope(scope) as _:
        onehot_label = tf.one_hot(gt_labels, hyper_params['nb_cls'])
        correct = tf.equal(tf.argmax(logit_prob, 1), tf.argmax(onehot_label, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct) * 100.0

    return accuracy


def test_procedure(net, test_records):
    # load data
    reader = SketchReader(tfrecord_list=test_records, raw_size=[256, 256, 17], shuffle=False,
                          num_threads=hyper_params['nbThreads'], batch_size=1, nb_epoch=1)
    raw_input = reader.next_batch()

    user_strokes, _, context_normal, context_depth, gt_face, gt_bCurve, gt_pCurve, gt_oCurve, \
    _, _, _, _, opt_label, _, _ = net.cook_raw_inputs(raw_input)

    # network forward
    logit_prob, _ = net.load_cls_opt_tiny_net(user_strokes,
                                              context_normal,
                                              context_depth,
                                              num_classes=hyper_params['nb_cls'],
                                              is_training=False)

    # Loss
    val_loss = loss(logit_prob, opt_label, scope='test_loss')
    val_acc = cal_accuracy(logit_prob, opt_label, scope='test_acc')

    return val_loss, val_acc, \
           [user_strokes, context_normal, context_depth, gt_face, gt_bCurve, gt_pCurve, gt_oCurve]


def test_net():
    # Set logging
    test_logger = logging.getLogger('main.testing')
    test_logger.info('---Begin testing: ---')

    # Load network
    net = SKETCH2CADNET()

    # Test
    data_records = [item for item in os.listdir(hyper_params['dbTest']) if item.endswith('.tfrecords')]
    test_records = [os.path.join(hyper_params['dbTest'], item) for item in data_records if item.find('test') != -1]

    test_bce_loss, test_acc_loss, test_inputList = test_procedure(net, test_records)

    # Saver
    tf_saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # config.log_device_placement = True

    with tf.Session(config=config) as sess:
        # initialize
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # Restore model
        ckpt = tf.train.latest_checkpoint(hyper_params['ckpt'])
        if ckpt:
            tf_saver.restore(sess, ckpt)
            test_logger.info('restore from the checkpoint {}'.format(ckpt))

        # writeGraph:
        tf.train.write_graph(sess.graph_def, out_net_dir, "Opt_Cls.pbtxt", as_text=True)
        test_logger.info('save graph tp pbtxt, done')

        # Start input enqueue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            titr = 0
            avg_loss = 0.0
            avg_acc = 0.0
            while not coord.should_stop():
                t_loss, t_acc_loss, t_input_list = sess.run([test_bce_loss, test_acc_loss, test_inputList])

                avg_loss += t_loss
                avg_acc += t_acc_loss

                test_logger.info('Test case {}, bce: {}, acc: {}'.format(titr, t_loss, t_acc_loss))

                if titr < 50:
                    fn1 = os.path.join(out_img_dir, 'out_img_stroke_' + str(titr) + '.exr')
                    fn3 = os.path.join(out_img_dir, 'out_img_normal_' + str(titr) + '.exr')
                    fn4 = os.path.join(out_img_dir, 'out_img_depth_' + str(titr) + '.exr')
                    fn5 = os.path.join(out_img_dir, 'gt_face_map_' + str(titr) + '.exr')
                    fn6 = os.path.join(out_img_dir, 'gt_base_curve_' + str(titr) + '.exr')
                    fn7 = os.path.join(out_img_dir, 'gt_profile_curve_' + str(titr) + '.exr')
                    fn8 = os.path.join(out_img_dir, 'gt_offset_curve_' + str(titr) + '.exr')

                    out_userStroke = t_input_list[0][0, :, :, :]
                    out_userStroke.astype(np.float32)
                    cv2.imwrite(fn1, out_userStroke)

                    out_normal = t_input_list[1][0, :, :, :]
                    out_normal = out_normal[:, :, [2, 1, 0]]
                    out_normal.astype(np.float32)
                    cv2.imwrite(fn3, out_normal)

                    out_depth = t_input_list[2][0, :, :, :]
                    out_depth.astype(np.float32)
                    cv2.imwrite(fn4, out_depth)

                    gt_face = t_input_list[3][0, :, :, :]
                    gt_face.astype(np.float32)
                    cv2.imwrite(fn5, gt_face)

                    gt_bCurve = t_input_list[4][0, :, :, :]
                    gt_bCurve.astype(np.float32)
                    cv2.imwrite(fn6, gt_bCurve)

                    gt_proCurve = t_input_list[5][0, :, :, :]
                    gt_proCurve.astype(np.float32)
                    cv2.imwrite(fn7, gt_proCurve)

                    gt_offCurve = t_input_list[6][0, :, :, :]
                    gt_offCurve.astype(np.float32)
                    cv2.imwrite(fn8, gt_offCurve)

                titr += 1
                if titr % 100 == 0:
                    print('Iteration: {}'.format(titr))
        except tf.errors.OutOfRangeError:
            avg_loss /= titr
            avg_acc /= titr
            test_logger.info('Finish test mode, average bce: {}, acc: {}'.format(avg_loss, avg_acc))

            print('Test Done.')
        finally:
            coord.request_stop()

        # Finish testing
        coord.join(threads)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbTest', required=True, help='evaluation dataset directory', type=str)
    parser.add_argument('--outDir', required=True, help='output directory', type=str)
    parser.add_argument('--devices', help='GPU device indices', type=str, default='0')
    parser.add_argument('--ckpt', help='checkpoint folder', type=str)
    parser.add_argument('--nbThr', help='n=Number of loading thread', type=int, default=1)
    parser.add_argument('--nb_cls', help='number of classes', type=int, default=4)

    args = parser.parse_args()
    hyper_params['dbTest'] = args.dbTest
    hyper_params['outDir'] = args.outDir
    hyper_params['device'] = args.devices
    hyper_params['ckpt'] = args.ckpt
    hyper_params['nbThreads'] = args.nbThr
    hyper_params['nb_cls'] = args.nb_cls

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = hyper_params['device']

    # out img dir
    out_folder = os.path.join(hyper_params['outDir'], 'test')
    hyper_params['outDir'] = out_folder
    out_img_dir = os.path.join(out_folder, 'out_img')
    make_dir(out_img_dir)
    out_net_dir = os.path.join(out_folder, 'pbtxt')
    make_dir(out_net_dir)

    # Set logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(out_folder, 'log.txt'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(fh)

    # Training preparation
    logger.info('---Test preparation: ---')

    # Dump parameters
    dump_params(out_folder, hyper_params)

    # Begin training
    test_net()
