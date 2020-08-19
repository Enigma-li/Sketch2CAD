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
    'rootFt': 32,
    'nbThreads': 1,
}

# Variable Placeholder
userStroke_input = tf.placeholder(tf.float32, [None, None, None, 1], name='user_input')
cnormal_input = tf.placeholder(tf.float32, [None, None, None, 3], name='normal_input')
cdepth_input = tf.placeholder(tf.float32, [None, None, None, 1], name='depth_input')
gtFH_input = tf.placeholder(tf.float32, [None, None, None, 1], name='gtFH_input')
gtCurve_input = tf.placeholder(tf.float32, [None, None, None, 1], name='gtCurve_input')


# Loss term
def loss(logit_fh, logit_line, gt_fh, gt_line, user_stroke, scope='loss'):
    with tf.name_scope(scope) as _:
        # stitching face heat map (l2)
        gt_fh = slice_tensor(gt_fh, logit_fh)
        fh_loss = tf.losses.mean_squared_error(gt_fh, logit_fh)
        real_fh_loss = tf.losses.absolute_difference(gt_fh, logit_fh)

        # classification loss (l2)
        img_shape = tf.shape(logit_fh)
        N = img_shape[0]
        H = img_shape[1]
        W = img_shape[2]
        C = img_shape[3]
        full_one = tf.ones([N, H, W, C], tf.float32)
        stroke_mask = full_one - user_stroke
        logit_line = logit_line * stroke_mask
        line_loss = tf.losses.mean_squared_error(gt_line, logit_line, weights=stroke_mask)
        real_line_loss = tf.losses.absolute_difference(gt_line, logit_line, weights=stroke_mask)
        # total loss
        total_loss = fh_loss + line_loss

        return total_loss, fh_loss, real_fh_loss, line_loss, real_line_loss, gt_fh, logit_fh, \
               gt_line, logit_line, user_stroke, stroke_mask


def test_procedure(net, test_records):
    # Load data
    reader = SketchReader(tfrecord_list=test_records, raw_size=[256, 256, 17], shuffle=False,
                          num_threads=hyper_params['nbThreads'], batch_size=1, nb_epoch=1)
    raw_input = reader.next_batch()

    user_strokes, _, context_normal, context_depth, face_heatm, _, _, _, _, _, _, \
    _, _, _, line_reg = net.cook_raw_inputs(raw_input)

    # Network forward
    logit_face, logit_curve, _ = net.load_addSub_reg_net(userStroke_input,
                                                         cnormal_input,
                                                         cdepth_input,
                                                         hyper_params['rootFt'],
                                                         is_training=False)

    # Loss
    test_loss, test_fh_loss, test_real_fh_loss, test_line_loss, test_real_line_loss, \
    test_gt_face, test_pred_face, test_gt_line, test_pred_curve, test_userStroke, \
    test_strokeMask = loss(logit_face,
                           logit_curve,
                           gtFH_input,
                           gtCurve_input,
                           userStroke_input,
                           scope='test_loss')

    return test_loss, test_fh_loss, test_real_fh_loss, test_line_loss, test_real_line_loss, test_gt_face, \
           test_pred_face, test_gt_line, test_pred_curve, test_userStroke, test_strokeMask, \
           [user_strokes, context_normal, context_depth, face_heatm, line_reg]


def test_net():
    # Set logging
    test_logger = logging.getLogger('main.testing')
    test_logger.info('---Begin testing: ---')

    # Load network
    net = SKETCH2CADNET()

    # Test
    data_records = [item for item in os.listdir(hyper_params['dbTest']) if item.endswith('.tfrecords')]
    test_records = [os.path.join(hyper_params['dbTest'], item) for item in data_records if item.find('test') != -1]

    test_loss, test_fh_loss, test_real_fh_loss, test_line_loss, test_real_line_loss, \
    test_gt_face, test_pred_face, test_gt_line, test_pred_curve, test_userStroke, \
    test_strokeMask, test_inputList = test_procedure(net, test_records)

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
        tf.train.write_graph(sess.graph_def, out_net_dir,
                             "AddSub_RegTgh.pbtxt", as_text=True)
        test_logger.info('save graph tp pbtxt, done')

        # Start input enqueue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        titr = 0
        avg_loss = 0.0
        avg_fh_loss = 0.0
        avg_line_loss = 0.0
        avg_real_fh_loss = 0.0
        avg_real_line_loss = 0.0
        avg_fIOU = 0.0
        avg_bAcc = 0.0
        try:
            while not coord.should_stop():
                # get real input
                test_realInput = sess.run(test_inputList)

                t_total_loss, t_fh_loss, t_real_fh_loss, t_line_loss, t_real_line_loss, \
                t_gt_fh, t_pred_fh, t_gt_bc, t_pred_bc, t_stroke, t_strokeMask \
                    = sess.run([test_loss, test_fh_loss, test_real_fh_loss, test_line_loss, test_real_line_loss,
                                test_gt_face, test_pred_face, test_gt_line, test_pred_curve, test_userStroke,
                                test_strokeMask],
                               feed_dict={'user_input:0': test_realInput[0],
                                          'normal_input:0': test_realInput[1],
                                          'depth_input:0': test_realInput[2],
                                          'gtFH_input:0': test_realInput[3],
                                          'gtCurve_input:0': test_realInput[4]}
                               )

                # IOU
                t_gt_fh = t_gt_fh.reshape(256, 256)
                t_pred_fh = t_pred_fh.reshape(256, 256)
                t_pred_fh[t_pred_fh > 0.5] = 1.0
                t_pred_fh[t_pred_fh <= 0.5] = 0.0
                t_strokeMask = t_strokeMask.reshape(256, 256)
                t_gt_bc = t_gt_bc.reshape(256, 256)
                t_pred_bc = t_pred_bc.reshape(256, 256)
                t_pred_bc[t_pred_bc < 0.5] = 0.0
                t_pred_bc[(t_pred_bc >= 0.5) * (t_pred_bc <= 1.5)] = 1.0
                t_pred_bc[t_pred_bc > 1.5] = 2.0

                if t_gt_fh.sum() == 0:
                    fIOU_acc = 1.0
                else:
                    fIOU_acc = (t_gt_fh * t_pred_fh).sum() / (
                            t_gt_fh.sum() + t_pred_fh.sum() - (t_gt_fh * t_pred_fh).sum())
                avg_fIOU += fIOU_acc

                bAcc = (t_strokeMask * (t_pred_bc == t_gt_bc)).sum() / (t_strokeMask.sum())
                avg_bAcc += bAcc

                # Record loss
                avg_loss += t_total_loss
                avg_fh_loss += t_fh_loss
                avg_line_loss += t_line_loss
                avg_real_fh_loss += t_real_fh_loss
                avg_real_line_loss += t_real_line_loss

                test_logger.info('Test case {}, total loss: {}, fh: {}, line: {}, real_fh:{}, '
                                 'real_line: {}, fIOU: {}, bAcc: {}'.format(titr,
                                                                            t_total_loss,
                                                                            t_fh_loss,
                                                                            t_line_loss,
                                                                            t_real_fh_loss,
                                                                            t_real_line_loss,
                                                                            fIOU_acc,
                                                                            bAcc
                                                                            ))

                # Write image out
                if titr < 50:
                    fn1 = os.path.join(out_img_dir, 'pred_face_' + str(titr) + '.exr')
                    fn2 = os.path.join(out_img_dir, 'pred_curve_' + str(titr) + '.exr')
                    fn3 = os.path.join(out_img_dir, 'gt_face_' + str(titr) + '.exr')
                    fn4 = os.path.join(out_img_dir, 'gt_curve_' + str(titr) + '.exr')
                    fn5 = os.path.join(out_img_dir, 'userStroke_' + str(titr) + '.exr')
                    fn6 = os.path.join(out_img_dir, 'strokeMask_' + str(titr) + '.exr')

                    out_pred_face = t_pred_fh
                    cv2.imwrite(fn1, out_pred_face)

                    out_pred_bCurve = t_pred_bc
                    cv2.imwrite(fn2, out_pred_bCurve)

                    out_gt_face = t_gt_fh
                    cv2.imwrite(fn3, out_gt_face)

                    out_gt_curve = t_gt_bc
                    cv2.imwrite(fn4, out_gt_curve)

                    out_stroke = t_stroke[0, :, :, :]
                    out_stroke.astype(np.float32)
                    cv2.imwrite(fn5, out_stroke)

                    out_strokeMake = t_strokeMask
                    cv2.imwrite(fn6, out_strokeMake)

                titr += 1
                if titr % 100 == 0:
                    print('Iteration: {}'.format(titr))

        except tf.errors.OutOfRangeError:
            avg_loss /= titr * 1.0
            avg_fh_loss /= titr * 1.0
            avg_line_loss /= titr * 1.0
            avg_real_fh_loss /= titr * 1.0
            avg_real_line_loss /= titr * 1.0
            avg_fIOU /= titr * 1.0
            avg_bAcc /= titr * 1.0
            test_logger.info('Finish test model, average - total loss: {}, fh: {}, line: {}, real_fh: {}, '
                             'real_line: {}, fIOU: {}, bAcc: {}'.format(avg_loss,
                                                                        avg_fh_loss,
                                                                        avg_line_loss,
                                                                        avg_real_fh_loss,
                                                                        avg_real_line_loss,
                                                                        avg_fIOU,
                                                                        avg_bAcc
                                                                        ))
            print('Test Done.')

        finally:
            coord.request_stop()

        # Finish
        coord.join(threads)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbTest', required=True, help='evaluation dataset directory', type=str)
    parser.add_argument('--outDir', required=True, help='output directory', type=str)
    parser.add_argument('--devices', help='GPU device indices', type=str, default='0')
    parser.add_argument('--ckpt', help='checkpoint folder', type=str)
    parser.add_argument('--nbThr', help='n=Number of loading thread', type=int, default=1)
    parser.add_argument('--rootFt', help='root feature size', type=int, default=32)

    args = parser.parse_args()
    hyper_params['dbTest'] = args.dbTest
    hyper_params['outDir'] = args.outDir
    hyper_params['device'] = args.devices
    hyper_params['ckpt'] = args.ckpt
    hyper_params['nbThreads'] = args.nbThr
    hyper_params['rootFt'] = args.rootFt

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
