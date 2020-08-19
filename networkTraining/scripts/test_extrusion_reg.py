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

userStroke_input = tf.placeholder(tf.float32, [None, None, None, 1], name='user_input')
cnormal_input = tf.placeholder(tf.float32, [None, None, None, 3], name='normal_input')
cdepth_input = tf.placeholder(tf.float32, [None, None, None, 1], name='depth_input')
gtFH_input = tf.placeholder(tf.float32, [None, None, None, 1], name='gtFH_input')
gtoffCurve_input = tf.placeholder(tf.float32, [None, None, None, 1], name='gtoffCurve_input')


# Loss term
def loss(logit_fh, logit_curve, fh, curve, user_stroke, normal, depth, scope='loss'):
    with tf.name_scope(scope) as _:
        # stitching face heat map (l2)
        gt_fh = slice_tensor(fh, logit_fh)
        fh_loss = tf.losses.mean_squared_error(gt_fh, logit_fh)
        real_fh_loss = tf.losses.absolute_difference(gt_fh, logit_fh)

        # curve loss (l2)
        curve_mask = slice_tensor(curve, logit_curve)
        img_shape = tf.shape(logit_fh)
        N = img_shape[0]
        H = img_shape[1]
        W = img_shape[2]
        C = img_shape[3]
        full_one = tf.ones([N, H, W, C], tf.float32)
        full_zero = tf.zeros([N, H, W, C], tf.float32)
        stroke_mask = full_one - user_stroke
        diff_mask = stroke_mask - curve_mask

        fg_pred = logit_curve * curve_mask
        bg_pred = logit_curve * diff_mask
        curve_fg_sum = tf.reduce_sum(tf.pow(curve_mask - fg_pred, 2), axis=[1, 2, 3])  # offset curve loss sum
        curve_bg_sum = tf.reduce_sum(tf.pow(full_zero - bg_pred, 2), axis=[1, 2, 3])  # orthogonal curve loss sum
        nb_stroke_pixels = tf.reduce_sum(stroke_mask, axis=[1, 2, 3])  # number of stroke pixels
        curve_loss = (curve_fg_sum + curve_bg_sum) / nb_stroke_pixels  # mean loss
        c_loss = tf.reduce_mean(curve_loss)

        real_curve_fg_sum = tf.reduce_sum(tf.abs(curve_mask - fg_pred), axis=[1, 2, 3])
        real_curve_bg_sum = tf.reduce_sum(tf.abs(full_zero - bg_pred), axis=[1, 2, 3])
        real_curve_loss = (real_curve_fg_sum + real_curve_bg_sum) / nb_stroke_pixels
        real_c_loss = tf.reduce_mean(real_curve_loss)

        total_loss = fh_loss + c_loss

        logit_curve = logit_curve * stroke_mask

        return total_loss, fh_loss, c_loss, real_fh_loss, real_c_loss, logit_fh, logit_curve, fh, curve, user_stroke, \
               normal, depth, curve_mask, stroke_mask


def test_procedure(net, test_records):
    # Load data
    reader = SketchReader(tfrecord_list=test_records, raw_size=[256, 256, 17], shuffle=False,
                          num_threads=hyper_params['nbThreads'], batch_size=1, nb_epoch=1)
    raw_input = reader.next_batch()

    user_strokes, _, context_normal, context_depth, face_heatm, _, _, off_curve, _, _, _, _, _, \
    _, _ = net.cook_raw_inputs(raw_input)

    # Network forward
    logit_fh, logit_curve, _ = net.load_extrusion_reg_net(userStroke_input,
                                                          cnormal_input,
                                                          cdepth_input,
                                                          hyper_params['rootFt'],
                                                          is_training=False)

    # losses
    test_total_loss, test_fh_loss, test_c_loss, test_real_fh_loss, test_real_c_loss, test_pred_fh, test_pred_c, \
    test_gt_fh, test_gt_c, test_userStroke, test_normal, test_depth, test_curveMask, strokeMask \
        = loss(logit_fh,
               logit_curve,
               gtFH_input,
               gtoffCurve_input,
               userStroke_input,
               cnormal_input,
               cdepth_input,
               scope='test_loss')

    return test_total_loss, test_fh_loss, test_c_loss, test_real_fh_loss, test_real_c_loss, test_pred_fh, test_pred_c, \
           test_gt_fh, test_gt_c, test_userStroke, test_normal, test_depth, test_curveMask, strokeMask, \
           [user_strokes, context_normal, context_depth, face_heatm, off_curve]


def test_net():
    # Set logging
    test_logger = logging.getLogger('main.testing')
    test_logger.info('---Begin testing: ---')

    # Load network
    net = SKETCH2CADNET()

    # Test
    data_records = [item for item in os.listdir(hyper_params['dbTest']) if item.endswith('.tfrecords')]
    test_records = [os.path.join(hyper_params['dbTest'], item) for item in data_records if item.find('test') != -1]

    test_total_loss, test_fh_loss, test_c_loss, test_real_fh_loss, test_real_c_loss, test_pred_fh, test_pred_c, \
    test_gt_fh, test_gt_c, test_userStroke, test_normal, test_depth, test_curveMask, \
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
                             "Extrusion_Reg.pbtxt", as_text=True)
        test_logger.info('save graph tp pbtxt, done')

        # Start input enqueue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            titr = 0
            avg_loss = 0.0
            avg_fh_loss = 0.0
            avg_c_loss = 0.0
            avg_real_fh_loss = 0.0
            avg_real_c_loss = 0.0
            avg_fIOU = 0.0
            avg_bAcc = 0.0
            while not coord.should_stop():
                # get real input
                test_realInput = sess.run(test_inputList)

                t_total_loss, t_fh_loss, t_c_loss, t_real_fh_loss, t_real_c_loss, t_pred_fh, t_pred_bc, t_gt_fh, \
                t_gt_bc, t_userStroke, t_normal, t_depth, t_curveMask, t_strokeMask = sess.run(
                    [test_total_loss, test_fh_loss, test_c_loss, test_real_fh_loss, test_real_c_loss, test_pred_fh,
                     test_pred_c, test_gt_fh, test_gt_c, test_userStroke, test_normal, test_depth,
                     test_curveMask, test_strokeMask],
                    feed_dict={'user_input:0': test_realInput[0],
                               'normal_input:0': test_realInput[1],
                               'depth_input:0': test_realInput[2],
                               'gtFH_input:0': test_realInput[3],
                               'gtoffCurve_input:0': test_realInput[4]})

                # Record loss
                avg_loss += t_total_loss
                avg_fh_loss += t_fh_loss
                avg_c_loss += t_c_loss
                avg_real_fh_loss += t_real_fh_loss
                avg_real_c_loss += t_real_c_loss

                # IOU
                t_gt_fh = t_gt_fh.reshape(256, 256)
                t_pred_fh = t_pred_fh.reshape(256, 256)
                t_pred_fh[t_pred_fh > 0.5] = 1.0
                t_pred_fh[t_pred_fh <= 0.5] = 0.0
                t_strokeMask = t_strokeMask.reshape(256, 256)
                t_gt_bc = t_gt_bc.reshape(256, 256)
                t_pred_bc = t_pred_bc.reshape(256, 256)
                t_pred_bc[t_pred_bc > 0.01] = 1.0
                t_pred_bc[t_pred_bc <= 0.01] = 0.0

                if t_gt_fh.sum() == 0:
                    fIOU_acc = 1.0
                else:
                    fIOU_acc = (t_gt_fh * t_pred_fh).sum() / (
                            t_gt_fh.sum() + t_pred_fh.sum() - (t_gt_fh * t_pred_fh).sum())
                avg_fIOU += fIOU_acc

                bAcc = (t_strokeMask * (t_pred_bc == t_gt_bc)).sum() / (t_strokeMask.sum())
                avg_bAcc += bAcc

                test_logger.info('Test case {}, total loss: {}, fh: {}, cv: {}, rfh: {}, '
                                 'rcv: {}, fIOU: {}, bAcc: {}'.format(titr,
                                                                      t_total_loss,
                                                                      t_fh_loss,
                                                                      t_c_loss,
                                                                      t_real_fh_loss,
                                                                      t_real_c_loss,
                                                                      fIOU_acc,
                                                                      bAcc))

                # write image out
                if titr < 50:
                    fn1 = os.path.join(out_img_dir, 'userStroke_' + str(titr) + '.exr')
                    fn2 = os.path.join(out_img_dir, 'gt_curve_' + str(titr) + '.exr')
                    fn3 = os.path.join(out_img_dir, 'pred_curve_' + str(titr) + '.exr')
                    fn4 = os.path.join(out_img_dir, 'pred_face_' + str(titr) + '.exr')
                    fn5 = os.path.join(out_img_dir, 'gt_face_' + str(titr) + '.exr')
                    fn7 = os.path.join(out_img_dir, 'normal_' + str(titr) + '.exr')
                    fn8 = os.path.join(out_img_dir, 'depth_' + str(titr) + '.exr')
                    fn9 = os.path.join(out_img_dir, 'strokeMask_' + str(titr) + '.exr')

                    out_userStroke = t_userStroke[0, :, :, :]
                    out_userStroke.astype(np.float32)
                    cv2.imwrite(fn1, out_userStroke)

                    out_gt_c = t_gt_bc
                    cv2.imwrite(fn2, out_gt_c)

                    out_pred_c = t_pred_bc
                    cv2.imwrite(fn3, out_pred_c)

                    out_pred_fh = t_pred_fh
                    cv2.imwrite(fn4, out_pred_fh)

                    out_gt_fh = t_gt_fh
                    cv2.imwrite(fn5, out_gt_fh)

                    # out_normal = t_normal[0, :, :, :]
                    # out_normal = out_normal[:, :, [2, 1, 0]]
                    # out_normal.astype(np.float32)
                    # cv2.imwrite(fn7, out_normal)
                    #
                    # out_depth = t_depth[0, :, :, :]
                    # out_depth.astype(np.float32)
                    # cv2.imwrite(fn8, out_depth)

                    out_strokeMake = t_strokeMask
                    cv2.imwrite(fn9, out_strokeMake)

                titr += 1
                if titr % 100 == 0:
                    print('Iteration: {}'.format(titr))

        except tf.errors.OutOfRangeError:
            avg_loss /= titr
            avg_fh_loss /= titr
            avg_c_loss /= titr
            avg_real_fh_loss /= titr
            avg_real_c_loss /= titr
            avg_fIOU /= titr
            avg_bAcc /= titr
            test_logger.info(
                'Finish test model, average total loss: {}, fh: {}, cv: {}, rfh: {}, '
                'rcv: {}, fIOU: {}, bAcc: {}'.format(avg_loss,
                                                     avg_fh_loss,
                                                     avg_c_loss,
                                                     avg_real_fh_loss,
                                                     avg_real_c_loss,
                                                     avg_fIOU,
                                                     avg_bAcc))
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
