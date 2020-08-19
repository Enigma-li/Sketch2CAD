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
from random import randint

# Hyper Parameters
hyper_params = {
    'maxIter': 160000,
    'batchSize': 16,
    'dbTrain': '',
    'dbEval': '',
    'outDir': '',
    'device': '0',
    'nb_gpus': 1,
    'rootFt': 32,
    'dispLossStep': 100,
    'exeValStep': 1500,
    'saveModelStep': 1500,
    'nbDispImg': 4,
    'nbThreads': 64,
    'ckpt': '',
    'cnt': False,
}

userStroke_input = tf.placeholder(tf.float32, [None, None, None, 1], name='user_input')
cnormal_input = tf.placeholder(tf.float32, [None, None, None, 3], name='normal_input')
cdepth_input = tf.placeholder(tf.float32, [None, None, None, 1], name='depth_input')
gtFH_input = tf.placeholder(tf.float32, [None, None, None, 1], name='gtFH_input')
gtBC_input = tf.placeholder(tf.float32, [None, None, None, 1], name='gtBaseCurve_input')


# TensorBoard
def collect_vis_img(logit_fh, logit_bc, user_stroke, cnormal, cdepth, face_heatmap, base_curve,
                    scope='collect_vis_image', is_training=True):
    with tf.name_scope(scope) as _:

        img_shape = tf.shape(logit_fh)
        N = img_shape[0]
        H = img_shape[1]
        W = img_shape[2]
        C = img_shape[3]
        full_one = tf.ones([N, H, W, C], tf.float32)
        stroke_mask = full_one - user_stroke
        logit_bc = logit_bc * stroke_mask

        if is_training:
            out_fh_fn = 'train_out_face'
            out_bc_fn = 'train_out_baseCurve'
            userStroke_fn = 'train_userSotrkes'
            normal_fn = 'train_context_normal'
            depth_fn = 'train_context_depth'
            face_map_fn = 'train_gt_faceMap'
            baseCurve_fn = 'train_gt_baseCurve'
        else:
            out_fh_fn = 'val_out_face'
            out_bc_fn = 'val_out_baseCurve'
            userStroke_fn = 'val_userSotrkes'
            normal_fn = 'val_context_normal'
            depth_fn = 'val_context_depth'
            face_map_fn = 'val_gt_faceMap'
            baseCurve_fn = 'val_gt_baseCurve'

        logit_fh_proto = tf.summary.image(out_fh_fn, logit_fh, hyper_params['nbDispImg'])
        logit_bc_proto = tf.summary.image(out_bc_fn, logit_bc, hyper_params['nbDispImg'])
        userStroke_proto = tf.summary.image(userStroke_fn, user_stroke, hyper_params['nbDispImg'])
        normal_proto = tf.summary.image(normal_fn, cnormal, hyper_params['nbDispImg'])
        depth_proto = tf.summary.image(depth_fn, cdepth, hyper_params['nbDispImg'])
        face_map_proto = tf.summary.image(face_map_fn, face_heatmap, hyper_params['nbDispImg'])
        baseCurve_proto = tf.summary.image(baseCurve_fn, base_curve, hyper_params['nbDispImg'])

    return [logit_fh_proto, logit_bc_proto, userStroke_proto, normal_proto, depth_proto, face_map_proto,
            baseCurve_proto]


# Loss term
def loss(logit_fh, logit_bc, fh, bc, user_stroke, scope='loss'):
    with tf.name_scope(scope) as _:
        # stitching face heat map (l2)
        gt_fh = slice_tensor(fh, logit_fh)
        fh_loss = tf.losses.mean_squared_error(gt_fh, logit_fh)
        real_fh_loss = tf.losses.absolute_difference(gt_fh, logit_fh)

        curve_mask = slice_tensor(bc, logit_bc)
        img_shape = tf.shape(logit_fh)
        N = img_shape[0]
        H = img_shape[1]
        W = img_shape[2]
        C = img_shape[3]
        full_one = tf.ones([N, H, W, C], tf.float32)
        full_zero = tf.zeros([N, H, W, C], tf.float32)
        stroke_mask = full_one - user_stroke
        diff_mask = stroke_mask - curve_mask

        fg_pred = logit_bc * curve_mask
        bg_pred = logit_bc * diff_mask
        curve_fg_sum = tf.reduce_sum(tf.pow(curve_mask - fg_pred, 2), axis=[1, 2, 3])  # offset curve loss sum
        curve_bg_sum = tf.reduce_sum(tf.pow(full_zero - bg_pred, 2), axis=[1, 2, 3])  # orthogonal curve loss sum
        nb_stroke_pixels = tf.reduce_sum(stroke_mask, axis=[1, 2, 3])  # number of stroke pixels
        curve_loss = (curve_fg_sum + curve_bg_sum) / nb_stroke_pixels  # mean loss
        bc_loss = tf.reduce_mean(curve_loss)

        real_curve_fg_sum = tf.reduce_sum(tf.abs(curve_mask - fg_pred), axis=[1, 2, 3])
        real_curve_bg_sum = tf.reduce_sum(tf.abs(full_zero - bg_pred), axis=[1, 2, 3])
        real_curve_loss = (real_curve_fg_sum + real_curve_bg_sum) / nb_stroke_pixels
        real_bc_loss = tf.reduce_mean(real_curve_loss)

        total_loss = fh_loss + bc_loss

        return total_loss, fh_loss, bc_loss, real_fh_loss, real_bc_loss


# multiple GPUs training
def average_gradient(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def average_losses(tower_losses_full):
    return_averaged_losses = []
    for per_loss_in_one_tower in tower_losses_full:
        losses = []
        for tower_loss in per_loss_in_one_tower:
            expand_loss = tf.expand_dims(tower_loss, 0)
            losses.append(expand_loss)

        average_loss = tf.concat(losses, axis=0)
        average_loss = tf.reduce_mean(average_loss, 0)

        return_averaged_losses.append(average_loss)

    return return_averaged_losses


# training process
def train_procedure(net, train_records):
    nb_gpus = hyper_params['nb_gpus']

    # Load data
    with tf.name_scope('train_input') as _:
        bSize = hyper_params['batchSize'] * nb_gpus
        nbThreads = hyper_params['nbThreads'] * nb_gpus
        reader = SketchReader(tfrecord_list=train_records, raw_size=[256, 256, 17], shuffle=True,
                              num_threads=nbThreads, batch_size=bSize)
        raw_input = reader.next_batch()

        user_strokes, _, context_normal, context_depth, face_heatm, base_curve, _, _, _, _, _, _, _, _, _ \
            = net.cook_raw_inputs(raw_input)

    # initialize optimizer
    opt = tf.train.AdamOptimizer()

    # split data
    with tf.name_scope('divide_data'):
        gpu_user_strokes = tf.split(userStroke_input, nb_gpus, axis=0)
        gpu_cnormal = tf.split(cnormal_input, nb_gpus, axis=0)
        gpu_cdepth = tf.split(cdepth_input, nb_gpus, axis=0)
        gpu_fh = tf.split(gtFH_input, nb_gpus, axis=0)
        gpu_bc = tf.split(gtBC_input, nb_gpus, axis=0)

    tower_grads = []
    tower_loss_collected = []
    tower_total_losses = []
    tower_fh_losses = []
    tower_bc_losses = []
    tower_abs_fh_losses = []
    tower_abs_bc_losses = []

    # TensorBoard images
    gpu0_logit_fh = None
    gpu0_logit_bc = None
    gpu0_userStroke = None
    gpu0_normal = None
    gpu0_depth = None
    gpu0_faceMap = None
    gpu0_baseCurve = None

    with tf.variable_scope(tf.get_variable_scope()):
        for gpu_id in range(nb_gpus):
            with tf.device('/gpu:%d' % gpu_id):
                with tf.name_scope('tower_%s' % gpu_id) as _:
                    # network forward
                    logit_fh, logit_bc, _ = net.load_bevel_reg_net(gpu_user_strokes[gpu_id],
                                                                   gpu_cnormal[gpu_id],
                                                                   gpu_cdepth[gpu_id],
                                                                   hyper_params['rootFt'],
                                                                   is_training=True)

                    # training loss
                    train_loss, train_fh_loss, train_bc_loss, train_real_fh_loss, \
                    train_real_bc_loss = loss(logit_fh,
                                              logit_bc,
                                              gpu_fh[gpu_id],
                                              gpu_bc[gpu_id],
                                              gpu_user_strokes[gpu_id],
                                              scope='train_loss')

                    # reuse variables
                    tf.get_variable_scope().reuse_variables()

                    # collect gradients and every loss
                    tower_grads.append(opt.compute_gradients(train_loss))
                    tower_total_losses.append(train_loss)
                    tower_fh_losses.append(train_fh_loss)
                    tower_bc_losses.append(train_bc_loss)
                    tower_abs_fh_losses.append(train_real_fh_loss)
                    tower_abs_bc_losses.append(train_real_bc_loss)

                    # TensorBoard: collect images from GPU 0
                    if gpu_id == 0:
                        gpu0_logit_fh = logit_fh
                        gpu0_logit_bc = logit_bc
                        gpu0_userStroke = gpu_user_strokes[gpu_id]
                        gpu0_normal = gpu_cnormal[gpu_id]
                        gpu0_depth = gpu_cdepth[gpu_id]
                        gpu0_faceMap = gpu_fh[gpu_id]
                        gpu0_baseCurve = gpu_bc[gpu_id]

        tower_loss_collected.append(tower_total_losses)
        tower_loss_collected.append(tower_fh_losses)
        tower_loss_collected.append(tower_bc_losses)
        tower_loss_collected.append(tower_abs_fh_losses)
        tower_loss_collected.append(tower_abs_bc_losses)

    # Solver
    with tf.name_scope('solve') as _:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            grads = average_gradient(tower_grads)
            averaged_losses = average_losses(tower_loss_collected)
            apply_gradient_op = opt.apply_gradients(grads)
            train_op = tf.group(apply_gradient_op)

    # TensorBoard: visualization
    train_diff_proto = tf.summary.scalar('Training_TotalLoss', averaged_losses[0])
    train_diff_fh_proto = tf.summary.scalar('Training_FhL2Loss', averaged_losses[1])
    train_diff_bc_proto = tf.summary.scalar('Training_BcL2Loss', averaged_losses[2])
    train_diff_real_fn_proto = tf.summary.scalar('Training_FhL1Loss', averaged_losses[3])
    train_diff_real_bc_proto = tf.summary.scalar('Training_BcL1Loss', averaged_losses[4])

    proto_list = collect_vis_img(gpu0_logit_fh,
                                 gpu0_logit_bc,
                                 gpu0_userStroke,
                                 gpu0_normal,
                                 gpu0_depth,
                                 gpu0_faceMap,
                                 gpu0_baseCurve,
                                 scope='collect_train_imgs')

    proto_list.append(train_diff_proto)
    proto_list.append(train_diff_fh_proto)
    proto_list.append(train_diff_bc_proto)
    proto_list.append(train_diff_real_fn_proto)
    proto_list.append(train_diff_real_bc_proto)

    merged_train = tf.summary.merge(proto_list)

    return merged_train, train_op, averaged_losses[0], \
           [user_strokes, context_normal, context_depth, face_heatm, base_curve]


# validation process
def validation_procedure(net, val_records):
    # Load data
    with tf.name_scope('eval_inputs') as _:
        reader = SketchReader(tfrecord_list=val_records, raw_size=[256, 256, 17], shuffle=False,
                              num_threads=hyper_params['nbThreads'], batch_size=hyper_params['batchSize'])
        raw_input = reader.next_batch()

        user_strokes, _, context_normal, context_depth, face_heatm, base_curve, _, _, _, _, _, _, _, _, _ \
            = net.cook_raw_inputs(raw_input)

    # Network forward
    logit_fh, logit_bc, _ = net.load_bevel_reg_net(userStroke_input,
                                                   cnormal_input,
                                                   cdepth_input,
                                                   hyper_params['rootFt'],
                                                   is_training=False,
                                                   reuse=True)

    # Loss
    val_loss, val_fh_loss, val_bc_loss, val_real_fh_loss, val_real_bc_loss = loss(logit_fh,
                                                                                  logit_bc,
                                                                                  gtFH_input,
                                                                                  gtBC_input,
                                                                                  userStroke_input,
                                                                                  scope='val_loss')

    # TensorBoard
    proto_list = collect_vis_img(logit_fh,
                                 logit_bc,
                                 userStroke_input,
                                 cnormal_input,
                                 cdepth_input,
                                 gtFH_input,
                                 gtBC_input,
                                 scope='collect_val_imgs',
                                 is_training=False)

    merged_val = tf.summary.merge(proto_list)

    return merged_val, val_loss, val_fh_loss, val_bc_loss, val_real_fh_loss, val_real_bc_loss, \
           [user_strokes, context_normal, context_depth, face_heatm, base_curve]


def train_net():
    # Set logging
    train_logger = logging.getLogger('main.training')
    train_logger.info('---Begin training: ---')

    # Load network
    net = SKETCH2CADNET()

    # Train
    train_data_records = [item for item in os.listdir(hyper_params['dbTrain']) if item.endswith('.tfrecords')]
    train_records = [os.path.join(hyper_params['dbTrain'], item) for item in train_data_records if
                     item.find('train') != -1]
    train_summary, train_step, train_loss, train_inputList = train_procedure(net, train_records)

    # Validation
    val_data_records = [item for item in os.listdir(hyper_params['dbEval']) if item.endswith('.tfrecords')]
    val_records = [os.path.join(hyper_params['dbEval'], item) for item in val_data_records if
                   item.find('eval') != -1]
    options_zlib = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    num_eval_samples = sum(1 for _ in tf.python_io.tf_record_iterator(val_records[0], options=options_zlib))
    num_eval_itr = num_eval_samples // hyper_params['batchSize']
    num_eval_itr += 1

    val_proto, val_loss, val_fh_loss, val_bc_loss, val_real_fh_loss, val_real_bc_loss, val_inputList \
        = validation_procedure(net,
                               val_records)

    valid_loss = tf.placeholder(tf.float32, name='mval_loss')
    valid_loss_proto = tf.summary.scalar('Validating_TotalLoss', valid_loss)
    valid_fh_loss = tf.placeholder(tf.float32, name='mval_fh_loss')
    valid_fh_loss_proto = tf.summary.scalar('Validating_FhL2Loss', valid_fh_loss)
    valid_bc_loss = tf.placeholder(tf.float32, name='mval_bc_loss')
    valid_bc_loss_proto = tf.summary.scalar('Validating_BcL2Loss', valid_bc_loss)
    valid_real_fh_loss = tf.placeholder(tf.float32, name='mval_real_fh_loss')
    valid_real_fh_loss_proto = tf.summary.scalar('Validating_FhL1Loss', valid_real_fh_loss)
    valid_real_bc_loss = tf.placeholder(tf.float32, name='mval_real_bc_loss')
    valid_real_bc_loss_proto = tf.summary.scalar('Validating_BcL1Loss', valid_real_bc_loss)
    valid_value_merge = tf.summary.merge([valid_loss_proto, valid_fh_loss_proto, valid_bc_loss_proto,
                                          valid_real_fh_loss_proto, valid_real_bc_loss_proto])

    # Saver
    tf_saver = tf.train.Saver(max_to_keep=200)

    # Session config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # config.log_device_placement = True

    with tf.Session(config=config) as sess:
        # TF summary
        train_writer = tf.summary.FileWriter(output_folder + '/train', sess.graph)

        # initialize
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # Continue training
        if hyper_params['cnt']:
            ckpt = tf.train.latest_checkpoint(hyper_params['ckpt'])
            if ckpt:
                tf_saver.restore(sess, ckpt)
                train_logger.info('restore from the checkpoint {}'.format(ckpt))

        # Start input enqueue threads
        train_logger.info('pre-load data into data buffer...')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for titr in range(hyper_params['maxIter']):

            #  Validation
            if titr % hyper_params['exeValStep'] == 0:
                idx = randint(0, num_eval_itr - 1)
                avg_loss = 0.0
                avg_fh_loss = 0.0
                avg_bc_loss = 0.0
                avg_real_fh_loss = 0.0
                avg_real_bc_loss = 0.0
                for eitr in range(num_eval_itr):

                    # get real input
                    val_real_input = sess.run(val_inputList)

                    if eitr == idx:
                        val_merge, cur_v_loss, cur_vfh_loss, cur_vbc_loss, cur_real_fh_loss, cur_real_bc_loss = sess.run(
                            [val_proto, val_loss, val_fh_loss, val_bc_loss, val_real_fh_loss, val_real_bc_loss],
                            feed_dict={'user_input:0': val_real_input[0],
                                       'normal_input:0': val_real_input[1],
                                       'depth_input:0': val_real_input[2],
                                       'gtFH_input:0': val_real_input[3],
                                       'gtBaseCurve_input:0': val_real_input[4]})
                        train_writer.add_summary(val_merge, titr)
                    else:
                        cur_v_loss, cur_vfh_loss, cur_vbc_loss, cur_real_fh_loss, cur_real_bc_loss = sess.run(
                            [val_loss, val_fh_loss, val_bc_loss, val_real_fh_loss, val_real_bc_loss],
                            feed_dict={'user_input:0': val_real_input[0],
                                       'normal_input:0': val_real_input[1],
                                       'depth_input:0': val_real_input[2],
                                       'gtFH_input:0': val_real_input[3],
                                       'gtBaseCurve_input:0': val_real_input[4]})

                    avg_loss += cur_v_loss
                    avg_fh_loss += cur_vfh_loss
                    avg_bc_loss += cur_vbc_loss
                    avg_real_fh_loss += cur_real_fh_loss
                    avg_real_bc_loss += cur_real_bc_loss

                avg_loss /= num_eval_itr
                avg_fh_loss /= num_eval_itr
                avg_bc_loss /= num_eval_itr
                avg_real_fh_loss /= num_eval_itr
                avg_real_bc_loss /= num_eval_itr

                valid_summary = sess.run(valid_value_merge,
                                         feed_dict={'mval_loss:0': avg_loss,
                                                    'mval_fh_loss:0': avg_fh_loss,
                                                    'mval_bc_loss:0': avg_bc_loss,
                                                    'mval_real_fh_loss:0': avg_real_fh_loss,
                                                    'mval_real_bc_loss:0': avg_real_bc_loss
                                                    })

                train_writer.add_summary(valid_summary, titr)
                train_logger.info('Validation loss at step {} is: {}'.format(titr, avg_loss))

            # Save model
            if titr % hyper_params['saveModelStep'] == 0:
                tf_saver.save(sess, output_folder + '/savedModel/my_model{:d}.ckpt'.format(titr))
                train_logger.info('Save model at step: {:d}'.format(titr))

            # Training
            # get real input
            train_real_input = sess.run(train_inputList)

            t_summary, _, t_loss = sess.run([train_summary, train_step, train_loss],
                                            feed_dict={'user_input:0': train_real_input[0],
                                                       'normal_input:0': train_real_input[1],
                                                       'depth_input:0': train_real_input[2],
                                                       'gtFH_input:0': train_real_input[3],
                                                       'gtBaseCurve_input:0': train_real_input[4]})

            # Display
            if titr % hyper_params['dispLossStep'] == 0:
                train_writer.add_summary(t_summary, titr)
                train_logger.info('Training loss at step {} is: {}'.format(titr, t_loss))

        # Finish training
        coord.request_stop()
        coord.join(threads)

        # Release resource
        train_writer.close()


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dbTrain', required=True, help='training dataset directory', type=str)
    parser.add_argument('--dbEval', required=True, help='evaluation dataset directory', type=str)
    parser.add_argument('--outDir', required=True, help='output directory', type=str)
    parser.add_argument('--nb_gpus', help='GPU number', type=int, default=1)
    parser.add_argument('--devices', help='GPU device indices', type=str, default='0')
    parser.add_argument('--ckpt', help='checkpoint path', type=str, default='')
    parser.add_argument('--cnt', help='continue training flag', type=bool, default=False)
    parser.add_argument('--rootFt', help='root feature size', type=int, default=32)

    args = parser.parse_args()
    hyper_params['dbTrain'] = args.dbTrain
    hyper_params['dbEval'] = args.dbEval
    hyper_params['outDir'] = args.outDir
    hyper_params['nb_gpus'] = args.nb_gpus
    hyper_params['device'] = args.devices
    hyper_params['ckpt'] = args.ckpt
    hyper_params['cnt'] = args.cnt
    hyper_params['rootFt'] = args.rootFt

    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = hyper_params['device']

    # Set output folder
    timeSufix = time.strftime(r'%Y%m%d_%H%M%S')
    output_folder = hyper_params['outDir'] + '_{}'.format(timeSufix)
    make_dir(output_folder)

    # Set logger
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_folder, 'log.txt'))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Training preparation
    logger.info('---Training preparation: ---')

    # Dump parameters
    dump_params(output_folder, hyper_params)

    # Begin training
    train_net()
