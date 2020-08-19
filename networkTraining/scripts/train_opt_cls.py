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
    'maxIter': 500000,
    'batchSize': 16,
    'dbTrain': '',
    'dbEval': '',
    'outDir': '',
    'device': '0',
    'nb_gpus': 1,
    'dispLossStep': 100,
    'exeValStep': 4000,
    'saveModelStep': 4000,
    'nbDispImg': 4,
    'nbThreads': 64,
    'ckpt': '',
    'cnt': False,
    'nb_cls': 4,
}


# TensorBoard
def collect_vis_img(user_stroke, cnormal, cdepth, scope='collect_vis_image', is_training=True):
    with tf.name_scope(scope) as _:
        if is_training:
            userStroke_fn = 'train_userSotrkes'
            normal_fn = 'train_context_normal'
            depth_fn = 'train_context_depth'
        else:
            userStroke_fn = 'val_userSotrkes'
            normal_fn = 'val_context_normal'
            depth_fn = 'val_context_depth'

        userStroke_proto = tf.summary.image(userStroke_fn, user_stroke, hyper_params['nbDispImg'])
        normal_proto = tf.summary.image(normal_fn, cnormal, hyper_params['nbDispImg'])
        depth_proto = tf.summary.image(depth_fn, cdepth, hyper_params['nbDispImg'])

    return [userStroke_proto, normal_proto, depth_proto]


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

        user_strokes, _, context_normal, context_depth, _, _, _, _, _, _, _, _, opt_label, _, _ = \
            net.cook_raw_inputs(raw_input)

    # network forward
    logit_prob, _ = net.load_cls_opt_tiny_net(user_strokes,
                                              context_normal,
                                              context_depth,
                                              num_classes=hyper_params['nb_cls'],
                                              is_training=True)

    # loss
    train_loss = loss(logit_prob, opt_label, scope='train_loss')
    train_acc = cal_accuracy(logit_prob, opt_label, scope='train_acc')

    # TensorBoard: visualization
    train_diff_cls_proto = tf.summary.scalar('Training_ClsLoss', train_loss)
    train_diff_acc_proto = tf.summary.scalar('Training_AccLoss', train_acc)

    # Solver
    with tf.name_scope('solve') as _:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer().minimize(train_loss)

    proto_list = collect_vis_img(user_strokes,
                                 context_normal,
                                 context_depth,
                                 scope='collect_train_imgs')

    proto_list.append(train_diff_cls_proto)
    proto_list.append(train_diff_acc_proto)

    merged_train = tf.summary.merge(proto_list)

    return merged_train, train_step, train_loss, train_acc


# validation process
def validation_procedure(net, val_records):
    # Load data
    with tf.name_scope('eval_inputs') as _:
        reader = SketchReader(tfrecord_list=val_records, raw_size=[256, 256, 17], shuffle=False,
                              num_threads=hyper_params['nbThreads'], batch_size=hyper_params['batchSize'])
        raw_input = reader.next_batch()

        user_strokes, _, context_normal, context_depth, _, _, _, _, _, _, _, _, opt_label, _, _ = \
            net.cook_raw_inputs(raw_input)

    # network forward
    logit_prob, _ = net.load_cls_opt_tiny_net(user_strokes,
                                              context_normal,
                                              context_depth,
                                              num_classes=hyper_params['nb_cls'],
                                              is_training=False,
                                              reuse=True)

    # Loss
    val_loss = loss(logit_prob, opt_label, scope='train_loss')
    val_acc = cal_accuracy(logit_prob, opt_label, scope='train_acc')

    # TensorBoard
    proto_list = collect_vis_img(user_strokes,
                                 context_normal,
                                 context_depth,
                                 scope='collect_val_imgs')

    merged_val = tf.summary.merge(proto_list)

    return merged_val, val_loss, val_acc


# training
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
    train_summary, train_step, train_loss, train_acc = train_procedure(net, train_records)

    # Validation
    val_data_records = [item for item in os.listdir(hyper_params['dbEval']) if item.endswith('.tfrecords')]
    val_records = [os.path.join(hyper_params['dbEval'], item) for item in val_data_records if
                   item.find('eval') != -1]
    options_zlib = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.ZLIB)
    num_eval_samples = sum(1 for _ in tf.python_io.tf_record_iterator(val_records[0], options=options_zlib))
    num_eval_itr = num_eval_samples // hyper_params['batchSize']
    num_eval_itr += 1

    val_proto, val_loss, val_acc = validation_procedure(net, val_records)

    valid_loss = tf.placeholder(tf.float32, name='mval_cls_loss')
    valid_loss_proto = tf.summary.scalar('Validating_ClsLoss', valid_loss)
    valid_acc_loss = tf.placeholder(tf.float32, name='mval_acc_loss')
    valid_acc_loss_proto = tf.summary.scalar('Validating_AccLoss', valid_acc_loss)
    valid_value_merge = tf.summary.merge([valid_loss_proto, valid_acc_loss_proto])

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
                avg_cls_loss = 0.0
                avg_acc_loss = 0.0
                for eitr in range(num_eval_itr):
                    if eitr == idx:
                        val_merge, cur_vcls_loss, cur_vacc_loss = sess.run([val_proto, val_loss, val_acc])
                        train_writer.add_summary(val_merge, titr)
                    else:
                        cur_v_loss, cur_vcls_loss, cur_vacc_loss = sess.run([val_proto, val_loss, val_acc])

                    avg_cls_loss += cur_vcls_loss
                    avg_acc_loss += cur_vacc_loss

                avg_cls_loss /= num_eval_itr
                avg_acc_loss /= num_eval_itr

                valid_summary = sess.run(valid_value_merge,
                                         feed_dict={'mval_cls_loss:0': avg_cls_loss,
                                                    'mval_acc_loss:0': avg_acc_loss})

                train_writer.add_summary(valid_summary, titr)
                train_logger.info(
                    'Validation loss at step {} is: classification - {}, accuracy - {}'.format(titr, avg_cls_loss,
                                                                                               avg_acc_loss))

            # Save model
            if titr % hyper_params['saveModelStep'] == 0:
                tf_saver.save(sess, output_folder + '/savedModel/my_model{:d}.ckpt'.format(titr))
                train_logger.info('Save model at step: {:d}'.format(titr))

            # Training
            t_summary, _, t_loss, t_acc = sess.run([train_summary, train_step, train_loss, train_acc])

            # Display
            if titr % hyper_params['dispLossStep'] == 0:
                train_writer.add_summary(t_summary, titr)
                train_logger.info(
                    'Training loss at step {} is: classification - {}, accuracy - {}'.format(titr, t_loss, t_acc))

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
    parser.add_argument('--nb_cls', help='number of classes', type=int, default=4)

    args = parser.parse_args()
    hyper_params['dbTrain'] = args.dbTrain
    hyper_params['dbEval'] = args.dbEval
    hyper_params['outDir'] = args.outDir
    hyper_params['nb_gpus'] = args.nb_gpus
    hyper_params['device'] = args.devices
    hyper_params['ckpt'] = args.ckpt
    hyper_params['cnt'] = args.cnt
    hyper_params['nb_cls'] = args.nb_cls

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
