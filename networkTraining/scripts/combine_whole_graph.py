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
    'ckpt_cls': '',
    'ckpt_bl': '',
    'ckpt_ex': '',
    'ckpt_add': '',
    'ckpt_swp': '',
    'rootFt': 32,
    'nbThreads': 1,
    'nb_cls': 4,
}

userStroke_input = tf.placeholder(tf.float32, [None, None, None, 1], name='user_input')
cnormal_input = tf.placeholder(tf.float32, [None, None, None, 3], name='normal_input')
cdepth_input = tf.placeholder(tf.float32, [None, None, None, 1], name='depth_input')


def test_procedure(net, test_records):
    # load data
    reader = SketchReader(tfrecord_list=test_records, raw_size=[256, 256, 17], shuffle=False,
                          num_threads=hyper_params['nbThreads'], batch_size=1, nb_epoch=1)
    raw_input = reader.next_batch()

    user_strokes, _, context_normal, context_depth, face_heatm, base_curve, _, off_curve, _, _, _, \
    _, opt_label, _, line_reg = net.cook_raw_inputs(raw_input)

    # cls network forward
    _, cls_var = net.load_cls_opt_tiny_net(user_strokes,
                                           context_normal,
                                           context_depth,
                                           num_classes=hyper_params['nb_cls'],
                                           is_training=False)

    # addSub network forward
    _, _, add_var = net.load_addSub_reg_net(userStroke_input,
                                            cnormal_input,
                                            cdepth_input,
                                            hyper_params['rootFt'],
                                            is_training=False)

    # extrusion network forward
    logit_fh_e, _, ext_var = net.load_extrusion_reg_net(userStroke_input,
                                                        cnormal_input,
                                                        cdepth_input,
                                                        hyper_params['rootFt'],
                                                        is_training=False)
    # bevel network forward
    _, _, bel_var = net.load_bevel_reg_net(userStroke_input,
                                           cnormal_input,
                                           cdepth_input,
                                           hyper_params['rootFt'],
                                           is_training=False)

    # sweep network forward
    swp_var = None
    _, _, swp_var = net.load_sweep_reg_net(userStroke_input,
                                           cnormal_input,
                                           cdepth_input,
                                           hyper_params['rootFt'],
                                           is_training=False)

    return logit_fh_e, cls_var, ext_var, add_var, bel_var, swp_var, \
           [user_strokes, context_normal, context_depth]


def write_whole_graph():
    # Set logging
    test_logger = logging.getLogger('main.testing')
    test_logger.info('---Begin testing: ---')

    # Load network
    net = SKETCH2CADNET()

    # Test
    data_records = [item for item in os.listdir(hyper_params['dbTest']) if item.endswith('.tfrecords')]
    test_records = [os.path.join(hyper_params['dbTest'], item) for item in data_records if item.find('test') != -1]

    pred_fh, cls_var, exr_var, add_var, bel_var, swp_var, test_inputList = test_procedure(net, test_records)

    # Saver
    tf_saver = tf.train.Saver()
    tf_cls_saver = tf.train.Saver(var_list=cls_var)
    tf_ex_saver = tf.train.Saver(var_list=exr_var)
    tf_add_saver = tf.train.Saver(var_list=add_var)
    tf_bl_saver = tf.train.Saver(var_list=bel_var)
    tf_swp_saver = tf.train.Saver(var_list=swp_var)

    # configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    # config.log_device_placement = True

    with tf.Session(config=config) as sess:
        # initialize
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # Restore model
        cls_ckpt = tf.train.latest_checkpoint(hyper_params['ckpt_cls'])
        if cls_ckpt:
            tf_cls_saver.restore(sess, cls_ckpt)
            test_logger.info('restore from the checkpoint {}'.format(cls_ckpt))

        add_ckpt = tf.train.latest_checkpoint(hyper_params['ckpt_add'])
        if add_ckpt:
            tf_add_saver.restore(sess, add_ckpt)
            test_logger.info('restore from the checkpoint {}'.format(add_ckpt))

        bl_ckpt = tf.train.latest_checkpoint(hyper_params['ckpt_bl'])
        if bl_ckpt:
            tf_bl_saver.restore(sess, bl_ckpt)
            test_logger.info('restore from the checkpoint {}'.format(bl_ckpt))

        ex_ckpt = tf.train.latest_checkpoint(hyper_params['ckpt_ex'])
        if ex_ckpt:
            tf_ex_saver.restore(sess, ex_ckpt)
            test_logger.info('restore from the checkpoint {}'.format(ex_ckpt))

        swp_ckpt = tf.train.latest_checkpoint(hyper_params['ckpt_swp'])
        if swp_ckpt:
            tf_swp_saver.restore(sess, swp_ckpt)
            test_logger.info('restore from the checkpoint {}'.format(swp_ckpt))

        # writeGraph:
        tf.train.write_graph(sess.graph_def, out_net_dir,
                             "S2CNet.pbtxt", as_text=True)
        test_logger.info('save graph tp pbtxt, done')

        # Start input enqueue threads
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            titr = 0
            while not coord.should_stop():
                # get real input
                test_realInput = sess.run(test_inputList)

                t_fh = sess.run([pred_fh], feed_dict={'user_input:0': test_realInput[0],
                                                      'normal_input:0': test_realInput[1],
                                                      'depth_input:0': test_realInput[2]})

                if titr % 100 == 0:
                    tf_saver.save(sess, out_folder + '/savedModel/my_model{:d}.ckpt'.format(titr))
                    test_logger.info('Save model at step: {:d}'.format(titr))

                # Write image out
                if titr < 50:
                    fn1 = os.path.join(out_img_dir, 'faceMap' + str(titr) + '.exr')
                    out_userStroke = np.array(t_fh)[0, 0, :, :, 0]
                    out_userStroke.astype(np.float32)
                    cv2.imwrite(fn1, out_userStroke)

                titr += 1
                if titr % 100 == 0:
                    print('Iteration: {}'.format(titr))

        except tf.errors.OutOfRangeError:
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
    parser.add_argument('--ckpt_cls', help='checkpoint folder', type=str)
    parser.add_argument('--ckpt_bl', help='bevel checkpoint folder', type=str)
    parser.add_argument('--ckpt_ex', help='extrude checkpoint folder', type=str)
    parser.add_argument('--ckpt_add', help='addSub checkpoint folder', type=str)
    parser.add_argument('--ckpt_swp', help='sweep checkpoint folder', type=str)
    parser.add_argument('--nbThr', help='n=Number of loading thread', type=int, default=1)
    parser.add_argument('--nb_cls', help='number of classes', type=int, default=4)

    args = parser.parse_args()
    hyper_params['dbTest'] = args.dbTest
    hyper_params['outDir'] = args.outDir
    hyper_params['device'] = args.devices
    hyper_params['ckpt_cls'] = args.ckpt_cls
    hyper_params['ckpt_bl'] = args.ckpt_bl
    hyper_params['ckpt_ex'] = args.ckpt_ex
    hyper_params['ckpt_add'] = args.ckpt_add
    hyper_params['ckpt_swp'] = args.ckpt_swp
    hyper_params['nb_cls'] = args.nb_cls
    hyper_params['nbThreads'] = args.nbThr

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
    write_whole_graph()
