#
# Project Sketch2CAD
#
#   Author: Changjian Li (chjili2011@gmail.com),
#   Copyright (c) 2020. All Rights Reserved.
#
# ==============================================================================
"""Network structure design for SketchCNN.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from Sketch2CADNet.utils.util_funcs import cropconcat_layer, concate_layers
import tensorflow.contrib.slim as slim
import logging

# network logger initialization
net_logger = logging.getLogger('main.network')


class SKETCH2CADNET(object):
    """
        CNN networks for operators classification, and parameters regression.
    """

    @staticmethod
    def cook_raw_inputs(raw_input):
        """
        Args:
            :param raw_input: raw input after custom decoder.

        Returns:
            :return: divided tensors.
        """

        with tf.name_scope("cook_raw_input") as _:
            input_data, label_data = raw_input

            user_stroke = tf.slice(input_data, [0, 0, 0, 0], [-1, -1, -1, 1])
            scaffold_line = tf.slice(input_data, [0, 0, 0, 1], [-1, -1, -1, 1])
            context_normal = tf.slice(input_data, [0, 0, 0, 2], [-1, -1, -1, 3])
            context_depth = tf.slice(input_data, [0, 0, 0, 5], [-1, -1, -1, 1])

            face_heatmp = tf.slice(label_data, [0, 0, 0, 0], [-1, -1, -1, 1])
            base_curve = tf.slice(label_data, [0, 0, 0, 1], [-1, -1, -1, 1])
            profile_curve = tf.slice(label_data, [0, 0, 0, 2], [-1, -1, -1, 1])
            offset_curve = tf.slice(label_data, [0, 0, 0, 3], [-1, -1, -1, 1])
            shape_mask = tf.slice(label_data, [0, 0, 0, 4], [-1, -1, -1, 1])
            off_dis = tf.slice(label_data, [0, 0, 0, 5], [-1, -1, -1, 1])
            off_dir = tf.slice(label_data, [0, 0, 0, 6], [-1, -1, -1, 3])
            off_sign = tf.slice(label_data, [0, 0, 0, 9], [-1, -1, -1, 1])
            opt_type = tf.slice(label_data, [0, 0, 0, 10], [-1, -1, -1, 1])
            opt_label_float = tf.slice(opt_type, [0, 0, 0, 0], [-1, 1, 1, 1])
            opt_label_float = tf.reshape(opt_label_float, [-1])
            opt_label = tf.cast(opt_label_float, dtype=tf.int32)

            line_label = tf.slice(label_data, [0, 0, 0, 11], [-1, -1, -1, 3])
            line_reg_label = tf.slice(label_data, [0, 0, 0, 14], [-1, -1, -1, 1])

            return user_stroke, scaffold_line, context_normal, context_depth, face_heatmp, base_curve, profile_curve, \
                   offset_curve, shape_mask, off_dis, off_dir, off_sign, opt_label, line_label, line_reg_label

    # Bevel operation regression - regression
    @staticmethod
    def load_bevel_reg_net(user_stroke, context_normal, context_depth, root_feature=32,
                           is_training=True, padding='SAME', reuse=None, d_rate=1, l2_reg=0.005):
        """
        Bevel operator parameters regression: stitching face and base curve

        Args:
            :param user_stroke: user strokes
            :param context_normal: current context normal.
            :param context_depth: current context depth.
            :param root_feature: root feature size.
            :param is_training: if True, use calculated BN statistics (Training); otherwise,
                                use the store value (Validating and Testing).
            :param padding: SAME/VALID, if SAME, output same size with input; otherwise, real size after pooling layer.
            :param reuse: if True, reuse current weight (Validating and Testing); otherwise, update (Training).
            :param d_rate: dilation rate, default 1.0.
            :param l2_reg: weight decay factor.

        Returns:
            :return: regressed stitching face heat map, base curve map
        """
        with tf.variable_scope('BevelOptNet', reuse=reuse) as b_vs:
            with slim.arg_scope([slim.conv2d], kernel_size=[3, 3], rate=d_rate,
                                padding=padding, activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training, 'decay': 0.95, 'fused': True},
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(l2_reg)
                                ):
                # encoder
                # input variable name
                input_userStroke = tf.identity(user_stroke, name='cls_stroke_input')
                input_normal = tf.identity(context_normal, name='cls_normal_input')
                input_depth = tf.identity(context_depth, name='cls_depth_input')

                reg_bevel_input = tf.concat([input_userStroke, input_normal, input_depth],
                                            axis=3, name='concat_bevel_input')

                b_input = tf.identity(reg_bevel_input, name='bevel_reg_input')
                conv1 = slim.conv2d(b_input, root_feature, scope='bevel_conv1')
                conv2 = slim.conv2d(conv1, root_feature, scope='bevel_conv2')
                pool1 = slim.max_pool2d(conv2, [2, 2], scope='bevel_pool1')
                conv3 = slim.conv2d(pool1, root_feature * 2, scope='bevel_conv3')
                conv4 = slim.conv2d(conv3, root_feature * 2, scope='bevel_conv4')
                pool2 = slim.max_pool2d(conv4, [2, 2], scope='bevel_pool2')
                conv5 = slim.conv2d(pool2, root_feature * 4, scope='bevel_onv5')
                conv6 = slim.conv2d(conv5, root_feature * 4, scope='bevel_conv6')
                pool3 = slim.max_pool2d(conv6, [2, 2], scope='bevel_pool3')
                conv7 = slim.conv2d(pool3, root_feature * 8, scope='bevel_conv7')
                conv8 = slim.conv2d(conv7, root_feature * 8, scope='bevel_conv8')
                pool4 = slim.max_pool2d(conv8, [2, 2], scope='bevel_pool4')
                conv9 = slim.conv2d(pool4, root_feature * 16, scope='bevel_conv9')
                conv10 = slim.conv2d(conv9, root_feature * 16, scope='bevel_conv10')

                # decoder
                with slim.arg_scope([slim.conv2d_transpose], kernel_size=[2, 2], stride=2,
                                    padding=padding, activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    weights_regularizer=slim.l2_regularizer(l2_reg)):
                    fm_deconv1 = slim.conv2d_transpose(conv10, root_feature * 8, scope='fm_deconv1_1')
                    bc_deconv1 = slim.conv2d_transpose(conv10, root_feature * 8, scope='bc_deconv1_1')

                    fm_concat1 = cropconcat_layer(conv8, fm_deconv1, 3, name='fm_concat1')
                    bc_concat1 = cropconcat_layer(conv8, bc_deconv1, 3, name='bc_concat1')

                    fm_deconv1_2 = slim.conv2d(fm_concat1, root_feature * 8, scope='fm_deconv1_2')
                    bc_deconv1_2 = slim.conv2d(bc_concat1, root_feature * 8, scope='bc_deconv1_2')

                    fm_deconv1_3 = slim.conv2d(fm_deconv1_2, root_feature * 8, scope='fm_deconv1_3')
                    bc_deconv1_3 = slim.conv2d(bc_deconv1_2, root_feature * 8, scope='bc_deconv1_3')

                    fm_deconv2 = slim.conv2d_transpose(fm_deconv1_3, root_feature * 4, scope='fm_deconv2_1')
                    bc_deconv2 = slim.conv2d_transpose(bc_deconv1_3, root_feature * 4, scope='bc_deconv2_1')

                    fm_concat2 = cropconcat_layer(conv6, fm_deconv2, 3, name='fm_concat2')
                    bc_concat2 = cropconcat_layer(conv6, bc_deconv2, 3, name='bc_concat2')

                    fm_deconv2_2 = slim.conv2d(fm_concat2, root_feature * 4, scope='fm_deonv2_2')
                    bc_deconv2_2 = slim.conv2d(bc_concat2, root_feature * 4, scope='bc_deonv2_2')

                    fm_deconv2_3 = slim.conv2d(fm_deconv2_2, root_feature * 4, scope='fm_deconv2_3')
                    bc_deconv2_3 = slim.conv2d(bc_deconv2_2, root_feature * 4, scope='bc_deconv2_3')

                    fm_deconv3 = slim.conv2d_transpose(fm_deconv2_3, root_feature * 2, scope='fm_deconv3_1')
                    bc_deconv3 = slim.conv2d_transpose(bc_deconv2_3, root_feature * 2, scope='bc_deconv3_1')

                    fm_concat3 = cropconcat_layer(conv4, fm_deconv3, 3, name='fm_concat3')
                    bc_concat3 = cropconcat_layer(conv4, bc_deconv3, 3, name='bc_concat3')

                    fm_deconv3_2 = slim.conv2d(fm_concat3, root_feature * 2, scope='fm_deconv3_2')
                    bc_deconv3_2 = slim.conv2d(bc_concat3, root_feature * 2, scope='bc_deconv3_2')

                    fm_deconv3_3 = slim.conv2d(fm_deconv3_2, root_feature * 2, scope='fm_deconv3_3')
                    bc_deconv3_3 = slim.conv2d(bc_deconv3_2, root_feature * 2, scope='bc_deconv3_3')

                    fm_deconv4 = slim.conv2d_transpose(fm_deconv3_3, root_feature, scope='fm_deconv4_1')
                    bc_deconv4 = slim.conv2d_transpose(bc_deconv3_3, root_feature, scope='bc_deconv4_1')

                    fm_concat4 = cropconcat_layer(conv2, fm_deconv4, 3, name='fm_concat4')
                    bc_concat4 = cropconcat_layer(conv2, bc_deconv4, 3, name='bc_concat4')

                    fm_deconv4_2 = slim.conv2d(fm_concat4, root_feature, scope='fm_deconv4_2')
                    bc_deconv4_2 = slim.conv2d(bc_concat4, root_feature, scope='bc_deconv4_2')

                    fm_deconv4_3 = slim.conv2d(fm_deconv4_2, root_feature, scope='fm_deconv4_3')
                    bc_deconv4_3 = slim.conv2d(bc_deconv4_2, root_feature, scope='bc_deconv4_3')

                    res_fm = slim.conv2d(fm_deconv4_3, 1, kernel_size=[1, 1], activation_fn=tf.nn.sigmoid,
                                         scope='fm_reg_output')  # stitching face heat map
                    res_bc = slim.conv2d(bc_deconv4_3, 1, kernel_size=[1, 1], activation_fn=tf.nn.sigmoid,
                                         scope='bc_reg_output')  # base curve map

                    logit_fm = tf.identity(res_fm, name='output_fm')
                    logit_bc = tf.identity(res_bc, name='output_bc')

        bevel_reg_net_variables = tf.contrib.framework.get_variables(b_vs)

        return logit_fm, logit_bc, bevel_reg_net_variables

    # Extrusion operation regression - curve regression
    @staticmethod
    def load_extrusion_reg_net(user_stroke, context_normal, context_depth, root_feature=32,
                               is_training=True, padding='SAME', reuse=None, d_rate=1, l2_reg=0.0005):
        """
        Extrusion operator parameters regression: stitching face and extrude magnitude.

        Args:
            :param user_stroke: user strokes
            :param context_normal: current context normal.
            :param context_depth: current context depth.
            :param root_feature: root feature size.
            :param is_training: if True, use calculated BN statistics (Training); otherwise,
                                use the store value (Validating and Testing).
            :param padding: SAME/VALID, if SAME, output same size with input; otherwise, real size after pooling layer.
            :param reuse: if True, reuse current weight (Validating and Testing); otherwise, update (Training).
            :param d_rate: dilation rate, default 1.0.
            :param l2_reg: weight decay factor.

        Returns:
            :return: regressed stitching face map, extrusion magnitude map.
        """
        with tf.variable_scope('ExtruOptNet', reuse=reuse) as e_vs:
            with slim.arg_scope([slim.conv2d], kernel_size=[3, 3], rate=d_rate,
                                padding=padding, activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training, 'decay': 0.95, 'fused': True},
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(l2_reg)
                                ):
                # encoder
                # input variable name
                input_userStroke = tf.identity(user_stroke, name='cls_stroke_input')
                input_normal = tf.identity(context_normal, name='cls_normal_input')
                input_depth = tf.identity(context_depth, name='cls_depth_input')

                reg_extrusion_input = tf.concat([input_userStroke, input_normal, input_depth],
                                                axis=3, name='concat_extrusion_input')

                ext_input = tf.identity(reg_extrusion_input, name='ext_reg_input')
                conv1 = slim.conv2d(ext_input, root_feature, scope='ext_conv1')
                conv2 = slim.conv2d(conv1, root_feature, scope='ext_conv2')
                pool1 = slim.max_pool2d(conv2, [2, 2], scope='ext_pool1')
                conv3 = slim.conv2d(pool1, root_feature * 2, scope='ext_conv3')
                conv4 = slim.conv2d(conv3, root_feature * 2, scope='ext_conv4')
                pool2 = slim.max_pool2d(conv4, [2, 2], scope='ext_pool2')
                conv5 = slim.conv2d(pool2, root_feature * 4, scope='ext_onv5')
                conv6 = slim.conv2d(conv5, root_feature * 4, scope='ext_conv6')
                pool3 = slim.max_pool2d(conv6, [2, 2], scope='ext_pool3')
                conv7 = slim.conv2d(pool3, root_feature * 8, scope='ext_conv7')
                conv8 = slim.conv2d(conv7, root_feature * 8, scope='ext_conv8')
                pool4 = slim.max_pool2d(conv8, [2, 2], scope='ext_pool4')
                conv9 = slim.conv2d(pool4, root_feature * 16, scope='ext_conv9')
                conv10 = slim.conv2d(conv9, root_feature * 16, scope='ext_conv10')

                # decoder
                with slim.arg_scope([slim.conv2d_transpose], kernel_size=[2, 2], stride=2,
                                    padding=padding, activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    weights_regularizer=slim.l2_regularizer(l2_reg)):
                    fm_deconv1 = slim.conv2d_transpose(conv10, root_feature * 8, scope='fm_deconv1_1')
                    emg_deconv1 = slim.conv2d_transpose(conv10, root_feature * 8, scope='emg_deconv1_1')

                    fm_concat1 = cropconcat_layer(conv8, fm_deconv1, 3, name='fm_concat1')
                    emg_concat1 = cropconcat_layer(conv8, emg_deconv1, 3, name='emg_concat1')

                    fm_deconv1_2 = slim.conv2d(fm_concat1, root_feature * 8, scope='fm_deconv1_2')
                    emg_deconv1_2 = slim.conv2d(emg_concat1, root_feature * 8, scope='emg_deconv1_2')

                    fm_deconv1_3 = slim.conv2d(fm_deconv1_2, root_feature * 8, scope='fm_deconv1_3')
                    emg_deconv1_3 = slim.conv2d(emg_deconv1_2, root_feature * 8, scope='emg_deconv1_3')

                    fm_deconv2 = slim.conv2d_transpose(fm_deconv1_3, root_feature * 4, scope='fm_deconv2_1')
                    emg_deconv2 = slim.conv2d_transpose(emg_deconv1_3, root_feature * 4, scope='emg_deconv2_1')

                    fm_concat2 = cropconcat_layer(conv6, fm_deconv2, 3, name='fm_concat2')
                    emg_concat2 = cropconcat_layer(conv6, emg_deconv2, 3, name='emg_concat2')

                    fm_deconv2_2 = slim.conv2d(fm_concat2, root_feature * 4, scope='fm_deonv2_2')
                    emg_deconv2_2 = slim.conv2d(emg_concat2, root_feature * 4, scope='emg_deonv2_2')

                    fm_deconv2_3 = slim.conv2d(fm_deconv2_2, root_feature * 4, scope='fm_deconv2_3')
                    emg_deconv2_3 = slim.conv2d(emg_deconv2_2, root_feature * 4, scope='emg_deconv2_3')

                    fm_deconv3 = slim.conv2d_transpose(fm_deconv2_3, root_feature * 2, scope='fm_deconv3_1')
                    emg_deconv3 = slim.conv2d_transpose(emg_deconv2_3, root_feature * 2, scope='emg_deconv3_1')

                    fm_concat3 = cropconcat_layer(conv4, fm_deconv3, 3, name='fm_concat3')
                    emg_concat3 = cropconcat_layer(conv4, emg_deconv3, 3, name='emg_concat3')

                    fm_deconv3_2 = slim.conv2d(fm_concat3, root_feature * 2, scope='fm_deconv3_2')
                    emg_deconv3_2 = slim.conv2d(emg_concat3, root_feature * 2, scope='emg_deconv3_2')

                    fm_deconv3_3 = slim.conv2d(fm_deconv3_2, root_feature * 2, scope='fm_deconv3_3')
                    emg_deconv3_3 = slim.conv2d(emg_deconv3_2, root_feature * 2, scope='emg_deconv3_3')

                    fm_deconv4 = slim.conv2d_transpose(fm_deconv3_3, root_feature, scope='fm_deconv4_1')
                    emg_deconv4 = slim.conv2d_transpose(emg_deconv3_3, root_feature, scope='emg_deconv4_1')

                    fm_concat4 = cropconcat_layer(conv2, fm_deconv4, 3, name='fm_concat4')
                    emg_concat4 = cropconcat_layer(conv2, emg_deconv4, 3, name='emg_concat4')

                    fm_deconv4_2 = slim.conv2d(fm_concat4, root_feature, scope='fm_deconv4_2')
                    emg_deconv4_2 = slim.conv2d(emg_concat4, root_feature, scope='emg_deconv4_2')

                    fm_deconv4_3 = slim.conv2d(fm_deconv4_2, root_feature, scope='fm_deconv4_3')
                    emg_deconv4_3 = slim.conv2d(emg_deconv4_2, root_feature, scope='emg_deconv4_3')

                    res_fm = slim.conv2d(fm_deconv4_3, 1, kernel_size=[1, 1], activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None, scope='fm_reg_output')  # stitching face heat map
                    res_offC = slim.conv2d(emg_deconv4_3, 1, kernel_size=[1, 1], activation_fn=tf.nn.sigmoid,
                                           normalizer_fn=None, scope='offC_reg_output')  # offset curve map

                    logit_fm = tf.identity(res_fm, name='output_fm')
                    logit_offC = tf.identity(res_offC, name='output_offC')

        extrusion_reg_net_variables = tf.contrib.framework.get_variables(e_vs)

        return logit_fm, logit_offC, extrusion_reg_net_variables

    # Operator classification2: smaller network
    @staticmethod
    def load_cls_opt_tiny_net(user_stroke, context_normal, context_depth,
                              num_classes=1000, is_training=True, dropout_keep_prob=0.5,
                              spatial_squeeze=True, reuse=None, fc_conv_padding='VALID'):
        """
        Operation classification - extrusion, bevel, add/sub, sweep

        Args:
            :param user_stroke: user strokes
            :param context_normal: current context normal.
            :param context_depth: current context depth.
            :param num_classes: umber of predicted classes.
            :param is_training: whether or not the model is being trained.
            :param dropout_keep_prob: the probability that activations are kept in the dropout layers during training.
            :param spatial_squeeze: whether or not should squeeze the spatial dimensions of the outputs.
            :param reuse: whether or not the network and its variables should be reused.
            :param fc_conv_padding: the type of padding to use for the fully connected layer that is implemented as a convolutional layer.

        Returns:
            :return: the output of the logits layer with size [N, num_classes] after squeezing.
        """
        with tf.variable_scope('optCls_VGGTiny', reuse=reuse) as c_vs:
            with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training, 'decay': 0.95, 'fused': True},
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)
                                ):
                # input variable name
                input_userStroke = tf.identity(user_stroke, name='cls_stroke_input')
                input_normal = tf.identity(context_normal, name='cls_normal_input')
                input_depth = tf.identity(context_depth, name='cls_depth_input')

                cls_input = tf.concat([input_userStroke, input_normal, input_depth], axis=3,
                                      name='concat_cls_input')

                net = slim.conv2d(cls_input, 32, [3, 3], scope='conv1')
                net = slim.max_pool2d(net, [2, 2], scope='pool1')
                net = slim.conv2d(net, 64, [3, 3], scope='conv2')
                net = slim.max_pool2d(net, [2, 2], scope='pool2')
                net = slim.conv2d(net, 128, [3, 3], scope='conv3')
                net = slim.max_pool2d(net, [2, 2], scope='pool3')
                net = slim.conv2d(net, 256, [3, 3], scope='conv4')
                net = slim.max_pool2d(net, [2, 2], scope='pool4')
                net = slim.conv2d(net, 512, [3, 3], scope='conv5')
                net = slim.max_pool2d(net, [2, 2], scope='pool5')

                # Use conv2d instead of fully_connected layers.
                net = slim.conv2d(net, 256, [8, 8], padding=fc_conv_padding, scope='fc6')
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
                net = slim.conv2d(net, 128, [1, 1], scope='fc7')

                net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')

                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')

                output_prob = tf.identity(net, name='output_prob')

        cls_net_variables = tf.contrib.framework.get_variables(c_vs)

        return output_prob, cls_net_variables

    # Add/Subtract operation regression - curve regression (two branches)
    @staticmethod
    def load_addSub_reg_net(user_stroke, context_normal, context_depth, root_feature=32,
                            is_training=True, padding='SAME', reuse=None, d_rate=1, l2_reg=0.0005):
        """
        Extrusion operator parameters regression: stitching face and extrude magnitude.

        Args:
            :param user_stroke: user strokes
            :param context_normal: current context normal.
            :param context_depth: current context depth.
            :param root_feature: root feature size.
            :param is_training: if True, use calculated BN statistics (Training); otherwise,
                                use the store value (Validating and Testing).
            :param padding: SAME/VALID, if SAME, output same size with input; otherwise, real size after pooling layer.
            :param reuse: if True, reuse current weight (Validating and Testing); otherwise, update (Training).
            :param d_rate: dilation rate, default 1.0.
            :param l2_reg: weight decay factor.

        Returns:
            :return: regressed stitching face map, base curve, offset curve, offset sign classification.
        """
        with tf.variable_scope('AddSubOptNet', reuse=reuse) as e_vs:
            with slim.arg_scope([slim.conv2d], kernel_size=[3, 3], rate=d_rate,
                                padding=padding, activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training, 'decay': 0.95, 'fused': True},
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(l2_reg)
                                ):
                # encoder
                # input variable name
                input_userStroke = tf.identity(user_stroke, name='cls_stroke_input')
                input_normal = tf.identity(context_normal, name='cls_normal_input')
                input_depth = tf.identity(context_depth, name='cls_depth_input')

                reg_extrusion_input = tf.concat([input_userStroke, input_normal, input_depth],
                                                axis=3, name='concat_extrusion_input')

                ext_input = tf.identity(reg_extrusion_input, name='ext_reg_input')
                conv1 = slim.conv2d(ext_input, root_feature, scope='ext_conv1')
                conv2 = slim.conv2d(conv1, root_feature, scope='ext_conv2')
                pool1 = slim.max_pool2d(conv2, [2, 2], scope='ext_pool1')
                conv3 = slim.conv2d(pool1, root_feature * 2, scope='ext_conv3')
                conv4 = slim.conv2d(conv3, root_feature * 2, scope='ext_conv4')
                pool2 = slim.max_pool2d(conv4, [2, 2], scope='ext_pool2')
                conv5 = slim.conv2d(pool2, root_feature * 4, scope='ext_onv5')
                conv6 = slim.conv2d(conv5, root_feature * 4, scope='ext_conv6')
                pool3 = slim.max_pool2d(conv6, [2, 2], scope='ext_pool3')
                conv7 = slim.conv2d(pool3, root_feature * 8, scope='ext_conv7')
                conv8 = slim.conv2d(conv7, root_feature * 8, scope='ext_conv8')
                pool4 = slim.max_pool2d(conv8, [2, 2], scope='ext_pool4')
                conv9 = slim.conv2d(pool4, root_feature * 16, scope='ext_conv9')
                conv10 = slim.conv2d(conv9, root_feature * 16, scope='ext_conv10')

                # decoder
                with slim.arg_scope([slim.conv2d_transpose], kernel_size=[2, 2], stride=2,
                                    padding=padding, activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    weights_regularizer=slim.l2_regularizer(l2_reg)):
                    fm_deconv1 = slim.conv2d_transpose(conv10, root_feature * 8, scope='fm_deconv1_1')
                    bm_deconv1 = slim.conv2d_transpose(conv10, root_feature * 8, scope='bm_deconv1_1')

                    fm_concat1 = cropconcat_layer(conv8, fm_deconv1, 3, name='fm_concat1')
                    bm_concat1 = cropconcat_layer(conv8, bm_deconv1, 3, name='bm_concat1')

                    fm_deconv1_2 = slim.conv2d(fm_concat1, root_feature * 8, scope='fm_deconv1_2')
                    bm_deconv1_2 = slim.conv2d(bm_concat1, root_feature * 8, scope='bm_deconv1_2')

                    fm_deconv1_3 = slim.conv2d(fm_deconv1_2, root_feature * 8, scope='fm_deconv1_3')
                    bm_deconv1_3 = slim.conv2d(bm_deconv1_2, root_feature * 8, scope='bm_deconv1_3')

                    fm_deconv2 = slim.conv2d_transpose(fm_deconv1_3, root_feature * 4, scope='fm_deconv2_1')
                    bm_deconv2 = slim.conv2d_transpose(bm_deconv1_3, root_feature * 4, scope='bm_deconv2_1')

                    fm_concat2 = cropconcat_layer(conv6, fm_deconv2, 3, name='fm_concat2')
                    bm_concat2 = cropconcat_layer(conv6, bm_deconv2, 3, name='bm_concat2')

                    fm_deconv2_2 = slim.conv2d(fm_concat2, root_feature * 4, scope='fm_deonv2_2')
                    bm_deconv2_2 = slim.conv2d(bm_concat2, root_feature * 4, scope='bm_deonv2_2')

                    fm_deconv2_3 = slim.conv2d(fm_deconv2_2, root_feature * 4, scope='fm_deconv2_3')
                    bm_deconv2_3 = slim.conv2d(bm_deconv2_2, root_feature * 4, scope='bm_deconv2_3')

                    fm_deconv3 = slim.conv2d_transpose(fm_deconv2_3, root_feature * 2, scope='fm_deconv3_1')
                    bm_deconv3 = slim.conv2d_transpose(bm_deconv2_3, root_feature * 2, scope='bm_deconv3_1')

                    fm_concat3 = cropconcat_layer(conv4, fm_deconv3, 3, name='fm_concat3')
                    bm_concat3 = cropconcat_layer(conv4, bm_deconv3, 3, name='bm_concat3')

                    fm_deconv3_2 = slim.conv2d(fm_concat3, root_feature * 2, scope='fm_deconv3_2')
                    bm_deconv3_2 = slim.conv2d(bm_concat3, root_feature * 2, scope='bm_deconv3_2')

                    fm_deconv3_3 = slim.conv2d(fm_deconv3_2, root_feature * 2, scope='fm_deconv3_3')
                    bm_deconv3_3 = slim.conv2d(bm_deconv3_2, root_feature * 2, scope='bm_deconv3_3')

                    fm_deconv4 = slim.conv2d_transpose(fm_deconv3_3, root_feature, scope='fm_deconv4_1')
                    bm_deconv4 = slim.conv2d_transpose(bm_deconv3_3, root_feature, scope='bm_deconv4_1')

                    fm_concat4 = cropconcat_layer(conv2, fm_deconv4, 3, name='fm_concat4')
                    bm_concat4 = cropconcat_layer(conv2, bm_deconv4, 3, name='bm_concat4')

                    fm_deconv4_2 = slim.conv2d(fm_concat4, root_feature, scope='fm_deconv4_2')
                    bm_deconv4_2 = slim.conv2d(bm_concat4, root_feature, scope='bm_deconv4_2')

                    fm_deconv4_3 = slim.conv2d(fm_deconv4_2, root_feature, scope='fm_deconv4_3')
                    bm_deconv4_3 = slim.conv2d(bm_deconv4_2, root_feature, scope='bm_deconv4_3')

                    res_fm = slim.conv2d(fm_deconv4_3, 1, kernel_size=[1, 1], activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None, scope='fm_reg_output')  # stitching face heat map
                    res_c = slim.conv2d(bm_deconv4_3, 1, kernel_size=[1, 1], activation_fn=None,
                                        normalizer_fn=None, scope='curve_reg_output')  # base curve
                    logit_fm = tf.identity(res_fm, name='output_fm')
                    logit_curve = tf.identity(res_c, name='output_curve')

        addSub_reg_net_variables = tf.contrib.framework.get_variables(e_vs)

        return logit_fm, logit_curve, addSub_reg_net_variables

    # Sweep operation regression - curve regression (two branches)
    @staticmethod
    def load_sweep_reg_net(user_stroke, context_normal, context_depth, root_feature=32,
                           is_training=True, padding='SAME', reuse=None, d_rate=1, l2_reg=0.0005):
        """
        Sweep operator parameters regression: stitching face and extrude magnitude.

        Args:
            :param user_stroke: user strokes
            :param context_normal: current context normal.
            :param context_depth: current context depth.
            :param root_feature: root feature size.
            :param is_training: if True, use calculated BN statistics (Training); otherwise,
                                use the store value (Validating and Testing).
            :param padding: SAME/VALID, if SAME, output same size with input; otherwise, real size after pooling layer.
            :param reuse: if True, reuse current weight (Validating and Testing); otherwise, update (Training).
            :param d_rate: dilation rate, default 1.0.
            :param l2_reg: weight decay factor.

        Returns:
            :return: regressed stitching face map, base curve, offset curve, offset sign classification.
        """
        with tf.variable_scope('SweepOptNet', reuse=reuse) as e_vs:
            with slim.arg_scope([slim.conv2d], kernel_size=[3, 3], rate=d_rate,
                                padding=padding, activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': is_training, 'decay': 0.95, 'fused': True},
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                weights_regularizer=slim.l2_regularizer(l2_reg)
                                ):
                # encoder
                # input variable name
                input_userStroke = tf.identity(user_stroke, name='cls_stroke_input')
                input_normal = tf.identity(context_normal, name='cls_normal_input')
                input_depth = tf.identity(context_depth, name='cls_depth_input')

                reg_extrusion_input = tf.concat([input_userStroke, input_normal, input_depth],
                                                axis=3, name='concat_extrusion_input')

                ext_input = tf.identity(reg_extrusion_input, name='ext_reg_input')
                conv1 = slim.conv2d(ext_input, root_feature, scope='ext_conv1')
                conv2 = slim.conv2d(conv1, root_feature, scope='ext_conv2')
                pool1 = slim.max_pool2d(conv2, [2, 2], scope='ext_pool1')
                conv3 = slim.conv2d(pool1, root_feature * 2, scope='ext_conv3')
                conv4 = slim.conv2d(conv3, root_feature * 2, scope='ext_conv4')
                pool2 = slim.max_pool2d(conv4, [2, 2], scope='ext_pool2')
                conv5 = slim.conv2d(pool2, root_feature * 4, scope='ext_onv5')
                conv6 = slim.conv2d(conv5, root_feature * 4, scope='ext_conv6')
                pool3 = slim.max_pool2d(conv6, [2, 2], scope='ext_pool3')
                conv7 = slim.conv2d(pool3, root_feature * 8, scope='ext_conv7')
                conv8 = slim.conv2d(conv7, root_feature * 8, scope='ext_conv8')
                pool4 = slim.max_pool2d(conv8, [2, 2], scope='ext_pool4')
                conv9 = slim.conv2d(pool4, root_feature * 16, scope='ext_conv9')
                conv10 = slim.conv2d(conv9, root_feature * 16, scope='ext_conv10')

                # decoder
                with slim.arg_scope([slim.conv2d_transpose], kernel_size=[2, 2], stride=2,
                                    padding=padding, activation_fn=None,
                                    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                    weights_regularizer=slim.l2_regularizer(l2_reg)):
                    fm_deconv1 = slim.conv2d_transpose(conv10, root_feature * 8, scope='fm_deconv1_1')
                    bm_deconv1 = slim.conv2d_transpose(conv10, root_feature * 8, scope='bm_deconv1_1')

                    fm_concat1 = cropconcat_layer(conv8, fm_deconv1, 3, name='fm_concat1')
                    bm_concat1 = cropconcat_layer(conv8, bm_deconv1, 3, name='bm_concat1')

                    fm_deconv1_2 = slim.conv2d(fm_concat1, root_feature * 8, scope='fm_deconv1_2')
                    bm_deconv1_2 = slim.conv2d(bm_concat1, root_feature * 8, scope='bm_deconv1_2')

                    fm_deconv1_3 = slim.conv2d(fm_deconv1_2, root_feature * 8, scope='fm_deconv1_3')
                    bm_deconv1_3 = slim.conv2d(bm_deconv1_2, root_feature * 8, scope='bm_deconv1_3')

                    fm_deconv2 = slim.conv2d_transpose(fm_deconv1_3, root_feature * 4, scope='fm_deconv2_1')
                    bm_deconv2 = slim.conv2d_transpose(bm_deconv1_3, root_feature * 4, scope='bm_deconv2_1')

                    fm_concat2 = cropconcat_layer(conv6, fm_deconv2, 3, name='fm_concat2')
                    bm_concat2 = cropconcat_layer(conv6, bm_deconv2, 3, name='bm_concat2')

                    fm_deconv2_2 = slim.conv2d(fm_concat2, root_feature * 4, scope='fm_deonv2_2')
                    bm_deconv2_2 = slim.conv2d(bm_concat2, root_feature * 4, scope='bm_deonv2_2')

                    fm_deconv2_3 = slim.conv2d(fm_deconv2_2, root_feature * 4, scope='fm_deconv2_3')
                    bm_deconv2_3 = slim.conv2d(bm_deconv2_2, root_feature * 4, scope='bm_deconv2_3')

                    fm_deconv3 = slim.conv2d_transpose(fm_deconv2_3, root_feature * 2, scope='fm_deconv3_1')
                    bm_deconv3 = slim.conv2d_transpose(bm_deconv2_3, root_feature * 2, scope='bm_deconv3_1')

                    fm_concat3 = cropconcat_layer(conv4, fm_deconv3, 3, name='fm_concat3')
                    bm_concat3 = cropconcat_layer(conv4, bm_deconv3, 3, name='bm_concat3')

                    fm_deconv3_2 = slim.conv2d(fm_concat3, root_feature * 2, scope='fm_deconv3_2')
                    bm_deconv3_2 = slim.conv2d(bm_concat3, root_feature * 2, scope='bm_deconv3_2')

                    fm_deconv3_3 = slim.conv2d(fm_deconv3_2, root_feature * 2, scope='fm_deconv3_3')
                    bm_deconv3_3 = slim.conv2d(bm_deconv3_2, root_feature * 2, scope='bm_deconv3_3')

                    fm_deconv4 = slim.conv2d_transpose(fm_deconv3_3, root_feature, scope='fm_deconv4_1')
                    bm_deconv4 = slim.conv2d_transpose(bm_deconv3_3, root_feature, scope='bm_deconv4_1')

                    fm_concat4 = cropconcat_layer(conv2, fm_deconv4, 3, name='fm_concat4')
                    bm_concat4 = cropconcat_layer(conv2, bm_deconv4, 3, name='bm_concat4')

                    fm_deconv4_2 = slim.conv2d(fm_concat4, root_feature, scope='fm_deconv4_2')
                    bm_deconv4_2 = slim.conv2d(bm_concat4, root_feature, scope='bm_deconv4_2')

                    fm_deconv4_3 = slim.conv2d(fm_deconv4_2, root_feature, scope='fm_deconv4_3')
                    bm_deconv4_3 = slim.conv2d(bm_deconv4_2, root_feature, scope='bm_deconv4_3')

                    res_fm = slim.conv2d(fm_deconv4_3, 1, kernel_size=[1, 1], activation_fn=tf.nn.sigmoid,
                                         normalizer_fn=None, scope='fm_reg_output')  # stitching face heat map
                    res_c = slim.conv2d(bm_deconv4_3, 1, kernel_size=[1, 1], activation_fn=None,
                                        normalizer_fn=None, scope='curve_reg_output')  # base curve
                    logit_fm = tf.identity(res_fm, name='output_fm')
                    logit_curve = tf.identity(res_c, name='output_curve')

        sweep_reg_net_variables = tf.contrib.framework.get_variables(e_vs)

        return logit_fm, logit_curve, sweep_reg_net_variables
