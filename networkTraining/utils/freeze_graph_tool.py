# @ Microsoft Research Asia, Internet Graphics Group,
# @ Project SketchCNN
# =========================================================================
"""
This file is written to convert model checkpoint to freeze graph
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from Sketch2CADNet.scripts.loader import SketchReader
import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--output_dir', dest='output_dir', type=str, default='test', help='output folder')
parser.add_argument('--ckpt_dir', dest='ckpt_dir', type=str, default='ckpt', help='checkpoint folder')
parser.add_argument('--in_gn', dest='in_gn', type=str, default='in_graph_name.pbtxt', help='input graph name')
parser.add_argument('--out_gn', dest='out_gn', type=str, default='out_graph_name.pb', help='output graph name')
parser.add_argument('--out_nodes', dest='out_nodes', type=str, default='node1,node2,xxx', help='output node name')
parser.add_argument('--devices', dest='device', type=str, default='0', help='GPU device indices')

args = parser.parse_args()


def convert_model(args):

    input_checkpoint_path = tf.train.latest_checkpoint(args.ckpt_dir)
    input_graph_path = os.path.join(args.output_dir, args.in_gn)
    input_saver_def_path = ""
    input_binary = False

    output_node_names = args.out_nodes
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    output_graph_path = os.path.join(args.output_dir, args.out_gn)
    clear_devices = False

    freeze_graph.freeze_graph(input_graph_path, input_saver_def_path, input_binary, input_checkpoint_path,
                              output_node_names, restore_op_name, filename_tensor_name, output_graph_path,
                              clear_devices, initializer_nodes='', variable_names_blacklist='')


if __name__ == '__main__':
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    convert_model(args)
