# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import configargparse

def parse_config(argv=None):
    arg_formatter = configargparse.ArgumentDefaultsHelpFormatter
    cfg_parser = configargparse.YAMLConfigFileParser
    description = 'SEU-VCL occlusion project'
    parser = configargparse.ArgParser(formatter_class=arg_formatter,
                                      config_file_parser_class=cfg_parser,
                                      description=description,
                                      prog='occlusion')

    parser.add_argument('--trainset',
                        default='',
                        type=str,
                        help='trainset.')
    parser.add_argument('--testset',
                        default='',
                        type=str,
                        help='testset.')
    parser.add_argument('--data_folder',
                        default='',
                        help='The directory that contains the data.')

    parser.add_argument('--use_mask',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='True for use mask.')

    parser.add_argument('-c', '--config',
                        required=True, is_config_file=True,
                        help='config file path')
    parser.add_argument('--note',
                        default='test',
                        type=str,
                        help='code note')
    parser.add_argument('--lr',
                        default=0.001,
                        type=float,
                        help='learning rate.')
    parser.add_argument('--batchsize',
                        default=10,
                        type=int,
                        help='batch size.')
    parser.add_argument('--epoch',
                        default=500,
                        type=int,
                        help='num epoch.')
    parser.add_argument('--worker',
                        default=0,
                        type=int,
                        help='workers for dataloader.')

    parser.add_argument('--mode',
                        default='',
                        type=str,
                        help='running mode.')

    parser.add_argument('--rgb_mode',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='True for add rgb encoder.')
                        
    parser.add_argument('--virtual_mask',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='True for using virtual mask.')                
    parser.add_argument('--pretrain',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='True for use pretrain parameters.')
    parser.add_argument('--pretrain_dir',
                        default='',
                        type=str,
                        help='The directory that contains the pretrain model.')
    parser.add_argument('--model_dir',
                        default='',
                        type=str,
                        help='(if test only) The directory that contains the model.')
    parser.add_argument('--model',
                        default='',
                        type=str,
                        help='the model used for this project.')
    parser.add_argument('--train_loss',
                        default='L1 partloss',
                        type=str,
                        help='training loss type.')
    parser.add_argument('--test_loss',
                        default='L1',
                        type=str,
                        help='testing loss type.')
    parser.add_argument('--viz',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='True for visualize input.')
    parser.add_argument('--task',
                        default='ed_train',
                        type=str,
                        help='ee_train: encoder-encoder only, else ed_train.')
    parser.add_argument('--fixmodel_dir',
                        default='',
                        type=str,
                        help='fixed model dir for ee_train.')
    parser.add_argument('--gpu_index',
                        default=0,
                        type=int,
                        help='gpu index.')
    parser.add_argument('--human_seg',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='human seg.')
    parser.add_argument('--model_seg',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='model seg.')
    parser.add_argument('--supervise',
                        default=False,
                        type=lambda x: x.lower() in ['true', '1'],
                        help='supervise mask training.')
    parser.add_argument('--dataset',
                        default='',
                        type=str,
                        help='dataset.')
    parser.add_argument('--fixmodel_type',
                        default='',
                        type=str,
                        help='fixmodel_type.')

    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict
