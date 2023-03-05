#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from glob import glob
import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='GPU flag. 1 = gpu, 0 = cpu')
parser.add_argument('--test_dir', dest='test_dir', default='/content/test', help='test examples are saved here')
parser.add_argument('--test_data', dest='test_data', default='/content/SAR2SAR-GRD-test/test_data', help='data set for testing')
parser.add_argument('--stride_size', dest='stride_size', type=int, default=64, help='define stride when image dim exceeds 264')
args = parser.parse_args()

checkpoint_dir = '/content/SAR2SAR-GRD-test/checkpoint'
if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
from model import denoiser

def denoiser_test(denoiser):
    test_data = args.test_data
    print(
        "[*] Start testing on real data. Working directory: %s. Collecting data from %s and storing test results in %s" % (
        os.getcwd(), test_data, args.test_dir))
    test_files = glob((test_data+'/*.npy').format('float32'))
    denoiser.test(test_files, ckpt_dir=checkpoint_dir, save_dir=args.test_dir, dataset_dir=test_data, stride=args.stride_size)

if __name__ == '__main__':
    if args.use_gpu:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess)
            denoiser_test(model)
    else:
        with tf.Session() as sess:
            model = denoiser(sess)
            denoiser_test(model)
