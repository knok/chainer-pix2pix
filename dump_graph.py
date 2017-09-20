#!/usr/bin/env python

from __future__ import print_function
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers
import numpy as np

from net import Discriminator
from net import Encoder
from net import Decoder
from webdnn.backend import generate_descriptor
from webdnn.frontend.chainer import ChainerConverter

def main():
    parser = argparse.ArgumentParser(description='chainer implementation of pix2pix')
    parser.add_argument('--batchsize', '-b', type=int, default=1,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--enc-npz', default='', required=True)
    parser.add_argument('--dec-npz', default='', required=True)
    args = parser.parse_args()

    # Set up a neural network to train
    enc = Encoder(in_ch=3)
    dec = Decoder(out_ch=3)

    chainer.serializers.load_npz(args.enc_npz, enc)
    chainer.serializers.load_npz(args.dec_npz, dec)

    # graph
    with chainer.using_config('train', False):
        input = np.zeros((1, 3, 256, 256), dtype=np.float32)
        x = chainer.Variable(input)
        z = enc(x)
        y = dec(z)
    graph = ChainerConverter().convert([x], [y])
    exec_info = generate_descriptor("webassembly", graph, constant_encoder_name="eightbit")
    exec_info.save(args.out)

if __name__ == '__main__':
    main()
