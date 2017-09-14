#!/usr/bin/env python

from __future__ import print_function
import argparse
import os

import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers
import numpy as np
from PIL import Image

from net import Discriminator
from net import Encoder
from net import Decoder

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
    parser.add_argument('input')
    args = parser.parse_args()

    # Set up a neural network to train
    enc = Encoder(in_ch=3)
    dec = Decoder(out_ch=3)

    chainer.serializers.load_npz(args.enc_npz, enc)
    chainer.serializers.load_npz(args.dec_npz, dec)

    # load image and predict

    img = Image.open(args.input).convert('RGB')
    img = np.asarray(img).astype(np.float32)
    img /= 255.0
    img = img.transpose(2, 0, 1).reshape((1, 3, 256, 256))
    
    x = chainer.Variable(img)
    z = enc(x)
    y = dec(z) # (batchsize, ch, width, height)

    y = y.reshape(3, 256, 256).transpose(2, 1, 0)
    import matplotlib.pyplot as plt
    plt.imshow(y)
    plt.show()

if __name__ == '__main__':
    main()
