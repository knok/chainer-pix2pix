# -*- coding: utf-8 -*-
#

from __future__ import print_function

import os
import numpy as np
from PIL import Image

import chainer
from chainer.dataset import dataset_mixin

class Pix2pixDataset(dataset_mixin.DatasetMixin):
    def __init__(self, datadir, cache=True):
        self.datadir = datadir
        self.cache = cache
        files = []
        for fname in os.listdir(datadir):
            if fname.endswith(".png") or fname.endswith(".jpg"):
                files.append(fname)
        self.files = files
        if len(files) <= 0:
            raise FileNotFoundError("no image file in the dir %s" % datadir)
        self.images = []

    def __len__(self):
        return len(self.files)

    def load_image(self, i):
        def _load(i):
            path = os.path.join(self.datadir, self.files[i])
            img = Image.open(path)
            img = np.asarray(img, dtype=np.float32)
            img = img.transpose(2, 0, 1)
            img /= 255.0
            return img
        if self.cache:
            img = self.images.get(i, None)
            if img is None:
                img = _load(i)
        else:
            img = _load(i)
        return img

    # return (A image, B image)
    def get_example(self, i):
        if i > len(self):
            raise IndexError("index too large")
        img = self.load_image(i)
        ch, h, w = img.shape
        w = w // 2
        a_img = np.zeros((ch, h, w), dtype=np.float32)
        b_img = np.zeros((ch, h, w), dtype=np.float32)
        a_img = img[:, :, 0:w]
        b_img = img[:, :, w:w*2]
        return a_img, b_img
