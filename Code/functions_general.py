# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:22:58 2023

@author: felixsr
"""

import os
import io
import PIL.Image, PIL.ImageDraw
import base64
import zipfile
import json
import requests
import numpy as np
import matplotlib.pylab as pl
import glob
import matplotlib.pyplot as plt

import tensorflow as tf

from IPython.display import Image, HTML, clear_output
#import tqdm

import os
os.environ['FFMPEG_BINARY'] = 'ffmpeg'
#import moviepy.editor as mvp
#from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
clear_output()

print(tf.__version__)

def get_living_mask(x, mode='full'):
    if mode=='full':
        alpha = x[:, :, :, 3:4]
    elif mode=='alpha':
        alpha = x[:, :, :, 0:1]
    # cells with alpha>0.1 are considered living, indepently from the neighbor
    return tf.nn.max_pool2d(alpha, 1, [1, 1, 1, 1], 'SAME') > 0.1

def get_inverted_living_mask(x,mode='full'):
    if mode=='full':
        alpha = x[:, :, :, 3:4]
    elif mode=='alpha':
        alpha = x[:, :, :, 0:1]
    # cells with alpha>0.1 are considered living, indepently from the neighbor
    return tf.nn.max_pool2d(alpha, 1, [1, 1, 1, 1], 'SAME') <= 0.1

def get_energy_mask(x, threshold=0.1):
    energy = x[:, :, :, 2:3]
    # cells with energy>0.1 can continue living, indepently from the neighbor
    return tf.nn.max_pool2d(energy, 1, [1, 1, 1, 1], 'SAME') > threshold

def numpy_argwhere(x):
    return np.argwhere(x)

def wrap_pad(x):
    return np.pad(x, [(0, 0), (1, 1), (1, 1), (0, 0)], mode='wrap')