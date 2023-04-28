# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:28:42 2023

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

from functions_general import get_energy_mask

import os
os.environ['FFMPEG_BINARY'] = 'ffmpeg'
#import moviepy.editor as mvp
#from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
clear_output()

print(tf.__version__)

def pre_update(x):
    energy_mask = get_energy_mask(x)
    x1,x2,x3,x4 = tf.split(x,[1,1,1,1],axis=-1)
    # update alpha and hidden channel for cells that died due to energy shortage
    x4 = x4 * tf.cast(energy_mask, tf.float32) 
    x1 = x1 * tf.cast(energy_mask, tf.float32) 
    # update chemical channel according to living/dead cells
    inverted_living_mask = get_inverted_living_mask(x4,mode='alpha')
    living_mask = get_living_mask(x4,mode='alpha')
    x2 += CHEMISTRY_RATE*tf.cast(living_mask, tf.float32)
    x2 -= CHEMISTRY_RATE*tf.cast(inverted_living_mask, tf.float32)
    x2 = tf.clip_by_value(x2, 0, 10)
    # update energy channel
    x3 -= ENERGY_RATE*tf.cast(living_mask, tf.float32)
    return tf.concat([x1,x2,x3,x4], axis=-1)

def local_energy_update(x_padded,sum_alphas,idx,energy_update):
    patch = np.divide(x_padded[idx[0],idx[1]:idx[1]+3,idx[2]:idx[2]+3,2:3],sum_alphas[idx[0],idx[1]:idx[1]+3,idx[2]:idx[2]+3,0:1])
    energy_update[idx[0],idx[1],idx[2],0] = np.sum(patch)/x_padded[idx[0],idx[1],idx[2],3]
    return energy_update

def global_energy_update(x):
    living_mask = get_living_mask(x)
    x1,x2,x3,x4 = tf.split(x,[1,1,1,1],axis=-1)
    x4 = x4 * tf.cast(living_mask, tf.float32)
    x4_padded = tf.numpy_function(wrap_pad, [x4], tf.float32)
    x3_padded = tf.numpy_function(wrap_pad, [x3], tf.float32)
    # conv to get sum alpha for every neighborhood
    ones = np.ones((3,3))
    kernel = tf.expand_dims(tf.expand_dims(ones, -1), -1)
    sum_alphas = tf.nn.conv2d(tf.cast(x4_padded[...,0:1], tf.float32), tf.cast(kernel, tf.float32), [1, 1, 1, 1], 'SAME')
    living_index = numpy_argwhere(living_mask)

    energy_update1 = np.zeros((BATCH_SIZE,TARGET_SIZE,TARGET_SIZE,1))
    """There are three cases for energy:
        1. energy (can be 0, but does not matter) on living cell. 
        In this case, redistribute energy in neighborhood to cell
        2. energy without living cell in neighborhood. 
        In this case, do nothing
        3. energy without living cell on it, but with one in neighborhood.
        In this case, alpha=0.
        Then either nothing happens like in 2 (use a mask for the patch calculation)
        Or it gets drained (use energy_delete_update, which is not done yet)
        """
        
    for idx in living_index:
        mask = tf.reshape((sum_alphas[idx[0],idx[1]:idx[1]+3,idx[2]:idx[2]+3,0]!=0),-1)
        patch1 = tf.math.multiply(tf.math.divide(tf.reshape(x3_padded[idx[0],idx[1]:idx[1]+3,idx[2]:idx[2]+3,0:1],-1)[mask], tf.reshape(sum_alphas[idx[0],idx[1]:idx[1]+3,idx[2]:idx[2]+3,0],-1)[mask]), x[idx[0],idx[1],idx[2],3:4])
        energy_update1[idx[0],idx[1],idx[2],0:1] = tf.reduce_sum(patch1)

    living_mask = get_living_mask(x)
    x3 += tf.math.scalar_mul(-1.0, x3) * tf.cast(living_mask, tf.float32) + energy_update1 * tf.cast(living_mask, tf.float32)   
    
    return tf.concat([x1,x2,x3,x4], axis=-1)

def post_update(x):
    x=global_energy_update(x)
    return x