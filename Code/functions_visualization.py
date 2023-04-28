# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:26:40 2023

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

def plot_sample(sample,target,start):
    fig, axs = plt.subplots(nrows=1, ncols=4)

    #add title
    fig.suptitle('start&target | chemistry | energy | alpha')

    i = sample
    i[target[0],target[1],0]=1
    i[start[0],start[1],0]=1
    #i[:,:,3] = ((i[:,:,3] + i[:,:,3].min()) * (1/np.maximum(i[:,:,3].max(),0.00001)) * 255)
    i = abs(i)
    i[:,:,3] = ((i[:,:,3]) * 255)
    i[:,:,2] = ((i[:,:,2] + i[:,:,2].min()) * (1/np.maximum(i[:,:,2].max(),0.00001)) * 255)
    i[:,:,1] = ((i[:,:,1] + i[:,:,1].min()) * (1/np.maximum(i[:,:,1].max(),0.00001)) * 255)
    i[:,:,0] = ((i[:,:,0] + i[:,:,0].min()) * (1/np.maximum(i[:,:,0].max(),0.00001)) * 255)
    #add data to plots
    axs[0].matshow(i[:,:,0])
    axs[1].matshow(i[:,:,1])
    axs[2].matshow(i[:,:,2])
    axs[3].matshow(i[:,:,3])
    return fig

def create_frame(t,sample,target,start,name,path):
    fig = plot_sample(sample,target,start)
    fig.savefig(f'..\\Visualization\\{path}\\img_{name}_{t}.png', 
                transparent = False,  
                facecolor = 'white'
               )
    plt.close()