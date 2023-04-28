# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:25:38 2023

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