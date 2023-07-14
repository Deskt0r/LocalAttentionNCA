import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import imageio
import argparse
import tensorflow as tf

from functions_general import *
from functions_data import SamplePool, target_change, get_initial_state, create_distance_mask
from functions_update import pre_update, post_update
from functions_visualization_Copy1 import plot_loss, plot_sample, create_frame
from functions_network import CAModel
import config 

if __name__ == "__main__":
    
    mode = "spiral" # spiral, long, short, split, barrier
    run = "5"
    sub_path = mode+"_"+run
    path = "Dynamic_Positioning/04_07_2023_11_03_Logs"
    start = [20,20]
    # spiral
    targets = [[16, 16],[16, 12],[20, 8],[24, 8],[26,12]] #spiral4: energy 8
    
    # split
    #targets = [[24, 20],[28, 16],[28,12],[18, 24],[14, 28],[16,32]] #split1: energy8, split2: energy 5
    #targets = [[25, 15],[30, 10],[35,5],[15, 25],[10, 30],[5,35]] #split3: energy5, split4: energy 8
#targets = [[24, 16],[28, 12],[32,8],[16, 24],[12, 28],[8,32]] #split5: energy5, split6: energy 3, split7: energy start 5, split 3, split8: energy start 5, split 2
    #targets = [[24, 20]] #short1: energy 5
    #targets = [[25, 20]] #long1: energy 5
    
    if not os.path.exists(f'../Visualization/{path}'):
        os.mkdir(f'../Visualization/{path}')
    if not os.path.exists(f'../Visualization/{path}/{sub_path}'):
        os.mkdir(f'../Visualization/{path}/{sub_path}')
    else:
        print('files get overridden')
    
    ca = CAModel()
    dummy = tf.zeros((config.BATCH_SIZE,config.TARGET_SIZE,config.TARGET_SIZE,4))
    ca(dummy)
    #print(ca.summary())
    
    lr = 1e-3
    trainer = tf.keras.optimizers.Adam(lr)
    #trainer = tf.keras.optimizers.legacy.Adam(lr)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=trainer, net=ca)
    ckpt.restore('../Logs/Dynamic_Positioning/04_07_2023_11_03_Logs/TF_Checkpoints/ckpt-5')
    #train_log_dynamic_1/tf_ckpts
    seed = np.zeros([8,40, 40, 4], np.float32)
    seed[0, start[0], start[1], 3] = 1.0 #alpha channel
    seed[0, start[0], start[1], 2] = 8.0 #energy channel
    seed[0, start[0], start[1], 1] = 1.0 #chemical channel
    seed[0, start[0], start[1], 0] = 0.0 #hidden channel
    for j in targets:
        print(j)
        seed[0,j[0],j[1],2] = 8.0
    
    x = seed
    
    fig = plot_sample(x[0],targets, start, title = False)
    
    for t in range(100):
        x = pre_update(x)
        x = ca(x)
        x = post_update(x)
        #print(np.sum(x[0,:,:,2]))
        x_n = x.numpy()
        create_frame(t,x_n[0],targets,start,sub_path,path,sub_path)
    
    frames = []
    for t in range(100):
        image = imageio.v2.imread(f'../Visualization/{path}/{sub_path}/{sub_path}_{t}.png')
        frames.append(image)
        
    imageio.mimsave(f'../Visualization/{path}/{sub_path}/{sub_path}.gif', # output gif
                    frames,          # array of input frames
                    duration = 1)         # optional: duration = 1000 * 1/frames per second