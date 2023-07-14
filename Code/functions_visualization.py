# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:26:40 2023

@author: felixsr
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

def plot_loss(loss_log,d1):
    plt.figure(figsize=(10, 4))
    plt.title('Loss history')
    plt.plot(loss_log, 'r-.', alpha=0.5)
    plt.savefig('../Logs/Dynamic_Positioning/'+ str(d1) +'/Loss_Plots/loss_plot.png')

def plot_sample(sample,targets,start,title=True):
    fig, axs = plt.subplots(nrows=1, ncols=4)
    
    plt.subplots_adjust(wspace=0.4)
    #add title
    #if title:
    #    fig.suptitle('start&target | chemistry | energy | alpha',y=0.25)
    axs[0].set_title('start&target')
    axs[1].set_title('chemistry')
    axs[2].set_title('energy')
    axs[3].set_title('alpha')
        
    for ax in axs:
        # ax.set_yticks(ax.get_xticks()[::2])  # Set ticks at every second value
        ax.tick_params(axis='x', labelsize=7)  # Change font size of x-axis coordinates
        ax.tick_params(axis='y', labelsize=7)
    
    i = sample
    for j in targets:
        i[j[0],j[1],0]=1
    i[start[0],start[1],0]=1
    #i[:,:,3] = ((i[:,:,3] + i[:,:,3].min()) * (1/np.maximum(i[:,:,3].max(),0.00001)) * 255)
    i = abs(i)
    i[:,:,3] = ((i[:,:,3]) * 255)
    #i[:,:,2] = ((i[:,:,2] + i[:,:,2].min()) * (1/np.maximum(i[:,:,2].max(),0.00001)) * 255)
    #i[:,:,2] = ((i[:,:,2]) * (1/5) * 255)
    #i[:,:,1] = ((i[:,:,1] + i[:,:,1].min()) * (1/np.maximum(i[:,:,1].max(),0.00001)) * 255)
    i[:,:,1] = ((i[:,:,1]) * (1/10) * 255)
    #i[:,:,0] = ((i[:,:,0] + i[:,:,0].min()) * (1/np.maximum(i[:,:,0].max(),0.00001)) * 255)
    i[:,:,0] = ((i[:,:,0]) * 255)
    #add data to plots
    for j in range(4):
        axs[j].matshow(i[:,:,j])
            
    return fig

def plot_sample_vertical(sample,targets,start,title=True):
    fig, axs = plt.subplots(nrows=3, ncols=1)
    
    plt.subplots_adjust(hspace=0.7)
    
    axs[0].set_title('chemistry')
    axs[1].set_title('energy')
    axs[2].set_title('alpha')
        
    for ax in axs:
        # ax.set_yticks(ax.get_xticks()[::2])  # Set ticks at every second value
        ax.tick_params(axis='x', labelsize=5)  # Change font size of x-axis coordinates
        ax.tick_params(axis='y', labelsize=5)
    
    i = sample
    for j in targets:
        i[j[0],j[1],0]=1
    i[start[0],start[1],0]=1
    i = abs(i)
    #add data to plots
    for j in range(3):
        axs[j].matshow(i[:,:,j+1])
        
    return fig

def create_frame(t,sample,targets,start,name,path,sub_path):
    fig = plot_sample(sample,targets,start)
    fig.savefig(f'../Visualization/{path}/{sub_path}/{name}_{t}.png', 
                transparent = False,  
                facecolor = 'white'
               )
    fig.set_size_inches(fig.get_size_inches())  # Set figure size to match the original figure
    fig.savefig(f'../Visualization/{path}/{sub_path}/{name}_{t}.pdf'
               )
    plt.close()
    