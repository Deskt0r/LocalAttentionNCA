# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:25:19 2023

@author: felixsr
"""
import tensorflow as tf
import config

from functions_general import wrap_pad

class Convolutional_Attention(tf.keras.layers.Layer):
    def __init__(self, h, d_k, d_v, d_model, flag_padding=True, input_dims=[config.BATCH_SIZE,config.TARGET_SIZE,config.TARGET_SIZE,4], **kwargs):
        super(Convolutional_Attention, self).__init__(**kwargs)
        self.input_dims=input_dims
        self.flag_padding = flag_padding
        self.attention = tf.keras.layers.Attention(use_scale=True)  # Scaled dot product attention 
        self.heads = h  # Number of attention heads to use
        self.d_k = d_k  # Dimensionality of the linearly projected queries and keys
        self.d_v = d_v  # Dimensionality of the linearly projected values
        self.d_model = d_model
        self.W_q = tf.keras.layers.Dense(d_k)  # Learned projection matrix for the queries
        #self.W_k = tf.keras.layers.Dense(d_k)  # Learned projection matrix for the keys
        self.W_v = tf.keras.layers.Dense(d_v)  # Learned projection matrix for the values
        self.W_o = tf.keras.layers.Dense(d_model)  # Learned projection matrix for the multi-head output
    
    @tf.function
    def image_patches_layer(self,x,input_dims,mode='batch'):
        # extract patches. input [B,padded_h,padded_w,c], output [B,h,w,3*3*c]
        b, h, w, c=input_dims
        x = tf.image.extract_patches(x,
                                        sizes=[1, 3, 3, 1],
                                        strides=[1, 1, 1, 1],
                                        rates=[1, 1, 1, 1],
                                        padding='VALID')
        patch_dim=9*c
        if mode=='batch':
            # reshape to [Batch,h*w,9,c]
            x = tf.reshape(x,shape=[b,h*w,int(patch_dim/c),int(patch_dim/9)])
        elif mode=='patch':
            x = tf.reshape(x,shape=[b,h,w,int(patch_dim/(c*3)),int(patch_dim/(c*3)),int(patch_dim/9)])
        return x
    
    @tf.function
    def call(self, x):
        b, h, w, c=self.input_dims
        # apply circular padding to input
        if self.flag_padding:
            x = tf.numpy_function(wrap_pad, [x], tf.float32)
        # extract the patches from padded input
        x = self.image_patches_layer(x,self.input_dims)
        # Apply linear projections
        query = tf.reshape(self.W_q(x),shape=[b*h*w,9,c])
        value = tf.reshape(self.W_v(x),shape=[b*h*w,9,c])
        # key = tf.reshape(self.W_k(x),shape=[-1,9,c])
        # Apply Scaled Dot Product Attention
        x = self.attention([query,value],training=True)
        # reshape and linearly project into output
        x = tf.reshape(x,shape=[b,h,w,9*c])
        x = self.W_o(x)
        #print('shape', x.shape) (32, 40, 40, 36)
        # output shape should be [b,h,w,d_model]
        return x

class CAModel(tf.keras.Model):
    def __init__(self, channel_n=config.CHANNEL_N, fire_rate=config.CELL_FIRE_RATE, input_dims=[config.BATCH_SIZE,config.TARGET_SIZE,config.TARGET_SIZE,4]):
        super().__init__()
        self.input_dims=input_dims
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        #self.conv_attention = Convolutional_Attention(1, self.channel_n, self.channel_n, 9*self.channel_n)

        # self.dmodel = tf.keras.Sequential([
        #       Convolutional_Attention(1, self.channel_n, self.channel_n, 9*self.channel_n),
        #       tf.keras.layers.Conv2D(128, 1, activation=tf.nn.relu),
        #       tf.keras.layers.Conv2D(1, 1, activation=None,
        #           kernel_initializer=tf.zeros_initializer),
        # ])
        
        self.conv_att = Convolutional_Attention(1, self.channel_n, self.channel_n, 4*self.channel_n)
        #self.conv_att = tf.keras.layers.Conv2D(kernel_size = (3,3), filters=20, padding='SAME',activation=tf.nn.relu)
        self.conv1 = tf.keras.layers.Conv2D(32, 1, activation=tf.nn.relu)#activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(1, 1, activation=None)#, kernel_initializer=tf.zeros_initializer)
    
        #self(tf.zeros(self.input_dims))  # dummy call to build the model

    @tf.function
    def call(self, x, fire_rate=None, angle=0.0, step_size=1.0):
        x_padded = tf.numpy_function(wrap_pad, [x], tf.float32)
        #pre_life_mask = get_living_mask(x)
        # split, only alpha values are updated
        x1,x2 = tf.split(x,[3,1],axis=-1)
        dx2 = self.conv_att(x)
        #print('dx2 shape',dx2.shape) # (32, 40, 40, 36)
        #print('attention',tf.math.count_nonzero(dx2))
        dx2 = self.conv1(dx2)
        dx2 = self.conv2(dx2)
        dx2 = dx2*step_size
        # print('dx2 shape',dx2.shape) #(32, 40, 40, 1)
        #tf.print('dx2',dx2[0,18:23,18:23,0])
        #print(dx.shape)
        if fire_rate is None:
            fire_rate = self.fire_rate
        #update_mask = tf.random.uniform(tf.shape(x[:, :, :, 3:4])) <= fire_rate
        #tf.print('tf.cast(x2 <= 0, tf.float32)',tf.cast(x2 <= 0, tf.float32)[0,18:23,18:23,0])
        #tf.print('tf.nn.max_pool2d',tf.nn.max_pool2d(x_padded[...,3:4], 3, [1, 1, 1, 1], 'VALID')[0,18:23,18:23,0])
        default_fr_dead_cells = tf.cast(x2 <= 0, tf.float32)*tf.nn.max_pool2d(x_padded[...,3:4], 3, [1, 1, 1, 1], 'VALID')
        #the dead cells get the fr of their most excited neighbor
        #tf.print('default_fr_dead_cells', default_fr_dead_cells[0,18:23,18:23,0])
        update_mask =  (x2 + default_fr_dead_cells - tf.random.uniform(tf.shape(x[:, :, :, :1]))) >= 0
        update_mask = tf.cast(update_mask, tf.float32)
        #tf.print('update mask',update_mask[0,18:23,18:23,0])
        # print('update mask',update_mask.shape) #update mask (32, 40, 40, 1)
        # print(update_mask[0,18:23,18:23,0])
        #print(tf.math.count_nonzero(dx2))
        #tf.print('update',(dx2 * tf.cast(update_mask, tf.float32))[0,18:23,18:23,0])
        x2 += dx2 * tf.cast(update_mask, tf.float32)
        #tf.print('x2 1',x2[0,18:23,18:23,0])
        #print(x.shape)
        #post_life_mask = get_living_mask(x2,mode='alpha')
        #life_mask = pre_life_mask & post_life_mask
        #x2 = x2 * tf.cast(life_mask, tf.float32)
        #x2 = tf.clip_by_value(x2[..., 0:1], 0.0, 1.0)
        x2 = tf.clip_by_value(x2[..., 0:1], 0, 1.0)
        #tf.print('x2 2',x2[0,10:30,10:30,0])
        return tf.concat([x1,x2], axis=-1)
    
class CAModel_conv(tf.keras.Model):
    def __init__(self, channel_n=config.CHANNEL_N, fire_rate=config.CELL_FIRE_RATE, input_dims=[config.BATCH_SIZE,config.TARGET_SIZE,config.TARGET_SIZE,4]):
        super().__init__()
        self.input_dims=input_dims
        self.channel_n = channel_n
        self.fire_rate = fire_rate
        #self.conv_attention = Convolutional_Attention(1, self.channel_n, self.channel_n, 9*self.channel_n)
        
        #self.conv_att = Convolutional_Attention(1, self.channel_n, self.channel_n, 4*self.channel_n)
        self.conv_att = tf.keras.layers.Conv2D(kernel_size = (3,3), filters=20, padding='SAME',activation=tf.nn.relu)
        self.conv1 = tf.keras.layers.Conv2D(32, 1, activation=tf.nn.relu)#activation=tf.nn.relu)
        self.conv2 = tf.keras.layers.Conv2D(1, 1, activation=None)#, kernel_initializer=tf.zeros_initializer)
    
        #self(tf.zeros(self.input_dims))  # dummy call to build the model

    @tf.function
    def call(self, x, fire_rate=None, angle=0.0, step_size=1.0):
        x_padded = tf.numpy_function(wrap_pad, [x], tf.float32)
        #pre_life_mask = get_living_mask(x)
        # split, only alpha values are updated
        x1,x2 = tf.split(x,[3,1],axis=-1)
        dx2 = self.conv_att(x)
        #print('dx2 shape',dx2.shape) # (32, 40, 40, 36)
        dx2 = self.conv1(dx2)
        dx2 = self.conv2(dx2)
        dx2 = dx2*step_size
        # print('dx2 shape',dx2.shape) #(32, 40, 40, 1)
        if fire_rate is None:
            fire_rate = self.fire_rate
        #update_mask = tf.random.uniform(tf.shape(x[:, :, :, 3:4])) <= fire_rate
        default_fr_dead_cells = tf.cast(x2 <= 0, tf.float32)*tf.nn.max_pool2d(x_padded[...,3:4], 3, [1, 1, 1, 1], 'VALID')
        #the dead cells get the fr of their most excited neighbor
        update_mask =  (x2 + default_fr_dead_cells - tf.random.uniform(tf.shape(x[:, :, :, :1]))) >= 0
        update_mask = tf.cast(update_mask, tf.float32)
        # print('update mask',update_mask.shape) #update mask (32, 40, 40, 1)
        x2 += dx2 * tf.cast(update_mask, tf.float32)
        x2 = tf.clip_by_value(x2[..., 0:1], 0, 1.0)
        return tf.concat([x1,x2], axis=-1)