# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import PIL.Image, PIL.ImageDraw
import json
import numpy as np
import matplotlib.pylab as pl
import tensorflow as tf

from functions_general import *
from functions_data import *
from functions_update import *
from functions_visualization import *

os.environ['FFMPEG_BINARY'] = 'ffmpeg'
clear_output()

print(tf.__version__)
print(os.getcwd())
        
CHANNEL_N = 4        # Number of CA state channels
TARGET_PADDING = 16   # Number of pixels used to pad the target image border
TARGET_SIZE = 40
BATCH_SIZE = 8
POOL_SIZE = 1024
CELL_FIRE_RATE = 1.0

CHEMISTRY_RATE=0.1
ENERGY_RATE=0.1
CA_SIZE=20

TARGET_CELL=[3,3]

TARGET_EMOJI = "🦎" #@param {type:"string"}

TARGET_MODE = "Changing_Corners" #"Random Energy", "Corners","Curriculum Barriers", "Barriers",

EXPERIMENT_TYPE = "Regenerating" #@param ["Growing", "Persistent", "Regenerating"]
EXPERIMENT_MAP = {"Growing":0, "Persistent":1, "Regenerating":2}
EXPERIMENT_N = EXPERIMENT_MAP[EXPERIMENT_TYPE]

USE_PATTERN_POOL = False#[0, 1, 1][EXPERIMENT_N]
#DAMAGE_N = [0, 0, 3][EXPERIMENT_N]  # Number of patterns to damage in a batch

'''Network'''

class Convolutional_Attention(tf.keras.layers.Layer):
    def __init__(self, h, d_k, d_v, d_model, flag_padding=True, input_dims=[BATCH_SIZE,TARGET_SIZE,TARGET_SIZE,4], **kwargs):
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
    def __init__(self, channel_n=CHANNEL_N, fire_rate=CELL_FIRE_RATE, input_dims=[BATCH_SIZE,TARGET_SIZE,TARGET_SIZE,4]):
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

'''Sample Pool'''

def get_initial_state(N, size, chan, pos_start, pos_target,init_type="seed", seed=None):
    if init_type == "random":
        if N is None:
            return (np.random.random([size, size, chan])-0.5)*0.2
        x= (np.random.random([N,size, size, chan])-0.5)*0.2
    elif init_type == "zeros":
        if N is None:
            return np.zeros([size, size, chan])
        x= np.zeros([N,size, size, chan])
    elif init_type == 'seed':
        if N is None:
            seed = np.zeros([size, size, chan], np.float32)
            seed[pos_start[0], pos_start[1], 3] = 1.0 #alpha channel
            seed[pos_start[0], pos_start[1], 2] = 10.0 #energy channel
            seed[pos_start[0], pos_start[1], 1] = 1.0 #chemical channel
            seed[pos_start[0], pos_start[1], 0] = 0.0 #hidden channel
            seed[pos_target[0],pos_target[1],2] = 10.0 
            x=seed
            #x = np.repeat(seed[None, ...], N, 0)
        elif N == 1:
            seed = np.zeros([N, size, size, chan], np.float32)
            seed[pos_start[0], pos_start[1], 3] = 1.0 #alpha channel
            seed[pos_start[0], pos_start[1], 2] = 10.0 #energy channel
            seed[pos_start[0], pos_start[1], 1] = 1.0 #chemical channel
            seed[pos_start[0], pos_start[1], 0] = 0.0 #hidden channel
            seed[pos_target[0],pos_target[1],2] = 10.0 
        else:
            arrays=[]
            for i in range(len(pos_start)):
                seed = np.zeros([size, size, chan], np.float32)
                #print(pos_start[i])
                seed[pos_start[i][0], pos_start[i][1], 3] = 1.0 #alpha channel
                seed[pos_start[i][0], pos_start[i][1], 2] = 10.0 #energy channel
                seed[pos_start[i][0], pos_start[i][1], 1] = 1.0 #chemical channel
                seed[pos_start[i][0], pos_start[i][1], 0] = 0.0 #hidden channel
                seed[pos_target[i][0],pos_target[i][1],2] = 10.0 
                arrays.append(np.repeat(np.expand_dims(seed,axis=0),BATCH_SIZE, axis=0))
                #print('array',arrays[i][0,16:24,16:24,3])
            x = np.concatenate(arrays, axis=0)
            #print('x',x.shape)
    elif init_type =='seed_old':
        seed = np.zeros([size, size, chan], np.float32)
        seed[pos_start[0], pos_start[1], 3] = 1.0 #alpha channel
        seed[pos_start[0], pos_start[1], 2] = 30.0 #energy channel
        seed[pos_start[0], pos_start[1], 1] = 1.0 #chemical channel
        seed[pos_start[0], pos_start[1], 0] = 0.0 #hidden channel
        seed[pos_target[0],pos_target[1],2] = 10.0 
        seed[pos_start[0]+(pos_target[0]-pos_start[0])//4,pos_start[1]+(pos_target[1]-pos_start[1])//4,2]=10.0
        seed[pos_start[0]+(pos_target[0]-pos_start[0])//2,pos_start[1]+(pos_target[1]-pos_start[1])//2,2]=10.0
        seed[pos_start[0]+((pos_target[0]-pos_start[0])//4)*3,pos_start[1]+((pos_target[1]-pos_start[1])//4)*3,2]=10.0
        x = np.repeat(seed[None, ...], N, 0)
    return x.astype("float32")

rng = np.random.default_rng()

def perturb_position(positions,radius=2, proba_move=1.):    
    if positions=='outer corners':
        perturb_pos = rng.choice([2,3,4,TARGET_SIZE-3,TARGET_SIZE-4,TARGET_SIZE-5], size=(2), replace=True)
    if positions=='middle corners':
        perturb_pos = rng.choice([8,9,10,TARGET_SIZE-9,TARGET_SIZE-10,TARGET_SIZE-11], size=(2), replace=True)
    if positions=='inner corners':
        perturb_pos = rng.choice([13,14,15,TARGET_SIZE-14,TARGET_SIZE-15,TARGET_SIZE-16], size=(2), replace=True)
    if  positions == 'center':
        perturb_pos = [TARGET_SIZE//2,TARGET_SIZE//2]+ rng.integers(-3,3,size=(2))
    return perturb_pos

def create_distance_mask(output_coo, N=TARGET_SIZE, shape="square"):
    if shape=="square":
        d = np.zeros((1,N,N,1), "float32")
        d[0][output_coo[0]][output_coo[1]][0] = 1.
        d = tf.convert_to_tensor(d)
        for t in range(N):
            m = tf.nn.max_pool2d(d, 3, [1, 1, 1, 1], "SAME")
            m = tf.cast(m>0, tf.float32)
            #we add a little amount of noise such that each value in d will be unique
            #this becomes useful when we want to easily recover *one* of the living cells the colsest
            #to the output.
            m*= tf.random.uniform((1,N,N,1))*0.001 + 0.9995
            d +=m
        d = tf.reshape(d, (N,N))
        return d*100

def create_distance_masks(batches, pool_size,batch_size, grid_size, pos, shape="square"):
    arrays =[]
    for i,b in enumerate(batches):
        mask = create_distance_mask(pos[i])
        arrays.append(np.repeat(np.expand_dims(mask,axis=0),batch_size, axis=0))
        #print('array',arrays[0].shape) #array (8, 40, 40)
    masks = np.concatenate(arrays, axis=0)
    return masks.astype("float32")

def target_change(x, learning_step):
    if learning_step=='early':
        pos = perturb_position('inner corners')
    if learning_step=="medium":
        pos = perturb_position('middle corners')
    if learning_step=="late":
        pos = perturb_position('outer corners')
    #new = get_initial_state(1, self.grid_size, self.nb_chan, self.batches_input_coo[self.current_idx], pos,init_type="seed", seed=None)
    x[pos[0],pos[1],2] = 5.0 
    dist_mask = create_distance_mask(pos)
    dist_mask = np.expand_dims(dist_mask,axis=-1)
    return pos, x, dist_mask

class SamplePool:
    """A pool of batches. With each batch comes the output cells coordinates
        attached and the list of hidden cells."""
    def __init__(self, pool_size, grid_size,nb_chan,output_centers,
                 input_centers, radius, batch_size,
                 range_nb_hidden_cells=(0,3), damage=False,
                 proba_move=1., move_inputs=True,
                 move_outputs=True, init_type="seed", seed=None, random_start_and_target=False):
        """pool_size: the number of batches in the pool"""
        self.grid_size = grid_size
        self.nb_chan = nb_chan
        #self.range_nb_hidden_cells = range_nb_hidden_cells
        self.proba_move_out = proba_move*move_outputs
        self.proba_move_in = proba_move*move_inputs

        self.init_type = init_type
        self.seed = seed
        self.random_start_and_target = random_start_and_target

        self.size = pool_size
        self.damage = damage
        self.radius = radius
        self.output_centers = output_centers
        self.input_centers = input_centers
        self.batch_size = batch_size
        self.current_idx = None
        if random_start_and_target:
            self.batches_output_coo = [perturb_position(output_centers,radius,self.proba_move_out) for i in range(pool_size)]
            self.batches_input_coo = [perturb_position(input_centers,radius,self.proba_move_in) for i in range(pool_size)]
            #self.batches = get_initial_state(pool_size*batch_size, grid_size,nb_chan, pos_start=pos_cen, pos_target=pos_tar, init_type=self.init_type, seed=seed)
            #self.batches = self.batches.reshape(pool_size,batch_size, grid_size, grid_size, nb_chan)
        elif TARGET_MODE=="Changing_Corners":
            #self.pos_tar = perturb_position('inner corners',radius,self.proba_move_out)
            #self.pos_cen =perturb_position('center',radius,self.proba_move_in)
            self.batches_output_coo = [perturb_position('inner corners',radius,self.proba_move_out) for i in range(pool_size)]
            self.batches_input_coo = [perturb_position('center',radius,self.proba_move_in) for i in range(pool_size)]
            self.batches = get_initial_state(pool_size, grid_size,nb_chan, pos_start=self.batches_input_coo, pos_target=self.batches_output_coo, init_type=self.init_type, seed=seed)
            self.batches = self.batches.reshape(pool_size,batch_size, grid_size, grid_size, nb_chan)
            self.distance_masks = create_distance_masks(self.batches, pool_size,batch_size, grid_size, self.batches_output_coo)
            self.distance_masks = self.distance_masks.reshape(pool_size,batch_size, grid_size, grid_size, 1)
            #print(self.batches_output_coo)
            self.batches_output_coo = [[item  for i in range(8)] for item in self.batches_output_coo]
            #print(self.batches_output_coo)
            #print(self.distance_masks.shape)
        else:
            self.pos_tar = perturb_position('inner corners',radius,self.proba_move_out)
            self.pos_cen =perturb_position('center',radius,self.proba_move_in)
            self.batches_output_coo = [self.pos_tar for i in range(pool_size)]
            self.batches_input_coo = [self.pos_cen for i in range(pool_size)]
            self.batches = get_initial_state(pool_size*batch_size, grid_size,nb_chan, pos_start=self.pos_cen, pos_target=self.pos_tar, init_type=self.init_type, seed=seed)
            self.batches = self.batches.reshape(pool_size,batch_size, grid_size, grid_size, nb_chan)
        
        # self.batches_hidden_inputs = []
        # for i in range(len(self.batches_input_coo)):
        #     nb_hid = np.random.randint(range_nb_hidden_cells[0], range_nb_hidden_cells[1]+1)
        #     hid = random_choice(self.batches_input_coo[i], nb_hid)

        #     self.batches_hidden_inputs.append(hid)

    def sample(self):
        idx = np.random.randint(self.size)
        self.current_idx = idx #the idicies of the samples currently under update

        # if self.damage and np.random.random() < 0.5:
        #     grids = damage_grids(self.batches[idx])
        # else:
        grids = self.batches[idx]
        masks = self.distance_masks[idx]

        return (tf.convert_to_tensor(grids), self.batches_output_coo[idx],
               self.batches_input_coo[idx], tf.convert_to_tensor(masks)) #, self.batches_hidden_inputs[idx])
    
    def reinit_and_sample(self):
        idx = np.random.randint(self.size)
        self.current_idx = idx
        self.batches_output_coo[idx] = perturb_position(self.output_centers,
                                                self.radius,self.proba_move_out)
        self.batches_input_coo[idx] = perturb_position(self.input_centers,
                                                self.radius,self.proba_move_in)

        # nb_hid = np.random.randint(self.range_nb_hidden_cells[0],
        #                           self.range_nb_hidden_cells[1]+1)
        # hid = random_choice(self.batches_input_coo[idx], nb_hid)

        # self.batches_hidden_inputs[idx] = hid
        self.batches[idx] = get_initial_state(self.batch_size, self.grid_size,
                                              self.nb_chan, init_type=self.init_type, seed=self.seed)
        return (self.batches[idx], self.batches_output_coo[idx],
               self.batches_input_coo[idx])#, self.batches_hidden_inputs[idx])
    
    def reinit_start_and_sample(self):
        idx = np.random.randint(self.size)
        self.current_idx = idx
        
        self.batches_input_coo[idx] = perturb_position(self.input_centers,
                                                self.radius,self.proba_move_in)

        # self.batches[idx] = get_initial_state(self.batch_size, self.grid_size,
        #                                       self.nb_chan, init_type=self.init_type, seed=self.seed)
        self.batches[idx] = get_initial_state(self.batch_size, self.grid_size, self.nb_chan, 
                                         pos_start=self.pos_cen, pos_target=self.pos_tar, 
                                         init_type=self.init_type, seed=self.seed)

        return (tf.convert_to_tensor(self.batches[idx]), self.batches_output_coo[idx],
               self.batches_input_coo[idx])#, self.batches_hidden_inputs[idx])
    
    def commit(self, updated_batch, updated_masks, updated_target_pos):
        self.batches[self.current_idx] = updated_batch
        self.distance_masks[self.current_idx] = updated_masks
        self.batches_output_coo[self.current_idx] = updated_target_pos
        self.current_idx = None
    
def save_batch_arrays(batch,step_i,loss):
    np.save('train_log/%04d_batch_%04d_loss.npy'%(step_i, loss) , batch.numpy())

def plot_loss(loss_log):
    pl.figure(figsize=(10, 4))
    pl.title('Loss history (log10)')
    #pl.plot(np.log10(loss_log), '.', alpha=0.1)
    pl.plot(loss_log, '.', alpha=0.1)
    pl.savefig('loss_plot.png')
    #pl.show()

pool = SamplePool(4, grid_size=TARGET_SIZE, nb_chan=4, output_centers='corners', input_centers='center', radius=2, batch_size=8)
x0,target_ind,start_ind,dist_masks = pool.sample()
print('Target:',target_ind)
print('Start:',start_ind)

#distances = create_distance_mask(target_ind, 40)
#print(distances)

@tf.function
def loss_cell_distance(x, distance_masks,batch_size=BATCH_SIZE,
                            energy_budget=10, k=10, return_array=False):
    if len(distance_masks.shape)==4:
        #distance_masks=np.reshape(distance_masks, (8,40,40))
        distance_masks=tf.reshape(distance_masks, (8,40,40))
    energy = x[:,:,:,3]
    non_zero = tf.cast(energy != 0, tf.float32)
    #print('1',non_zero.shape) #1 (8, 40, 40)
    #print('distances',distance_masks.shape) # distances (40, 40) & distances (8, 40, 40)
    #print('multiplication',(non_zero*distance_masks).shape) # multiplication (8, 40, 40) & multiplication (8, 40, 40)
    max = tf.reduce_max(non_zero*distance_masks, axis=[1,2])
    #print('2',max.shape) #2 (8,)
    max = tf.reshape(max, (batch_size,1,1))
    #print('3',max.shape) # 3 (8, 1, 1)
    max_pos = tf.cast(tf.math.equal(distance_masks,max), tf.float32)
    #print('4',np.argwhere(max_pos)) # 4 (8, 40, 40)
    energy_cliped = -tf.nn.relu(1-energy) +1
    proximity_score = tf.reduce_sum(max_pos*energy_cliped*distance_masks, axis=[1,2])
    # regularization term: increased loss, if no cell allive
    penalty = 1000 - tf.reduce_sum(energy,axis=[1,2])
    #the proximity score is subject to maximization hence the -k factor
    batch_losses = -k*proximity_score #+penalty#+ energy_cost
    if return_array:
        return batch_losses, proximity_score#, energy_cost
    else:
        return batch_losses

def perturb_weights(model):
    # in case of a alpha==0, perturb the weights of the neural network
    ws = [w + tf.random.normal(w.shape) for w in model.get_weights()]
    model.set_weights(ws)
    return model

ca = CAModel()
dummy = tf.zeros((BATCH_SIZE,TARGET_SIZE,TARGET_SIZE,4))
ca(dummy)
print(ca.summary())

loss_log = []
lr = 1e-3
print('learning rate is:',lr)
trainer = tf.keras.optimizers.Adam(lr)

ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=trainer, net=ca)
manager = tf.train.CheckpointManager(ckpt, './train_log/tf_ckpts', max_to_keep=10)

def debugging(x):
    #living_mask = get_living_mask(x)
    #tf.print('energy',tf.reduce_sum(x[:,:,:,3],axis=[1,2]),summarize=-1)
    #tf.print('alpha',tf.reduce_sum(tf.cast(living_mask[:,:,:,0],tf.float32),axis=[1,2]),summarize=-1)
    tf.print(x[0,pool.pos_tar[0]-1:pool.pos_tar[0]+2,pool.pos_tar[1]-1:pool.pos_tar[1]+2,2])
    
#@tf.function
def train_step(x,dist_masks):
    iter_n = tf.random.uniform([], 8, 16, tf.int32)
    #iter_n = tf.random.uniform([], 8, 16, tf.int32)
    #iter_n=1
    max_loss = 0
    max_loss_iteration = 0
    with tf.GradientTape() as g:
        for i in tf.range(iter_n):
            # if tf.reduce_sum(x[:,:,:,3:4]) <1e-4:
            #     x,target_ind,start_ind  = pool.reinit_start_and_sample()
            x = pre_update(x)
            #print('1 update')
            #debugging(x)
            x = ca(x)
            #print('net')
            #debugging(x)
            #tf.print('alpha 2',x[0,10:30,10:30,3])
            x = post_update(x)
            #print('2 update')
            #debugging(x)
        batch_loss = loss_cell_distance(x,dist_masks)
        loss = tf.reduce_mean(batch_loss)
        if loss<max_loss:
            max_loss = loss
            max_loss_iteration = i
    grads = g.gradient(loss, ca.weights)
    #print('grads',grads)
    grads = [g/(tf.norm(g)+1e-8) for g in grads]
    #print('grads2',grads)
    trainer.apply_gradients(zip(grads, ca.weights))
    
    print(' - loss',max_loss.numpy(),'in iteration',max_loss_iteration.numpy(),'of',iter_n.numpy())
    return x, loss, batch_loss

k = 0
ckpt.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
  print("Restored from {}".format(manager.latest_checkpoint))
else:
  print("Initializing from scratch.")
  
for i in range(100+1):
    # take batch from pool
    # run train step (low iter_n? no, wont work for later stages)
    # reinitiate all died out samples
    # reinitiate the highest loss sample
    # after xyz iterations: start randomly adding! with curriculum
    #                       only those with lowest loss
    
    
    # if USE_PATTERN_POOL:
    #     batch = pool.sample(BATCH_SIZE)
    #     x0 = batch.x
    #     loss_rank = loss_cell_distance(x0).numpy().argsort()[::-1]
    #     x0 = x0[loss_rank]
    #     x0[:1] = seed
    # else:
    #     x0 = np.repeat(seed[None, ...], BATCH_SIZE, 0)
    x,target_ind,start_ind,dist_masks = pool.sample() 
    #tf.print('energy',x[0,:,:,2], summarize=-1)
    x, loss, batch_loss = train_step(x0,dist_masks)
    #print('loss',loss)
    # if USE_PATTERN_POOL:
    #     batch.x[:] = x
    #     batch.commit()

    step_i = len(loss_log)
    loss_log.append(loss.numpy())
  
    #if step_i%10 == 0:
    # if step_i%5 == 0:
    #     if USE_PATTERN_POOL:
    #         generate_pool_figures(pool, step_i)
    #     else:
    save_batch_arrays(x, step_i, loss.numpy())
    if step_i%25 == 0:
        ckpt.step.assign(int(step_i))
        clear_output()
        #visualize_batch(x0, x, step_i)
        plot_loss(loss_log)
        #ca.save('train_log/'+str(step_i)+'_model')
        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
      
        #export_model(ca, 'train_log/%04d'%step_i)

    # print('\r step: %d, log10(loss): %.3f'%(len(loss_log), np.log10(loss)), end='')
    print('\r step: %d, loss: %.3f'%(len(loss_log), loss), end='')
    
    ar = x.numpy()
    ar2 = dist_masks.numpy()
    ar3 = target_ind
    if rng.random()<0.9:
      #we replace the worst sample by the initial state
      batch_loss = batch_loss.numpy()
      #worst_ind = np.argmax(loss)
      #best_ind = np.argmin(loss)
      ind = np.argsort(batch_loss) # good to bad
      print('len ind',len(ind))
      #print('worst ind',worst_ind)
      #print('ind',worst_ind)
      ar[ind[-1]] = get_initial_state(None, 40, 4, start_ind, target_ind[ind[-1]],init_type="seed", seed=None)
      ar2[ind[-1]] = np.expand_dims(create_distance_mask(target_ind[ind[-1]]),axis=-1)
      if i>20:
          ar3[ind[0]], ar[ind[0]], ar2[ind[0]] = target_change(ar[ind[0]],'late')
          ar3[ind[1]], ar[ind[1]], ar2[ind[1]]  = target_change(ar[ind[1]],'medium')
          ar3[ind[2]], ar[ind[2]], ar2[ind[2]]  = target_change(ar[ind[2]],'early')
      elif i>15: 
          ar3[ind[0]], ar[ind[0]], ar2[ind[0]] = target_change(ar[ind[0]],'medium')
          ar3[ind[1]], ar[ind[1]], ar2[ind[1]] = target_change(ar[ind[1]],'early')
      elif i>5:
          ar3[ind[0]], ar[ind[0]], ar2[ind[0]] = target_change(ar[ind[0]],'early')
    


### TODO ###
# check: is energy distribution correct? 
# right direction?
# increase upper limit for iter_n: do the cells reach the target? or form just bigger blob?
# decrease start energy: still blobs or sleeker?
# pool sampling
# increase margin within distance mask?
# save files differently (use export_model and checkpoints, py_func does not work with .save())