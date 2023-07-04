# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:25:38 2023

@author: felixsr
"""
import numpy as np
import tensorflow as tf
import config

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
                arrays.append(np.repeat(np.expand_dims(seed,axis=0),config.BATCH_SIZE, axis=0))
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

def perturb_position(positions,radius=2, proba_move=1.):    
    if positions=='outer corners':
        perturb_pos = config.RNG.choice([2,3,4,config.TARGET_SIZE-3,config.TARGET_SIZE-4,config.TARGET_SIZE-5], size=(2), replace=True)
    if positions=='middle corners':
        perturb_pos = config.RNG.choice([8,9,10,config.TARGET_SIZE-9,config.TARGET_SIZE-10,config.TARGET_SIZE-11], size=(2), replace=True)
    if positions=='inner corners':
        perturb_pos = config.RNG.choice([13,14,15,config.TARGET_SIZE-14,config.TARGET_SIZE-15,config.TARGET_SIZE-16], size=(2), replace=True)
    if  positions == 'center':
        perturb_pos = [config.TARGET_SIZE//2,config.TARGET_SIZE//2]+ config.RNG.integers(-3,3,size=(2))
    return perturb_pos

def create_distance_mask(output_coo, N=config.TARGET_SIZE, shape="square"):
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
        elif config.TARGET_MODE=="Changing_Corners":
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