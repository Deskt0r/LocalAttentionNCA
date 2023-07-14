import numpy as np
import tensorflow as tf
import os
import datetime

from functions_general import *
from functions_data import SamplePool, target_change, get_initial_state, create_distance_mask
from functions_update import pre_update, post_update
from functions_visualization import plot_loss
from functions_network import CAModel
import config 

print(tf.__version__)
print(os.getcwd())

@tf.function
def loss_cell_distance(x, distance_masks,batch_size=config.BATCH_SIZE,
                            energy_budget=10, k=10, return_array=False):
    if len(distance_masks.shape)==4:
        distance_masks=tf.reshape(distance_masks, (8,40,40))
    energy = x[:,:,:,3]
    non_zero = tf.cast(energy != 0, tf.float32)
    max = tf.reduce_max(non_zero*distance_masks, axis=[1,2])
    max = tf.reshape(max, (batch_size,1,1))
    max_pos = tf.cast(tf.math.equal(distance_masks,max), tf.float32)
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

def train_step(x,dist_masks):
        iter_n = tf.random.uniform([], 8, 16, tf.int32)
        max_loss = 0
        max_loss_iteration = 0
        with tf.GradientTape() as g:
            for i in tf.range(iter_n):
                x = pre_update(x)
                x = ca(x)
                x = post_update(x)
            batch_loss = loss_cell_distance(x,dist_masks)
            loss = tf.reduce_mean(batch_loss)
            if loss<max_loss:
                max_loss = loss
                max_loss_iteration = i
        grads = g.gradient(loss, ca.weights)
        grads = [g/(tf.norm(g)+1e-8) for g in grads]
        trainer.apply_gradients(zip(grads, ca.weights))

        tf.print(' - loss',max_loss.numpy(),'in iteration',max_loss_iteration,'of',iter_n)
        return x, loss, batch_loss

if __name__ == "__main__":
    
    current_time = datetime.datetime.now()
    d1 = current_time.strftime("%d_%m_%Y_%H_%M")
    path = str(d1)+'_Logs'
    os.mkdir('../Logs/Dynamic_Positioning/'+ path)
    os.mkdir('../Logs/Dynamic_Positioning/'+ path+'/Batch_Log')
    os.mkdir('../Logs/Dynamic_Positioning/'+ path+'/Loss_Plots')
    
    pool = SamplePool(config.POOL_SIZE, grid_size=config.TARGET_SIZE, nb_chan=config.CHANNEL_N, output_centers='corners', input_centers='center', radius=2, batch_size=config.BATCH_SIZE)
    x0,target_ind,start_ind,dist_masks = pool.sample()
    
    ca = CAModel()
    dummy = tf.zeros((config.BATCH_SIZE,config.TARGET_SIZE,config.TARGET_SIZE,4))
    ca(dummy)
    print(ca.summary())

    loss_log = []
    lr = 1e-3
    print('learning rate is:',lr)
    trainer = tf.keras.optimizers.Adam(lr)

    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=trainer, net=ca)
    manager = tf.train.CheckpointManager(ckpt, '../Logs/Dynamic_Positioning/'+ path +'/TF_Checkpoints', max_to_keep=10)
    k = 0
    ckpt.restore(manager.latest_checkpoint)
    #ckpt.restore('./train_log/tf_ckpts\ckpt-5')

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

        x,target_ind,start_ind,dist_masks = pool.sample() 
        x, loss, batch_loss = train_step(x0,dist_masks)

        step_i = len(loss_log)
        loss_log.append(loss.numpy())

        np.save('../Logs/Dynamic_Positioning/'+ path +'/Batch_Log/%04d_batch_%04d_loss.npy'%(step_i, loss.numpy()) , x.numpy())
        if step_i%25 == 0:
            ckpt.step.assign(int(step_i))
            plot_loss(loss_log,path)
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        print('\r step: %d, loss: %.3f'%(len(loss_log), loss), end='')

        ar = x.numpy()
        ar2 = dist_masks.numpy()
        ar3 = target_ind
        if config.RNG.random()<0.9:
            #we replace the worst sample by the initial state
            batch_loss = batch_loss.numpy()
            ind = np.argsort(batch_loss) # good to bad
            #print('len ind',len(ind))
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