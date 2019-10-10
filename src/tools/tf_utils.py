import tensorflow as tf
import sys
import numpy as np
from scipy import misc
import random
import os
from os.path import expanduser
home = expanduser("~")
user = home.split('/')[-1]
sys.path.append(home + '/alaska/src/')
from tools.jpeg_utils import *
import pickle


def getLatestGlobalStep(LOG_DIR):
    # no directory
    if not os.path.exists(LOG_DIR):
        return 0
    checkpoints = [int(f.split('-')[-1].split('.')[0]) \
                   for f in os.listdir(LOG_DIR) if (f.startswith('model.ckpt') and 'temp' not in f)]
    # no checkpoint files
    if not checkpoints:
        return 0
    global_step = max(checkpoints)
    to_file = open(LOG_DIR+'checkpoint', 'w')
    line = 'model_checkpoint_path: "model.ckpt-{}"'.format(global_step)
    to_file.write(line)
    to_file.close()
    return global_step

def updateCheckpointFile(LOG_DIR, checkpoint_name):
    if not os.path.isfile(LOG_DIR + 'checkpoint'):
        return 0
    from_file = open(LOG_DIR+'checkpoint')
    line = from_file.readline()
    from_file.close()
    splits = line.split('"')
    new_line = splits[0] + '"' + checkpoint_name + '"' + splits[-1]
    to_file = open(LOG_DIR + 'checkpoint', mode="w")
    to_file.write(new_line)
    to_file.close()
    
def deleteCheckpointFile(LOG_DIR):
    if os.path.isfile(LOG_DIR + 'checkpoint'):
        os.remove(LOG_DIR + 'checkpoint')
        
def deleteExtraCheckpoints(LOG_DIR, step):
    checkpoint = LOG_DIR + 'model.ckpt-' + str(step)
    os.remove(checkpoint + '.meta')
    os.remove(checkpoint + '.index')
    os.remove(checkpoint + '.data-00000-of-00001')
    
    
def gen_train(COVER_DIR, STEGO_DIRS, im_name, priors, branches):
    channel_slice = branch_to_slice(branches)   
    emb = np.random.choice(len(priors), 1, p=priors)[0]
    STEGO_DIR = STEGO_DIRS[emb]
    try:
        im_name = im_name.decode()
        COVER_DIR = COVER_DIR.decode()
        STEGO_DIR = STEGO_DIR.decode()
    except AttributeError:
        pass
    tmp = jpeglib.jpeg(COVER_DIR+im_name, verbosity=0)
    cover = decompress(tmp)[:,:,channel_slice]
    tmp = jpeglib.jpeg(STEGO_DIR+im_name, verbosity=0)
    stego = decompress(tmp)[:,:,channel_slice]
    batch = np.vstack((cover,stego)).astype(np.float64)
    rot = random.randint(0,3)
    if random.random() < 0.5:
        return [np.rot90(batch, rot, axes=[1,2]), np.array([0,emb+1], dtype='uint8')]
    else:
        return [np.flip(np.rot90(batch, rot, axes=[1,2]), axis=2), np.array([0,emb+1], dtype='uint8')]   
    
def gen_valid(COVER_DIR, STEGO_DIRS, im_name, priors, branches):
    channel_slice = branch_to_slice(branches)    
    emb = np.random.choice(len(priors), 1, p=priors)[0]
    STEGO_DIR = STEGO_DIRS[emb]
    try:
        im_name = im_name.decode()
        COVER_DIR = COVER_DIR.decode()
        STEGO_DIR = STEGO_DIR.decode()
    except AttributeError:
        pass
    tmp = jpeglib.jpeg(COVER_DIR+im_name, verbosity=0)
    cover = decompress(tmp)[:,:,channel_slice]
    tmp = jpeglib.jpeg(STEGO_DIR+im_name, verbosity=0)
    stego = decompress(tmp)[:,:,channel_slice]
    batch = np.vstack((cover,stego)).astype(np.float64)
    return [batch, np.array([0,emb+1], dtype='uint8')]


def gen_feature_maps(COVER_DIR, STEGO_DIRS, im_name, priors, branches=['Y','YCrCb','CrCb']):
    emb = np.random.choice(len(priors), 1, p=priors)[0]
    STEGO_DIR = STEGO_DIRS[emb]
    try:
        im_name = im_name.decode()
        COVER_DIR = COVER_DIR.decode()
        STEGO_DIR = STEGO_DIR.decode()
        branches = [b.decode() for b in branches]
    except AttributeError:
        pass
    with open(COVER_DIR+im_name, 'rb') as handle:
        feature_map_cover = pickle.load(handle)
    with open(STEGO_DIR+im_name, 'rb') as handle:
        feature_map_stego = pickle.load(handle)
    cover = np.concatenate([feature_map_cover[branch].reshape(-1) for branch in branches])
    stego = np.concatenate([feature_map_stego[branch].reshape(-1) for branch in branches])
    batch = np.stack([cover, stego]).astype(np.float64)
    return [batch, np.array([0,emb+1], dtype='uint8')]   
    

def input_fn(COVER_DIR, STEGO_DIRS, branches, priors, batch_size, IL, gen_train, gen_valid, num_of_threads=10, training=False):
    
    nb_data = len(IL)
    if training:
        f = gen_train
        shuffle_buffer_size = nb_data
    else:
        f = gen_valid
    
    _input = f(COVER_DIR, STEGO_DIRS, IL[0], priors, branches)
    
    shapes = [_i.shape for _i in _input]
    features_shape = [batch_size] + [s for s in shapes[0][1:]]
    # add color channel
    # should be of shape (2, height, width, color),
    # because we are using pair constraint
    if len(shapes[0]) < 4:
        features_shape += [1]
    labels_shape = [batch_size] + [s for s in shapes[1][1:]]
    ds = tf.data.Dataset.from_tensor_slices(IL)
    if training:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
        ds = ds.repeat() # infinitely many data
    
    ds = ds.map(lambda filename : tf.py_func(f, [COVER_DIR, STEGO_DIRS, filename, priors, branches], [tf.float64, tf.uint8]), num_parallel_calls=num_of_threads)
    
    ds = ds.batch(batch_size//2) # divide by 2, because we already work with pairs and batch() adds 0-th dimension
    ds = ds.map(lambda x,y: (tf.reshape(x, features_shape), tf.reshape(y, labels_shape)), # reshape number of pairs into batch_size
                num_parallel_calls=num_of_threads).prefetch(buffer_size=num_of_threads*batch_size)
    
    iterator = ds.make_one_shot_iterator()
    
    return iterator.get_next()