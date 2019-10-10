import tensorflow as tf
from tensorflow.contrib.framework import arg_scope
from tensorflow.contrib import layers
import sys
from os.path import expanduser
home = expanduser("~")
user = home.split('/')[-1]
sys.path.append(home + '/alaska_github/src/')
from tools.jpeg_utils import *

def MLP(features, mode, hidden, *args):
    _inputs = layers.flatten(tf.cast(features, tf.float32, name='input'))
    layer_1 = layers.fully_connected(_inputs, num_outputs=hidden[0],
                              activation_fn=tf.nn.relu, normalizer_fn=None,
                              weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01), 
                              biases_initializer=tf.constant_initializer(0.), scope='layer_1')
    
    layer_2 = layers.fully_connected(layer_1, num_outputs=hidden[1],
                              activation_fn=tf.nn.relu, normalizer_fn=None,
                              weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01), 
                              biases_initializer=tf.constant_initializer(0.), scope='layer_2')
    
    layer_3 = layers.fully_connected(layer_2, num_outputs=5,
                              activation_fn=None, normalizer_fn=None,
                              weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01), 
                              biases_initializer=tf.constant_initializer(0.), scope='layer_3')
    return layer_3



def SR_net_feature_extractor_beast(features, mode, *args):
    features_extracted = []
    for branch in ['YCrCb', 'CrCb', 'Y', 'Cr', 'Cb']:
        _inputs = tf.cast(tf.transpose(features[:,:,:,branch_to_slice(branch)], perm=[0, 3, 1, 2]), tf.float32, name='input')
        f,_ = SR_net_feature_extractor_branch(_inputs, mode, branch+'/')
        features_extracted.append(f)
    with tf.variable_scope('feature_maps'):
        features_extracted = tf.stack(features_extracted)
    return features_extracted

def SR_net_multiclass(features, mode, *args):
    _inputs = tf.cast(tf.transpose(features, perm=[0, 3, 1, 2]), tf.float32, name='input')
    _, ip = SR_net_feature_extractor_branch(_inputs, mode, '')
    return ip

def SR_net_feature_extractor_branch(_inputs, mode, branch):
    data_format = 'NCHW'
    is_training = bool(mode == tf.estimator.ModeKeys.TRAIN)
    with arg_scope([layers.conv2d], num_outputs=16,
                   kernel_size=3, stride=1, padding='SAME',
                   data_format=data_format,
                   activation_fn=None,
                   weights_initializer=layers.variance_scaling_initializer(),
                   weights_regularizer=layers.l2_regularizer(2e-4),
                   biases_initializer=tf.constant_initializer(0.2),
                   biases_regularizer=None),\
        arg_scope([layers.batch_norm],
                   decay=0.9, center=True, scale=True, 
                   updates_collections=tf.GraphKeys.UPDATE_OPS, is_training=is_training,
                   fused=True, data_format=data_format),\
        arg_scope([layers.avg_pool2d],
                   kernel_size=[3,3], stride=[2,2], padding='SAME',
                   data_format=data_format):
        with tf.variable_scope(branch+'Layer1'): # 256*256
            conv=layers.conv2d(_inputs, num_outputs=64, kernel_size=3)
            actv=tf.nn.relu(layers.batch_norm(conv))
        with tf.variable_scope(branch+'Layer2'): # 256*256
            conv=layers.conv2d(actv)
            actv=tf.nn.relu(layers.batch_norm(conv))
        with tf.variable_scope(branch+'Layer3'): # 256*256
            conv1=layers.conv2d(actv)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1)
            bn2=layers.batch_norm(conv2)
            res= tf.add(actv, bn2)
        with tf.variable_scope(branch+'Layer4'): # 256*256
            conv1=layers.conv2d(res)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1)
            bn2=layers.batch_norm(conv2)
            res= tf.add(res, bn2)
        with tf.variable_scope(branch+'Layer5'): # 256*256
            conv1=layers.conv2d(res)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1)
            bn=layers.batch_norm(conv2)
            res= tf.add(res, bn)
        with tf.variable_scope(branch+'Layer6'): # 256*256
            conv1=layers.conv2d(res)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1)
            bn=layers.batch_norm(conv2)
            res= tf.add(res, bn)
        with tf.variable_scope(branch+'Layer7'): # 256*256
            conv1=layers.conv2d(res)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1)
            bn=layers.batch_norm(conv2)
            res= tf.add(res, bn)
        with tf.variable_scope(branch+'Layer8'): # 256*256
            convs = layers.conv2d(res, kernel_size=1, stride=2)
            convs = layers.batch_norm(convs)
            conv1=layers.conv2d(res)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1)
            bn=layers.batch_norm(conv2)
            pool = layers.avg_pool2d(bn)
            res= tf.add(convs, pool)
        with tf.variable_scope(branch+'Layer9'):  # 128*128
            convs = layers.conv2d(res, num_outputs=64, kernel_size=1, stride=2)
            convs = layers.batch_norm(convs)
            conv1=layers.conv2d(res, num_outputs=64)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1, num_outputs=64)
            bn=layers.batch_norm(conv2)
            pool = layers.avg_pool2d(bn)
            res= tf.add(convs, pool)
        with tf.variable_scope(branch+'Layer10'): # 64*64
            convs = layers.conv2d(res, num_outputs=128, kernel_size=1, stride=2)
            convs = layers.batch_norm(convs)
            conv1=layers.conv2d(res, num_outputs=128)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1, num_outputs=128)
            bn=layers.batch_norm(conv2)
            pool = layers.avg_pool2d(bn)
            res= tf.add(convs, pool)
        with tf.variable_scope(branch+'Layer11'): # 32*32
            convs = layers.conv2d(res, num_outputs=256, kernel_size=1, stride=2)
            convs = layers.batch_norm(convs)
            conv1=layers.conv2d(res, num_outputs=256)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1, num_outputs=256)
            bn=layers.batch_norm(conv2)
            pool = layers.avg_pool2d(bn)
            res= tf.add(convs, pool)
        with tf.variable_scope(branch+'Layer12'): # 16*16
            conv1=layers.conv2d(res, num_outputs=512)
            actv1=tf.nn.relu(layers.batch_norm(conv1))
            conv2=layers.conv2d(actv1, num_outputs=512)
            bn=layers.batch_norm(conv2)
            avgp, varp = tf.nn.moments(bn, axes=[2,3])
            minp, maxp = tf.reduce_min(bn, axis=[2,3]), tf.reduce_max(bn, axis=[2,3])
    ip=layers.fully_connected(layers.flatten(avgp), num_outputs=5,
                              activation_fn=None, normalizer_fn=None,
                              weights_initializer=tf.random_normal_initializer(mean=0., stddev=0.01), 
                              biases_initializer=tf.constant_initializer(0.), scope=branch+'ip')
    return tf.squeeze(tf.stack([avgp, varp, minp, maxp])), ip
