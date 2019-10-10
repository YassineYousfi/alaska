import tensorflow as tf
import numpy as np
import sys
import time
from glob import glob
from functools import partial
import os
from os.path import expanduser
home = expanduser("~")
user = home.split('/')[-1]
sys.path.append(home + '/alaska_github/src/')
from tools.jpeg_utils import *
from tools.tf_utils import *
from tools.models import *
from tqdm import tqdm
from scipy import misc, io, fftpack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf

class AdamaxOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adamax algorithm.
    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False, name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


def binary_acc(labels, predictions, weights=None, metrics_collections=None, updates_collections=None, name=None):
    """ Binary accuracy (cover VS stego). Assumes cover is at class 0 and the stego schemes at classes > 0
    Returns accuracy and accuracy update (consistent with the tf.metrics.accuracy functions)
    """
    binary_precitions = tf.to_int32(predictions>0)
    labels_precitions = tf.to_int32(labels>0)
    equal = tf.equal(binary_precitions, labels_precitions)
    binary_acc, binary_acc_op = tf.metrics.mean(equal)
    if metrics_collections:
        ops.add_to_collections(metrics_collections, binary_acc)
    if updates_collections:
        ops.add_to_collections(updates_collections, binary_acc_op)
    return binary_acc, binary_acc_op    
    
def cnn_model_fn(features, labels, mode, params):
    """Model function for Estimators API
    Inputs:
        -features: 
        -labels:
        -mode: a tf.estimator.ModeKeys instance
        -params: dictionary of extra parameters to pass to the funtion 
    """
    
    model = params['model']     
    nclass = params['nclass']  
    hidden = params['MLP_hidden']   
    logits = model(features, mode, hidden)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1, output_type=tf.int32),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    labels = tf.cast(labels, tf.int32)
    oh = tf.one_hot(labels, nclass)
    xen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=oh,logits=logits))  
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n([xen_loss] + reg_losses)
    accuracy = tf.metrics.accuracy(labels, predictions['classes'])
    binary_accuracy = binary_acc(labels, predictions['classes'])
    
    # Configure the warm start dict
    if params['warm_start_checkpoint'] is not None:
        warm_start_dict = dict()
        for i in range(1,13):
            warm_start_dict[params['branch']+'/Layer'+str(i)+'/'] = 'Layer'+str(i)+'/'
        warm_start_dict[params['branch']+'/ip/'] = 'ip/'
        tf.contrib.framework.init_from_checkpoint(params['warm_start_checkpoint'],  warm_start_dict)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = AdamaxOptimizer
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.piecewise_constant(global_step, params['boundaries'], params['values'])  
        tf.summary.scalar('learning_rate', learning_rate)
        optimizer = optimizer(learning_rate) 
        tf.summary.scalar('train_accuracy', accuracy[1]) # output to TensorBoard
        
        # Update batch norm
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    else:
        eval_metric_ops = {"valid_accuracy": (accuracy[0], accuracy[1]), "valid_binary_accuracy": binary_accuracy} 
        
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    
def train_estimator(model, hidden_sizes, num_classes, log_dir, boundaries, values, save_interval,
                    warm_start_checkpoint, load_checkpoint, branch, 
                    COVER_DIR, STEGO_DIR, stego_schemes, priors,
                    IL_train, IL_val, valid_batch_size, 
                    train_batch_size, num_runner_threads,
                    max_iter, gen_train, gen_valid, config=tf.ConfigProto(log_device_placement=True)):
    """
    Train function helper
    DOC WIP
    """
    params = {}
    params['boundaries'] = boundaries
    params['values'] = values
    params['test'] = False
    params['model'] = model
    params['nclass'] = num_classes
    params['warm_start_checkpoint'] = warm_start_checkpoint
    params['branch'] = branch
    params['MLP_hidden'] = hidden_sizes
    
    STEGO_DIRS = []
    for emb in stego_schemes:
        STEGO_DIRS.append(STEGO_DIR[emb])

    # Create the Estimator
    resnet_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir=log_dir,
        params=params,
        config=tf.estimator.RunConfig(save_summary_steps=save_interval,
                                      save_checkpoints_steps=save_interval,
                                      session_config=config,
                                      keep_checkpoint_max=10000), warm_start_from=None)

    if warm_start_checkpoint is not None:
        start = 0
        deleteCheckpointFile(log_dir)
    elif load_checkpoint == 'last' or load_checkpoint is None:
        start = getLatestGlobalStep(log_dir)
    else:
        start = int(load_checkpoint.split('-')[-1])
        updateCheckpointFile(log_dir, load_checkpoint)
    
    print('global step: ', start)
    input_fn_train = partial(input_fn, COVER_DIR, STEGO_DIRS, branch, priors, train_batch_size, IL_train, gen_train, gen_valid, num_runner_threads, True)
    input_fn_val = partial(input_fn, COVER_DIR, STEGO_DIRS, branch, priors, valid_batch_size, IL_val, gen_train, gen_valid, num_runner_threads, False)
    
    train_spec = tf.estimator.TrainSpec(input_fn=input_fn_train, max_steps=max_iter)
    eval_spec = tf.estimator.EvalSpec(input_fn=input_fn_val)
    tf.estimator.train_and_evaluate(resnet_classifier, train_spec, eval_spec)