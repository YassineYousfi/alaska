from tensorflow.graph_util import convert_variables_to_constants as freeze_graph
import tensorflow as tf
import numpy as np

def freeze_graph_function(checkpoint_path, frozen_path, output_name):
    """
    freezes checkpoint of a model into a TF frozen graph file
    Inputs:
        -checkpoint_path
        -frozen_path
        -output_name='feature_maps/stack'
    """
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(checkpoint_path + '.meta', clear_devices=True)
    with tf.Session() as sess:
        saver.restore(sess, checkpoint_path)
        graph_def = freeze_graph(sess, sess.graph_def, [output_name])
        with tf.gfile.GFile(frozen_path, "wb") as f:
            f.write(graph_def.SerializeToString())
        
        
def get_tensors_in_checkpoint(checkpoint_path):
    """
    Returns variables and their values in a TF checkpoint file
    Inputs:
        -checkpoint_path
    """
    varlist=[]
    var_value =[]
    reader = tf.pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in sorted(var_to_shape_map):
        varlist.append(key)
        var_value.append(reader.get_tensor(key))
    return (varlist, var_value)
    
    
def load_graph(frozen_path):
    with tf.gfile.GFile(frozen_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def group_branches(model_class, branches, branches_ckpt_paths_dict, grouped_ckpt_path):
    """
    Merges multiple checkpoints in a single multi-branch graph and saves it
    Inputs:
        -multibranch_model
        -branches
        -branch_prefix_path
        -output_path
    """
    model_class = model_class
    tf.reset_default_graph()
    input_image = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='input')
    global_step = tf.get_variable('global_step', dtype=tf.int32, shape=[], initializer=tf.constant_initializer(0), trainable=False)
    featuremaps = model_class(input_image,  tf.estimator.ModeKeys.PREDICT)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    savers = dict()
    for branch in branches:
        reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=branch+'?')
        # {'Layer1': YCrCb/Layer1}
        dict_vars = {('/'.join(v.name.encode('utf-8').split('/')[1:])).split(':')[0]: v for v in reuse_vars} 
        savers[branch] = tf.train.Saver(var_list=dict_vars, max_to_keep=1000)
    saver_all = tf.train.Saver(max_to_keep=1000)
    
    with tf.Session() as sess:
        sess.run(init_op)
        for branch in branches:
            savers[branch].restore(sess, branches_ckpt_paths_dict[branch])
        f = sess.run(featuremaps, feed_dict = {input_image: np.ones((1,256,256,3))}) 
        saver_all.save(sess, grouped_ckpt_path)