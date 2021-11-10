#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Definition of networks models with blocks
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import numpy as np
import tensorflow as tf
import kernels.convolution_ops as conv_ops
from models.genotypes import PRIMITIVES, FUSE
import tensorflow.contrib.slim as slim

OPS = {
  'unary_conv' : lambda layer_ind, inputs, features, radius, fdim, config, training: unary_block(layer_ind, inputs, features, radius, fdim, config, training),
  'KPConv' : lambda layer_ind, inputs, features, radius, fdim, config, training: simple_block(layer_ind, inputs, features, radius, fdim, config, training),
  'KP2Conv' : lambda layer_ind, inputs, features, radius, fdim, config, training: simple_block(layer_ind, inputs, features, radius*2, fdim, config, training),
  'KP4Conv' : lambda layer_ind, inputs, features, radius, fdim, config, training: simple_block(layer_ind, inputs, features, radius*4, fdim, config, training),
  'KPDConv' : lambda layer_ind, inputs, features, radius, fdim, config, training: simple_deformable_block(layer_ind, inputs, features, radius, fdim, config, training),
  'None': lambda layer_ind, inputs, features, radius, fdim, config, training:Zero(layer_ind, inputs, features, radius, fdim, config, training),
  'identity': lambda layer_ind, inputs, features, radius, fdim, config, training:identity(layer_ind, inputs, features, radius, fdim, config, training)

}

FUSES = {
    'None': lambda layer_ind, inputs, features, radius, fdim, config, training:Zero(layer_ind, inputs, features, radius, fdim, config, training),
    'identity': lambda layer_ind, inputs, features, radius, fdim, config, training:identity(layer_ind, inputs, features, radius, fdim, config, training)
}
# ----------------------------------------------------------------------------------------------------------------------
#
#           Utilities
#       \***************/
#

def weight_variable(shape):
    # tf.set_random_seed(42)
    initial = tf.truncated_normal(shape, stddev=np.sqrt(2 / shape[-1]))
    initial = tf.round(initial * tf.constant(1000, dtype=tf.float32)) / tf.constant(1000, dtype=tf.float32)
    return tf.Variable(initial, name='weights')


def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial, name='bias')


def ind_max_pool(x, inds):
    """
    This tensorflow operation compute a maxpooling according to the list of indices 'inds'.
    > x = [n1, d] features matrix
    > inds = [n2, max_num] each row of this tensor is a list of indices of features to be pooled together
    >> output = [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = tf.concat([x, tf.reduce_min(x, axis=0, keep_dims=True)], axis=0)

    # Get features for each pooling cell [n2, max_num, d]
    pool_features = tf.gather(x, inds, axis=0)

    # Pool the maximum
    return tf.reduce_max(pool_features, axis=1)


def closest_pool(x, inds):
    """
    This tensorflow operation compute a pooling according to the list of indices 'inds'.
    > x = [n1, d] features matrix
    > inds = [n2, max_num] We only use the first column of this which should be the closest points too pooled positions
    >> output = [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = tf.concat([x, tf.zeros((1, int(x.shape[1])), x.dtype)], axis=0)

    # Get features for each pooling cell [n2, d]
    pool_features = tf.gather(x, inds[:, 0], axis=0)

    return pool_features


def KPConv(query_points, support_points, neighbors_indices, features, K_values, radius, config):
    """
    Returns the output features of a KPConv
    """

    # Get KP extent from current radius and config density
    extent = config.KP_extent * radius / config.density_parameter

    # Convolution
    return conv_ops.KPConv(query_points,
                           support_points,
                           neighbors_indices,
                           features,
                           K_values,
                           fixed=config.fixed_kernel_points,
                           KP_extent=extent,
                           KP_influence=config.KP_influence,
                           aggregation_mode=config.convolution_mode,)


def KPConv_deformable(query_points, support_points, neighbors_indices, features, K_values, radius, config):
    """
    Returns the output features of a deformable KPConv
    """

    # Get KP extent from current radius and config density
    extent = config.KP_extent * radius / config.density_parameter

    # Convolution
    return conv_ops.KPConv_deformable(query_points,
                                      support_points,
                                      neighbors_indices,
                                      features,
                                      K_values,
                                      fixed=config.fixed_kernel_points,
                                      KP_extent=extent,
                                      KP_influence=config.KP_influence,
                                      aggregation_mode=config.convolution_mode,
                                      modulated=config.modulated)


def KPConv_deformable_v2(query_points, support_points, neighbors_indices, features, K_values, radius, config):
    """
    Perform a simple convolution followed by a batch normalization (or a simple bias) and ReLu
    """

    # Get KP extent from current radius and config density
    extent = config.KP_extent * radius / config.density_parameter

    # Convolution
    return conv_ops.KPConv_deformable_v2(query_points,
                                         support_points,
                                         neighbors_indices,
                                         features,
                                         K_values,
                                         config.num_kernel_points,
                                         fixed=config.fixed_kernel_points,
                                         KP_extent=extent,
                                         KP_influence=config.KP_influence,
                                         mode=config.convolution_mode,
                                         modulated=config.modulated)


def batch_norm(x, use_batch_norm=True, momentum=0.99, training=True):
    """
    This tensorflow operation compute a batch normalization.
    > x = [n1, d] features matrix
    >> output = [n1, d] normalized, scaled, offset features matrix
    """

    if use_batch_norm:
        return tf.layers.batch_normalization(x,
                                             momentum=momentum,
                                             epsilon=1e-6,
                                             training=training)

    else:
        # Just add biases
        beta = tf.Variable(tf.zeros([x.shape[-1]]), name='offset')
        return x + beta


def leaky_relu(features, alpha=0.2):
    return tf.nn.leaky_relu(features, alpha=alpha, name=None)

# ----------------------------------------------------------------------------------------------------------------------
#
#           Convolution blocks
#       \************************/
#


def unary_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a simple 1x1 convolution
    """

    w = weight_variable([int(features.shape[1]), fdim])
    x = conv_ops.unary_convolution(features, w)
    x = leaky_relu(batch_norm(x,
                              config.use_batch_norm,
                              config.batch_norm_momentum,
                              training))

    return x

##########--------search-----------
def MixOp_st1(layer_ind, inputs, features, radius, fdim, config, training, weights, hw):
    index = ['','']
    best_w = np.array([0.0,0.0])
    for primitive in PRIMITIVES:
        idx = np.argsort(best_w)
        ii = PRIMITIVES.index(primitive)
        w = hw[ii]
        if best_w[idx[0]] < w:
            best_w[idx[0]] = w
            index[idx[0]] = primitive
    ops = []
    iii=0
    for ind in index:
        x = OPS[ind](layer_ind, inputs, features, radius, fdim, config, training)
        mask = [i == iii for i in range(len(index))]
        w_mask = tf.constant(mask, tf.bool)
        w = tf.boolean_mask(weights, w_mask)
        ops.append(x * w)
        iii += 1
    return tf.add_n(ops)

def Node_st1(layer_ind, inputs, s0, s1, radius, fdim, config, training, weights, hw, steps=4):
    stats = [s0, s1]
    offset = 0
    for i in range(steps):
        s = sum(MixOp_st1(layer_ind, inputs, h, radius, fdim, config, training, weights[offset + j], hw[offset + j]) for j, h in enumerate(stats))
        offset += len(stats)
        stats.append(s)
    return tf.concat(stats[-3:], axis=-1)

def search_block_st1(layer_ind, inputs, s0, s1, radius, fdim, config, training, hw):
    """
    Block performing a simple 1x1 convolution
    """
    steps = 4
    num_cell = 1
    k = sum(1 for i in range(steps) for n in range(2 + i))

    weights = tf.get_variable("arch_var_weight2h{}_{}".format(steps, layer_ind), [k, 2],
                              initializer=tf.random_normal_initializer(0, 1e-3),
                              regularizer=slim.l2_regularizer(0.0001))
    weights = tf.nn.softmax(weights, axis=-1)

    for cell_i in range(num_cell):
        s0, s1 =s1, Node_st1(layer_ind, inputs, s0, s1, radius, fdim, config, training, weights, hw)

    return s1

def MixOp_st2(layer_ind, inputs, features, radius, fdim, config, training, hw, hw2):
    index = ['','']
    best_w = np.array([0.0,0.0])
    for primitive in PRIMITIVES:
        idx = np.argsort(best_w)
        ii = PRIMITIVES.index(primitive)
        w = hw[ii]
        if best_w[idx[0]] < w:
            best_w[idx[0]] = w
            index[idx[0]] = primitive

    iid = ''
    best_w2 = 0
    for ind in index:
        ii = index.index(ind)
        w = hw2[ii]
        if best_w2<w:
            best_w2=w
            iid=ind
    x = OPS[iid](layer_ind, inputs, features, radius, fdim, config, training)
    return x

def Node_st2(layer_ind, inputs, s0, s1, radius, fdim, config, training, hw, hw2, steps=4):
    stats = [s0, s1]
    offset = 0
    for i in range(steps):
        s = sum(MixOp_st2(layer_ind, inputs, h, radius, fdim, config, training, hw[offset + j], hw2[offset + j]) for j, h in enumerate(stats))
        offset += len(stats)
        stats.append(s)
    return tf.concat(stats[-3:], axis=-1)

def search_block_st2(layer_ind, inputs, s0, s1, radius, fdim, config, training, hw, hw2):
    """
    Block performing a simple 1x1 convolution
    """
    steps = 3
    num_cell = 1

    for cell_i in range(num_cell):
        s0, s1 =s1, Node_st2(layer_ind, inputs, s0, s1, radius, fdim, config, training, hw, hw2, steps)

    return s1

def MixOp_st3(layer_ind, inputs, features, radius, fdim, config, training, hw, hw2):
    index = ['','']
    best_w = np.array([0.0,0.0])
    for primitive in PRIMITIVES:
        idx = np.argsort(best_w)
        ii = PRIMITIVES.index(primitive)
        w = hw[ii]
        if best_w[idx[0]] < w:
            best_w[idx[0]] = w
            index[idx[0]] = primitive

    iid = ''
    best_w2 = 0
    for ind in index:
        ii = index.index(ind)
        w = hw2[ii]
        if best_w2<w:
            best_w2=w
            iid=ind
    x = OPS[iid](layer_ind, inputs, features, radius, fdim, config, training)
    return x

def Node_st3(layer_ind, inputs, s0, s1, radius, fdim, config, training, hw, hw2, cw, prev_feat, steps=4):
    stats = [s0, s1]
    offset = 0
    s = 0
    for i in range(steps):
        for j, h in enumerate(stats):
            t = tf.concat([h, cw[i] * prev_feat], axis=1)
            t = simple_block(layer_ind, inputs, t, radius, fdim, config, training)
            s += MixOp_st3(layer_ind, inputs, t, radius, fdim, config, training, hw[offset + j], hw2[offset + j])
        # s = sum(MixOp_st3(layer_ind, inputs, tf.concat([h, cw[i] * prev_feat], axis=1), radius, fdim, config, training, hw[offset + j], hw2[offset + j]) for j, h in enumerate(stats))
        offset += len(stats)
        stats.append(s)
    return tf.concat(stats[-3:], axis=-1)

def search_block_st3(layer_ind, inputs, s0, s1, radius, fdim, config, training, hw, hw2, prev_feat):
    """
    Block performing a simple 1x1 convolution
    """
    steps = 3
    num_cell = 1
    k = steps

    weights = tf.get_variable("arch_var_cweightsm{}_{}".format(steps, layer_ind), [k, 1],
                              initializer=tf.random_normal_initializer(0, 1e-3),
                              regularizer=slim.l2_regularizer(0.0001))
    weights = tf.nn.softmax(weights, axis=0)

    for cell_i in range(num_cell):
        s0, s1 =s1, Node_st3(layer_ind, inputs, s0, s1, radius, fdim, config, training, hw, hw2, weights, prev_feat, steps)

    return s1

def MixOp_st4(layer_ind, inputs, features, radius, fdim, config, training, hw, hw2):
    index = ['','']
    best_w = np.array([0.0,0.0])
    for primitive in PRIMITIVES:
        idx = np.argsort(best_w)
        ii = PRIMITIVES.index(primitive)
        w = hw[ii]
        if best_w[idx[0]] < w:
            best_w[idx[0]] = w
            index[idx[0]] = primitive

    iid = ''
    best_w2 = 0
    for ind in index:
        ii = index.index(ind)
        w = hw2[ii]
        if best_w2<w:
            best_w2=w
            iid=ind
    x = OPS[iid](layer_ind, inputs, features, radius, fdim, config, training)
    return x

def Node_st4(layer_ind, inputs, s0, s1, radius, fdim, config, training, hw, hw2, h2m, prev_feat, steps=4):
    stats = [s0, s1]
    offset = 0
    idx = np.argmax(h2m)
    s=0
    for i in range(steps):
        for j, h in enumerate(stats):
            t = h
            if j == idx:
                t = tf.concat([h, prev_feat], axis=1)
                t = simple_block(layer_ind, inputs, t, radius, fdim, config, training)
                # s = t * 0 + s
            s += MixOp_st3(layer_ind, inputs, t, radius, fdim, config, training, hw[offset + j], hw2[offset + j])

        offset += len(stats)
        stats.append(s)
    return tf.concat(stats[-3:], axis=-1)

def search_block_st4(layer_ind, inputs, s0, s1, radius, fdim, config, training, hw, hw2, prev_feat, h2m):
    """
    Block performing a simple 1x1 convolution
    """
    steps = 3
    num_cell = 1
    k = steps

    for cell_i in range(num_cell):
        s0, s1 =s1, Node_st4(layer_ind, inputs, s0, s1, radius, fdim, config, training, hw, hw2, h2m, prev_feat, steps)

    return s1
################################
def MixOp_stl2(layer_ind, inputs, features, radius, fdim, config, training, hw):
    index = ''
    best_w = 0.0
    for primitive in PRIMITIVES:
        ii = PRIMITIVES.index(primitive)
        w = hw[ii]
        if best_w < w:
            best_w = w
            index = primitive
    x = OPS[index](layer_ind, inputs, features, radius, fdim, config, training)
    return x

def Node_stl2(layer_ind, inputs, s0, radius, fdim, config, training, hw, cw, prev_feat, steps=4):
    stats = [s0]
    offset = 0
    s = 0
    for i in range(steps):
        for j, h in enumerate(stats):
            t = tf.concat([h, cw[i] * prev_feat], axis=1)
            t = simple_block(layer_ind, inputs, t, radius, fdim, config, training)
            s += MixOp_stl2(layer_ind, inputs, t, radius, fdim, config, training,
                       hw[offset + j])
        # s = sum(MixOp_stl2(layer_ind, inputs, tf.concat([h, cw[i]*prev_feat], axis=1), radius, fdim, config, training, hw[offset + j]) for j, h in enumerate(stats))
        offset += len(stats)
        stats.append(s)
    return tf.concat(stats[-2:], axis=-1)

def search_block_stl2(layer_ind, inputs, s0, radius, fdim, config, training, hw, prev_feat):
    """
    Block performing a simple 1x1 convolution
    """
    steps = 2
    num_cell = 1
    k = steps

    weights = tf.get_variable("arch_var_cweightsl{}_{}".format(steps, layer_ind), [k, 1],
                              initializer=tf.random_normal_initializer(0, 1e-3),
                              regularizer=slim.l2_regularizer(0.0001))
    weights = tf.nn.softmax(weights, axis=0)

    for cell_i in range(num_cell):
        s0 = Node_stl2(layer_ind, inputs, s0, radius, fdim, config, training, hw, weights, prev_feat, steps)

    return s0

def MixOp_stl4(layer_ind, inputs, features, radius, fdim, config, training, hw):
    index = ''
    best_w = 0.0
    for primitive in PRIMITIVES:
        ii = PRIMITIVES.index(primitive)
        w = hw[ii]
        if best_w < w:
            best_w = w
            index = primitive
    x = OPS[index](layer_ind, inputs, features, radius, fdim, config, training)
    return x

def Node_stl4(layer_ind, inputs, s0, radius, fdim, config, training, hw, cw, prev_feat, steps=4):
    stats = [s0]
    offset = 0
    idx = np.argmax(cw)
    s = 0
    for i in range(steps):
        for j, h in enumerate(stats):
            t = h
            if i == idx:
                t = tf.concat([h, prev_feat], axis=1)
                t = simple_block(layer_ind, inputs, t, radius, fdim, config, training)
                # s = t * 0 + s
            s += MixOp_stl2(layer_ind, inputs, t, radius, fdim, config, training, hw[offset + j])

        offset += len(stats)
        stats.append(s)
    return tf.concat(stats[-2:], axis=-1)

def search_block_stl4(layer_ind, inputs, s0, radius, fdim, config, training, hw, prev_feat, m2l):
    """
    Block performing a simple 1x1 convolution
    """
    steps = 2
    num_cell = 1
    k = steps

    for cell_i in range(num_cell):
        s0 = Node_stl4(layer_ind, inputs, s0, radius, fdim, config, training, hw, m2l, prev_feat, steps)

    return s0
##################
def MixOp_stl(layer_ind, inputs, features, radius, fdim, config, training, hw):
    index = ''
    best_w = 0.0
    for primitive in PRIMITIVES:
        ii = PRIMITIVES.index(primitive)
        w = hw[ii]
        if best_w < w:
            best_w = w
            index = primitive
    x = OPS[index](layer_ind, inputs, features, radius, fdim, config, training)
    return x

def Node_stl(layer_ind, inputs, s0, radius, fdim, config, training, hw, steps=4):
    stats = [s0]
    offset = 0
    for i in range(steps):
        s = sum(MixOp_stl(layer_ind, inputs, h, radius, fdim, config, training, hw[offset + j]) for j, h in enumerate(stats))
        offset += len(stats)
        stats.append(s)
    return tf.concat(stats[-2:], axis=-1)

def search_block_stl(layer_ind, inputs, s0, radius, fdim, config, training, hw):
    """
    Block performing a simple 1x1 convolution
    """
    steps = 2
    num_cell = 1

    for cell_i in range(num_cell):
        s0 = Node_stl(layer_ind, inputs, s0, radius, fdim, config, training, hw, steps)

    return s0
###############################

def MixOp(layer_ind, inputs, features, radius, fdim, config, training, weights):
    ops = []
    index = 0
    for primitive in PRIMITIVES:
        x = OPS[primitive](layer_ind, inputs, features, radius, fdim, config, training)
        mask = [i == index for i in range(len(PRIMITIVES))]
        w_mask = tf.constant(mask, tf.bool)
        w = tf.boolean_mask(weights, w_mask)
        ops.append(x * w)
        index += 1
    return tf.add_n(ops)

def Node(layer_ind, inputs, s0, s1, radius, fdim, config, training, weights, steps=4):
    stats = [s0, s1]
    offset = 0
    for i in range(steps):
        s = sum(MixOp(layer_ind, inputs, h, radius, fdim, config, training, weights[offset + j]) for j, h in enumerate(stats))
        offset += len(stats)
        stats.append(s)
    return tf.concat(stats[-3:], axis=-1)

def search_block(layer_ind, inputs, s0, s1, radius, fdim, config, training):
    """
    Block performing a simple 1x1 convolution
    """
    steps = 3
    num_cell = 1
    k = sum(1 for i in range(steps) for n in range(2 + i))
    weights = tf.get_variable("weightm{}_{}".format(steps, layer_ind), [k, len(PRIMITIVES)],
                             initializer=tf.random_normal_initializer(0, 1e-3), regularizer=slim.l2_regularizer(0.0001))
    weights = tf.nn.softmax(weights, axis=-1)

    for cell_i in range(num_cell):
        s0, s1 =s1, Node(layer_ind, inputs, s0, s1, radius, fdim, config, training, weights)

    return s1
###################################
def MixOp_l(layer_ind, inputs, features, radius, fdim, config, training, weights):
    ops = []
    index = 0
    for primitive in PRIMITIVES:
        x = OPS[primitive](layer_ind, inputs, features, radius, fdim, config, training)
        mask = [i == index for i in range(len(PRIMITIVES))]
        w_mask = tf.constant(mask, tf.bool)
        w = tf.boolean_mask(weights, w_mask)
        ops.append(x * w)
        index += 1
    return tf.add_n(ops)

def Node_l(layer_ind, inputs, s0, radius, fdim, config, training, weights, steps=4):
    stats = [s0]
    offset = 0
    for i in range(steps):
        s = sum(MixOp_l(layer_ind, inputs, h, radius, fdim, config, training, weights[offset + j]) for j, h in enumerate(stats))
        offset += len(stats)
        stats.append(s)
    return tf.concat(stats[-2:], axis=-1)

def search_block_l(layer_ind, inputs, s0, radius, fdim, config, training):
    """
    Block performing a simple 1x1 convolution
    """
    steps = 2
    num_cell = 1
    k = sum(1 for i in range(steps) for n in range(1 + i))
    weights = tf.get_variable("arch_var_weightl{}_{}".format(steps, layer_ind), [k, len(PRIMITIVES)],
                             initializer=tf.random_normal_initializer(0, 1e-3), regularizer=slim.l2_regularizer(0.0001))
    weights = tf.nn.softmax(weights, axis=-1)

    for cell_i in range(num_cell):
        s0 =Node_l(layer_ind, inputs, s0, radius, fdim, config, training, weights, steps)

    return s0
###################################
def MixOp_f(layer_ind, inputs, features, radius, fdim, config, training, weights):
    ops = []
    index = 0
    for f in FUSE:
        x = OPS[f](layer_ind, inputs, features, radius, fdim, config, training)
        mask = [i == index for i in range(len(FUSE))]
        w_mask = tf.constant(mask, tf.bool)
        w = tf.boolean_mask(weights, w_mask)
        ops.append(x * w)
        index += 1
    return tf.add_n(ops)

def Node_f(layer_ind, inputs, s0, radius, fdim, config, training, weights, steps=4):
    stats = [s0]
    offset = 0
    for i in range(steps):
        s = sum(MixOp_f(layer_ind, inputs, h, radius, fdim, config, training, weights[offset + j]) for j, h in enumerate(stats))
        offset += len(stats)
        stats.append(s)
    return tf.concat(stats[-1], axis=-1)

def fuse_block(layer_ind, inputs, s0, radius, fdim, config, training):
    """
    Block performing a simple 1x1 convolution
    """
    steps = 1
    num_cell = 1
    k = sum(1 for i in range(steps) for n in range(1 + i))
    weights = tf.get_variable("arch_var_weight{}_{}".format(steps, layer_ind), [k, len(FUSE)],
                             initializer=tf.random_normal_initializer(0, 1e-3), regularizer=slim.l2_regularizer(0.0001))
    weights = tf.nn.softmax(weights, axis=-1)

    for cell_i in range(num_cell):
        s0 =Node_f(layer_ind, inputs, s0, radius, fdim, config, training, weights, steps)

    return s0
###################################

def simple_deformable_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a simple convolution
    """

    # Weights
    w = weight_variable([config.num_kernel_points, int(features.shape[1]), fdim])

    # Convolution
    x = KPConv_deformable(inputs['points'][layer_ind],
               inputs['points'][layer_ind],
               inputs['neighbors'][layer_ind],
               features,
               w,
               radius,
               config)

    x = leaky_relu(batch_norm(x,
                              config.use_batch_norm,
                              config.batch_norm_momentum,
                              training))

    return x

def search_fuse(layer_ind, inputs, features, radius, fdim, config, training):
    ops = []
    weights = tf.get_variable("weight{}_{}".format(1, layer_ind), [len(FUSE)],
                              initializer=tf.random_normal_initializer(0, 1e-3),
                              regularizer=slim.l2_regularizer(0.0001))
    weights = tf.nn.softmax(weights)
    weights = tf.reshape(weights, [-1, 1])
    index = 0
    for fuse in FUSE:
        x = FUSES[fuse](layer_ind, inputs, features, radius, fdim,config, training)
        mask = [i == index for i in range(len(FUSE))]
        w_mask = tf.constant(mask, tf.bool)
        w = tf.boolean_mask(weights, w_mask)
        ops.append(x * w)
        index += 1
    return tf.add_n(ops)


def Zero(layer_ind, inputs, features, radius, fdim, config, training):
    return tf.zeros_like(features)*features

def identity(layer_ind, inputs, features, radius, fdim, config, training):
    return features
##########--------search-----------

def simple_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a simple convolution
    """

    # Weights
    w = weight_variable([config.num_kernel_points, int(features.shape[1]), fdim])

    # Convolution
    x = KPConv(inputs['points'][layer_ind],
               inputs['points'][layer_ind],
               inputs['neighbors'][layer_ind],
               features,
               w,
               radius,
               config)

    x = leaky_relu(batch_norm(x,
                              config.use_batch_norm,
                              config.batch_norm_momentum,
                              training))

    return x


def simple_strided_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a simple strided convolution
    """

    # Weights
    w = weight_variable([config.num_kernel_points, int(features.shape[1]), fdim])

    # Convolution
    x = KPConv(inputs['points'][layer_ind + 1],
               inputs['points'][layer_ind],
               inputs['pools'][layer_ind],
               features,
               w,
               radius,
               config)

    x = leaky_relu(batch_norm(x,
                              config.use_batch_norm,
                              config.batch_norm_momentum,
                              training))

    return x


def resnet_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a resnet double convolution (two convolution vgglike and a shortcut)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([config.num_kernel_points, int(features.shape[1]), fdim])
        x = KPConv(inputs['points'][layer_ind],
                   inputs['points'][layer_ind],
                   inputs['neighbors'][layer_ind],
                   features,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim])
        x = KPConv(inputs['points'][layer_ind],
                   inputs['points'][layer_ind],
                   inputs['neighbors'][layer_ind],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('shortcut'):
        if int(features.shape[1]) != fdim:
            w = weight_variable([int(features.shape[1]), fdim])
            shortcut = conv_ops.unary_convolution(features, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)
        else:
            shortcut = features

    return leaky_relu(x + shortcut)


def resnetb_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a resnet bottleneck convolution (1conv > KPconv > 1conv + shortcut)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([int(features.shape[1]), fdim // 2])
        x = conv_ops.unary_convolution(features, w)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim // 2])
        x = KPConv(inputs['points'][layer_ind],
                   inputs['points'][layer_ind],
                   inputs['neighbors'][layer_ind],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):
        if int(features.shape[1]) != 2 * fdim:
            w = weight_variable([int(features.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(features, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)
        else:
            shortcut = features

    return leaky_relu(x + shortcut)


def resnetb_light_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a resnet bottleneck convolution (1conv > KPconv > 1conv + shortcut)
    """

    with tf.variable_scope('conv1'):
        if int(features.shape[1]) != fdim:
            w = weight_variable([int(features.shape[1]), fdim])
            x = conv_ops.unary_convolution(features, w)
            x = batch_norm(x,
                           config.use_batch_norm,
                           config.batch_norm_momentum,
                           training)
        else:
            x = features

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim])
        x = KPConv(inputs['points'][layer_ind],
                   inputs['points'][layer_ind],
                   inputs['neighbors'][layer_ind],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):
        if int(features.shape[1]) != 2 * fdim:
            w = weight_variable([int(features.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(features, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)
        else:
            shortcut = features

    return leaky_relu(x + shortcut)


def resnetb_deformable_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a resnet bottleneck convolution (1conv > KPconv > 1conv + shortcut)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([int(features.shape[1]), fdim // 2])
        x = conv_ops.unary_convolution(features, w)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim // 2])
        x = KPConv_deformable(inputs['points'][layer_ind],
                              inputs['points'][layer_ind],
                              inputs['neighbors'][layer_ind],
                              x,
                              w,
                              radius,
                              config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):
        if int(features.shape[1]) != 2 * fdim:
            w = weight_variable([int(features.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(features, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)
        else:
            shortcut = features

    return leaky_relu(x + shortcut)


def inception_deformable_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing an inception style convolution combining rigid and deformable KPConv
             (1conv > rigid) \
                               > CONCAT > 1conv + shortcut
    (1conv > rigid > deform) /
    """

    with tf.variable_scope('path1'):

        with tf.variable_scope('unary'):
            w = weight_variable([int(features.shape[1]), fdim // 2])
            x1 = conv_ops.unary_convolution(features, w)
            x1 = leaky_relu(batch_norm(x1,
                                      config.use_batch_norm,
                                      config.batch_norm_momentum,
                                      training))

        with tf.variable_scope('conv'):
            w = weight_variable([config.num_kernel_points, int(x1.shape[1]), fdim // 2])
            x1 = KPConv(inputs['points'][layer_ind],
                        inputs['points'][layer_ind],
                        inputs['neighbors'][layer_ind],
                        x1,
                        w,
                        radius,
                        config)

    with tf.variable_scope('path2'):

        with tf.variable_scope('unary'):
            w = weight_variable([int(features.shape[1]), fdim // 2])
            x2 = conv_ops.unary_convolution(features, w)
            x2 = leaky_relu(batch_norm(x2,
                                       config.use_batch_norm,
                                       config.batch_norm_momentum,
                                       training))

        with tf.variable_scope('conv'):
            w = weight_variable([config.num_kernel_points, int(x2.shape[1]), fdim // 2])
            x2 = KPConv(inputs['points'][layer_ind],
                        inputs['points'][layer_ind],
                        inputs['neighbors'][layer_ind],
                        x2,
                        w,
                        radius,
                        config)

        with tf.variable_scope('conv2_deform'):
            w = weight_variable([config.num_kernel_points, int(x2.shape[1]), fdim // 2])
            x2 = KPConv_deformable_v2(inputs['points'][layer_ind],
                                      inputs['points'][layer_ind],
                                      inputs['neighbors'][layer_ind],
                                      x2,
                                      w,
                                      radius,
                                      config)

    with tf.variable_scope('concat'):
        x = tf.concat([x1, x2], axis=1)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('unary'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):
        if int(features.shape[1]) != 2 * fdim:
            w = weight_variable([int(features.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(features, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)
        else:
            shortcut = features

    return leaky_relu(x + shortcut)


def resnetb_strided_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a strided resnet bottleneck convolution (shortcut is a maxpooling)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([int(features.shape[1]), fdim // 2])
        x = conv_ops.unary_convolution(features, w)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim // 2])
        x = KPConv(inputs['points'][layer_ind + 1],
                   inputs['points'][layer_ind],
                   inputs['pools'][layer_ind],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):

        # Pool shortcuts to strided points TODO: max_pool or closest_pool ?
        shortcut = ind_max_pool(features, inputs['pools'][layer_ind])
        # shortcut = closest_pool(features, neighbors_indices)

        # Regular upsample of the features if not the same dimension
        if int(shortcut.shape[1]) != 2 * fdim:
            w = weight_variable([int(shortcut.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(shortcut, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)

    return leaky_relu(x + shortcut)


def resnetb_light_strided_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a strided resnet bottleneck convolution (shortcut is a maxpooling)
    """

    with tf.variable_scope('conv1'):
        if int(features.shape[1]) != fdim:
            w = weight_variable([int(features.shape[1]), fdim])
            x = conv_ops.unary_convolution(features, w)
            x = batch_norm(x,
                           config.use_batch_norm,
                           config.batch_norm_momentum,
                           training)
        else:
            x = features

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim])
        x = KPConv(inputs['points'][layer_ind + 1],
                   inputs['points'][layer_ind],
                   inputs['pools'][layer_ind],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):

        # Pool shortcuts to strided points TODO: max_pool or closest_pool ?
        shortcut = ind_max_pool(features, inputs['pools'][layer_ind])
        # shortcut = closest_pool(features, neighbors_indices)

        # Regular upsample of the features if not the same dimension
        if int(shortcut.shape[1]) != 2 * fdim:
            w = weight_variable([int(shortcut.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(shortcut, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)

    return leaky_relu(x + shortcut)


def resnetb_deformable_strided_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a strided resnet bottleneck convolution (shortcut is a maxpooling)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([int(features.shape[1]), fdim // 2])
        x = conv_ops.unary_convolution(features, w)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim // 2])
        x = KPConv_deformable(inputs['points'][layer_ind + 1],
                              inputs['points'][layer_ind],
                              inputs['pools'][layer_ind],
                              x,
                              w,
                              radius,
                              config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):

        # Pool shortcuts to strided points TODO: max_pool or closest_pool ?
        shortcut = ind_max_pool(features, inputs['pools'][layer_ind])
        # shortcut = closest_pool(features, neighbors_indices)

        # Regular upsample of the features if not the same dimension
        if int(shortcut.shape[1]) != 2 * fdim:
            w = weight_variable([int(shortcut.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(shortcut, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)

    return leaky_relu(x + shortcut)


def inception_deformable_strided_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing an inception style convolution combining rigid and deformable KPConv
             (1conv > rigid) \
                               > CONCAT > 1conv + shortcut
    (1conv > rigid > deform) /
    """

    with tf.variable_scope('path1'):

        with tf.variable_scope('unary'):
            w = weight_variable([int(features.shape[1]), fdim // 2])
            x1 = conv_ops.unary_convolution(features, w)
            x1 = leaky_relu(batch_norm(x1,
                                      config.use_batch_norm,
                                      config.batch_norm_momentum,
                                      training))

        with tf.variable_scope('conv'):
            w = weight_variable([config.num_kernel_points, int(x1.shape[1]), fdim // 2])
            x1 = KPConv(inputs['points'][layer_ind+1],
                        inputs['points'][layer_ind],
                        inputs['pools'][layer_ind],
                        x1,
                        w,
                        radius,
                        config)

    with tf.variable_scope('path2'):

        with tf.variable_scope('unary'):
            w = weight_variable([int(features.shape[1]), fdim // 2])
            x2 = conv_ops.unary_convolution(features, w)
            x2 = leaky_relu(batch_norm(x2,
                                       config.use_batch_norm,
                                       config.batch_norm_momentum,
                                       training))

        with tf.variable_scope('conv'):
            w = weight_variable([config.num_kernel_points, int(x2.shape[1]), fdim // 2])
            x2 = KPConv(inputs['points'][layer_ind+1],
                        inputs['points'][layer_ind],
                        inputs['pools'][layer_ind],
                        x2,
                        w,
                        radius,
                        config)

        with tf.variable_scope('conv2_deform'):
            w = weight_variable([config.num_kernel_points, int(x2.shape[1]), fdim // 2])
            x2 = KPConv_deformable_v2(inputs['points'][layer_ind+1],
                                      inputs['points'][layer_ind],
                                      inputs['pools'][layer_ind],
                                      x2,
                                      w,
                                      radius,
                                      config)

    with tf.variable_scope('concat'):
        x = tf.concat([x1, x2], axis=1)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('unary'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):

        # Pool shortcuts to strided points TODO: max_pool or closest_pool ?
        shortcut = ind_max_pool(features, inputs['pools'][layer_ind])
        # shortcut = closest_pool(features, neighbors_indices)

        # Regular upsample of the features if not the same dimension
        if int(shortcut.shape[1]) != 2 * fdim:
            w = weight_variable([int(shortcut.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(shortcut, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)

    return leaky_relu(x + shortcut)


def vgg_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing two simple convolutions vgg style
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([config.num_kernel_points, int(features.shape[1]), fdim])
        x = KPConv(inputs['points'][layer_ind],
                   inputs['points'][layer_ind],
                   inputs['neighbors'][layer_ind],
                   features,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim])
        x = KPConv(inputs['points'][layer_ind],
                   inputs['points'][layer_ind],
                   inputs['neighbors'][layer_ind],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    return x


def max_pool_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a max pooling
    """

    with tf.variable_scope('max_pool'):
        pooled_features = ind_max_pool(features, inputs['pools'][layer_ind])

    return pooled_features


def global_average_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a global average over batch pooling
    """

    # Average pooling to aggregate feature in the end
    with tf.variable_scope('average_pooling'):

        # Get the number of features
        N = tf.shape(features)[0]

        # Add a last zero features for shadow batch inds
        features = tf.concat([features, tf.zeros((1, int(features.shape[1])), features.dtype)], axis=0)

        # Collect each batch features
        batch_features1 = tf.gather(features, inputs['out_batches'], axis=0)

        # Average features in each batch
        batch_features = tf.reduce_sum(batch_features1, axis=1)
        #batch_num = tf.reduce_sum(tf.cast(inputs['out_batches'] >= 0, tf.float32), axis=1, keep_dims=True)
        batch_num = tf.reduce_sum(tf.cast(inputs['out_batches'] < N, tf.float32), axis=1, keep_dims=True)

        features = batch_features / batch_num

    return features


def simple_upsample_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a simple upsampling convolution
    """

    # Weights
    w = weight_variable([config.num_kernel_points, int(features.shape[1]), fdim])

    # Convolution
    x = KPConv(inputs['points'][layer_ind - 1],
               inputs['points'][layer_ind],
               inputs['upsamples'][layer_ind - 1],
               features,
               w,
               radius,
               config)

    x = leaky_relu(batch_norm(x,
                              config.use_batch_norm,
                              config.batch_norm_momentum,
                              training))

    return x


def resnetb_upsample_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing an upsampling resnet bottleneck convolution (shortcut is nearest interpolation)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([int(features.shape[1]), fdim // 2])
        x = conv_ops.unary_convolution(features, w)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim // 2])
        x = KPConv(inputs['points'][layer_ind - 1],
                   inputs['points'][layer_ind],
                   inputs['upsamples'][layer_ind - 1],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):

        # Pool shortcuts to strided points (nearest interpolation)
        shortcut = closest_pool(features, inputs['upsamples'][layer_ind - 1])

        # Regular upsample of the features if not the same dimension
        if int(shortcut.shape[1]) != 2 * fdim:
            w = weight_variable([int(shortcut.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(shortcut, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)

    return leaky_relu(x + shortcut)


def nearest_upsample_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing an upsampling by nearest interpolation
    """

    with tf.variable_scope('nearest_upsample'):
        upsampled_features = closest_pool(features, inputs['upsamples'][layer_ind - 1])

    return upsampled_features


def get_block_ops(block_name):
    if block_name == 'search_fuse':
        return search_fuse
    if block_name == 'search':
        return search_block

    if block_name == 'search_st1':
        return search_block_st1

    if block_name == 'search_st2':
        return search_block_st2

    if block_name == 'search_st3':
        return search_block_st3

    if block_name == 'search_st4':
        return search_block_st4

    if block_name == 'search_l':
        return search_block_l

    if block_name == 'fuse':
        return fuse_block

    if block_name == 'search_stl':
        return search_block_stl

    if block_name == 'search_stl2':
        return search_block_stl2

    if block_name == 'search_stl4':
        return search_block_stl4

    if block_name == 'unary':
        return unary_block

    if block_name == 'simple':
        return simple_block

    if block_name == 'simple_strided':
        return simple_strided_block

    elif block_name == 'resnet':
        return resnet_block

    elif block_name == 'resnetb':
        return resnetb_block

    elif block_name == 'resnetb_light':
        return resnetb_light_block

    elif block_name == 'resnetb_deformable':
        return resnetb_deformable_block

    elif block_name == 'inception_deformable':
        return inception_deformable_block()

    elif block_name == 'resnetb_strided':
        return resnetb_strided_block

    elif block_name == 'resnetb_light_strided':
        return resnetb_light_strided_block

    elif block_name == 'resnetb_deformable_strided':
        return resnetb_deformable_strided_block

    elif block_name == 'inception_deformable_strided':
        return inception_deformable_strided_block()

    elif block_name == 'vgg':
        return vgg_block

    elif block_name == 'max_pool' or block_name == 'max_pool_wide':
        return max_pool_block

    elif block_name == 'global_average':
        return global_average_block

    elif block_name == 'nearest_upsample':
        return nearest_upsample_block

    elif block_name == 'simple_upsample':
        return simple_upsample_block

    elif block_name == 'resnetb_upsample':
        return resnetb_upsample_block

    else:
        raise ValueError('Unknown block name in the architecture definition : ' + block_name)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Architectures
#       \*******************/
#


def assemble_CNN_blocks(inputs, config, dropout_prob):
    """
    Definition of all the layers according to config
    :param inputs: dictionary of inputs with keys [points, neighbors, pools, features, batches, labels]
    :param config:
    :param dropout_prob:
    :return:
    """

    # Current radius of convolution and feature dimension
    r = config.first_subsampling_dl * config.density_parameter
    layer = 0
    fdim = config.first_features_dim

    # Input features
    features = inputs['features']
    F = []

    # Boolean of training
    training = dropout_prob < 0.99

    # Loop over consecutive blocks
    block_in_layer = 0
    for block_i, block in enumerate(config.architecture):

        # Detect change to next layer
        if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):

            # Save this layer features
            F += [features]

        # Detect upsampling block to stop
        if 'upsample' in block:
            break

        with tf.variable_scope('layer_{:d}/{:s}_{:d}'.format(layer, block, block_in_layer)):

            # Get the function for this layer
            block_ops = get_block_ops(block)

            # Apply the layer function defining tf ops
            features = block_ops(layer,
                                 inputs,
                                 features,
                                 r,
                                 fdim,
                                 config,
                                 training)

        # Index of block in this layer
        block_in_layer += 1

        # Detect change to a subsampled layer
        if 'pool' in block or 'strided' in block:

            # Update radius and feature dimension for next layer
            layer += 1
            r *= 2
            fdim *= 2
            block_in_layer = 0

        # Save feature vector after global pooling
        if 'global' in block:
            # Save this layer features
            F += [features]

    return F


def assemble_FCNN_blocks(inputs, config, dropout_prob):
    """
    Definition of all the layers according to config
    :param inputs: dictionary of inputs with keys [points, neighbors, pools, upsamples, features, batches, labels]
    :param config:
    :param dropout_prob:
    :return:
    """

    # First get features from CNN
    F = assemble_CNN_blocks(inputs, config, dropout_prob)
    features = F[-1]

    # Current radius of convolution and feature dimension
    layer = config.num_layers - 1
    r = config.first_subsampling_dl * config.density_parameter * 2 ** layer
    fdim = config.first_features_dim * 2 ** layer # if you use resnet, fdim is actually 2 times that

    # Boolean of training
    training = dropout_prob < 0.99

    # Find first upsampling block
    start_i = 0
    for block_i, block in enumerate(config.architecture):
        if 'upsample' in block:
            start_i = block_i
            break

    # Loop over upsampling blocks
    block_in_layer = 0
    for block_i, block in enumerate(config.architecture[start_i:]):

        with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, block, block_in_layer)):

            # Get the function for this layer
            block_ops = get_block_ops(block)

            # Apply the layer function defining tf ops
            features = block_ops(layer,
                                 inputs,
                                 features,
                                 r,
                                 fdim,
                                 config,
                                 training)

        # Index of block in this layer
        block_in_layer += 1

        # Detect change to a subsampled layer
        if 'upsample' in block:

            # Update radius and feature dimension for next layer
            layer -= 1
            r *= 0.5
            fdim = fdim // 2
            block_in_layer = 0

            # Concatenate with CNN feature map
            features = tf.concat((features, F[layer]), axis=1)

    return features


def assemble_SFCNN_blocks(inputs, config, dropout_prob):
    """
    Definition of all the layers according to config
    :param inputs: dictionary of inputs with keys [points, neighbors, pools, upsamples, features, batches, labels]
    :param config:
    :param dropout_prob:
    :return:
    """

    # First get features from CNN
    F = assemble_CNN_blocks(inputs, config, dropout_prob)
    features = F[-1]

    # Current radius of convolution and feature dimension
    layer = config.num_layers - 1
    r = config.first_subsampling_dl * config.density_parameter * 2 ** layer
    fdim = config.first_features_dim * 2 ** layer # if you use resnet, fdim is actually 2 times that

    # Boolean of training
    training = dropout_prob < 0.99
    block_in_layer = 0
    # Search decoder
    #---------------

    with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, 'upsample', block_in_layer)):
        block_ops = get_block_ops('nearest_upsample')

        # Apply the layer function defining tf ops
        features = block_ops(layer,
                             inputs,
                             features,
                             r,
                             fdim,
                             config,
                             training)

    #-------conv4
    layer -= 1
    fdim0 = fdim // 2
    block_in_layer = 0
    features = tf.concat((features, F[layer]), axis=1)
    with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, 'unary', block_in_layer)):
        block_ops = get_block_ops('unary')

        # Apply the layer function defining tf ops
        features = block_ops(layer,
                             inputs,
                             features,
                             r,
                             fdim0,
                             config,
                             training)
    ####---------------
    block_in_layer += 1
    with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, 'upsample', block_in_layer)):
        block_ops = get_block_ops('nearest_upsample')

        # Apply the layer function defining tf ops
        features = block_ops(layer,
                             inputs,
                             features,
                             r,
                             fdim0,
                             config,
                             training)

    block_in_layer += 1
    with tf.variable_scope('unary_{:d}/{:s}_{:d}'.format(layer, 'unary', block_in_layer)):
        block_ops = get_block_ops('unary')

        # Apply the layer function defining tf ops
        features42 = block_ops(layer,
                             inputs,
                             features,
                             r,
                             fdim0//4,
                             config,
                             training)
    with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, 'upsample2', block_in_layer)):
        block_ops = get_block_ops('nearest_upsample')

        # Apply the layer function defining tf ops
        features42 = block_ops(layer-1,
                             inputs,
                             features42,
                             r*0.5*0.5,
                             fdim0//4,
                             config,
                             training)
    with tf.variable_scope('fuse_{:d}/{:s}_{:d}'.format(layer, 'fuse42', block_in_layer)):
        block_ops = get_block_ops('search_fuse')

        # Apply the layer function defining tf ops
        features42 = block_ops(layer-1,
                             inputs,
                             features42,
                             r*0.5*0.5,
                             fdim0//4,
                             config,
                             training)
    ###----------4---->1

    block_in_layer += 1
    with tf.variable_scope('unary_{:d}/{:s}_{:d}'.format(layer, 'unary', block_in_layer)):
        block_ops = get_block_ops('unary')

        # Apply the layer function defining tf ops
        features41 = block_ops(layer,
                             inputs,
                             features,
                             r*0.5*0.5*0.5,
                             fdim0//8,
                             config,
                             training)
    with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, 'upsample2', block_in_layer)):
        block_ops = get_block_ops('nearest_upsample')

        # Apply the layer function defining tf ops
        features41 = block_ops(layer - 1,
                             inputs,
                             features41,
                             r * 0.5*0.5*0.5,
                             fdim0//8,
                             config,
                             training)
    with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, 'upsample4', block_in_layer)):
        block_ops = get_block_ops('nearest_upsample')

        # Apply the layer function defining tf ops
        features41 = block_ops(layer - 2,
                             inputs,
                             features41,
                             r * 0.5*0.5*0.5,
                             fdim0//8,
                             config,
                             training)
    with tf.variable_scope('fuse_{:d}/{:s}_{:d}'.format(layer, 'fuse41', block_in_layer)):
        block_ops = get_block_ops('search_fuse')

        # Apply the layer function defining tf ops
        features41 = block_ops(layer-2,
                             inputs,
                             features41,
                             r*0.5*0.5*0.5,
                             fdim0//8,
                             config,
                             training)
    # -------conv3
    block_in_layer = 0
    layer -= 1
    fdim1 = fdim0 // 2
    r *= 0.5
    features = tf.concat((features, F[layer]), axis=1)
    with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, 'search', block_in_layer)):
        block_ops = get_block_ops('search')

        # Apply the layer function defining tf ops
        features = block_ops(layer,
                             inputs,
                             features,
                             r,
                             fdim1,
                             config,
                             training)

    ####---------------
    block_in_layer += 1
    with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, 'upsample', block_in_layer)):
        block_ops = get_block_ops('nearest_upsample')

        # Apply the layer function defining tf ops
        features = block_ops(layer,
                             inputs,
                             features,
                             r,
                             fdim1,
                             config,
                             training)
    with tf.variable_scope('unary_{:d}/{:s}_{:d}'.format(layer, 'unary', block_in_layer)):
        block_ops = get_block_ops('unary')

        # Apply the layer function defining tf ops
        features31 = block_ops(layer,
                             inputs,
                             features,
                             r,
                             fdim1//4,
                             config,
                             training)
    with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, 'upsample2', block_in_layer)):
        block_ops = get_block_ops('nearest_upsample')

        # Apply the layer function defining tf ops
        features31 = block_ops(layer-1,
                             inputs,
                             features31,
                             r*0.5*0.5,
                             fdim1//4,
                             config,
                             training)
    with tf.variable_scope('fuse_{:d}/{:s}_{:d}'.format(layer, 'fuse31', block_in_layer)):
        block_ops = get_block_ops('search_fuse')

        # Apply the layer function defining tf ops
        features31 = block_ops(layer-1,
                             inputs,
                             features31,
                             r*0.5*0.5,
                             fdim1//4,
                             config,
                             training)
    #####search fusion############

    features = tf.concat((features, features42), axis=1)

    # -------conv2
    layer -= 1
    fdim2 = fdim1 // 2
    r *= 0.5
    block_in_layer = 0
    features = tf.concat((features, F[layer]), axis=1)
    with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, 'search', block_in_layer)):
        block_ops = get_block_ops('search')

        # Apply the layer function defining tf ops
        features = block_ops(layer,
                             inputs,
                             features,
                             r,
                             fdim2,
                             config,
                             training)
    ####---------------

    block_in_layer += 1
    with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, 'upsample', block_in_layer)):
        block_ops = get_block_ops('nearest_upsample')

        # Apply the layer function defining tf ops
        features = block_ops(layer,
                             inputs,
                             features,
                             r,
                             fdim2,
                             config,
                             training)
    ####search fuse#####
    features = tf.concat((features, features41, features31), axis=1)
    # -------conv1
    layer -= 1
    fdim3 = fdim2 // 2
    r *= 0.5
    block_in_layer = 0
    features = tf.concat((features, F[layer]), axis=1)
    with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, 'search', block_in_layer)):
        block_ops = get_block_ops('search')

        # Apply the layer function defining tf ops
        features = block_ops(layer,
                             inputs,
                             features,
                             r,
                             fdim3,
                             config,
                             training)

    return features

def assemble_SHCNN_blocks(inputs, config, dropout_prob, hw, mw, lw, hw2, mw2, lw2, h2m, m2l):
    """
    Definition of all the layers according to config
    :param inputs: dictionary of inputs with keys [points, neighbors, pools, upsamples, features, batches, labels]
    :param config:
    :param dropout_prob:
    :return:
    """

    # First get features from CNN
    F = assemble_CNN_blocks(inputs, config, dropout_prob)
    # features = F[-1]

    # Current radius of convolution and feature dimension
    layer = config.num_layers - 1
    r = config.first_subsampling_dl * config.density_parameter * 2 ** layer
    fdim = config.first_features_dim * 2 ** layer # if you use resnet, fdim is actually 2 times that

    # Boolean of training
    training = dropout_prob < 0.99
    block_in_layer = 0

    fdim_h = F[layer-1].shape[1].value
    with tf.variable_scope('reduce_{:d}/{:s}_{:d}'.format(layer, 'unary', block_in_layer)):
        block_ops = get_block_ops('unary')

        # Apply the layer function defining tf ops
        features4 = block_ops(layer,
                             inputs,
                             F[layer],
                             r,
                             fdim_h,
                             config,
                             training)
    block_in_layer += 1
    with tf.variable_scope('down_{:d}/{:s}_{:d}'.format(layer-1, 'down', block_in_layer)):
        block_ops = get_block_ops('max_pool')

        # Apply the layer function defining tf ops
        features3 = block_ops(layer-1,
                              inputs,
                              F[layer-1],
                              r,
                              fdim_h,
                              config,
                              training)

    fdim_m = F[layer-3].shape[1].value
    with tf.variable_scope('reduce_{:d}/{:s}_{:d}'.format(layer-2, 'unary', block_in_layer)):
        block_ops = get_block_ops('unary')

        # Apply the layer function defining tf ops
        features2 = block_ops(layer-2,
                              inputs,
                              F[layer-2],
                              r,
                              fdim_m,
                              config,
                              training)
    block_in_layer += 1
    with tf.variable_scope('down_{:d}/{:s}_{:d}'.format(layer - 1, 'down', block_in_layer)):
        block_ops = get_block_ops('max_pool')

        # Apply the layer function defining tf ops
        features1 = block_ops(layer - 3,
                              inputs,
                              F[layer-3],
                              r,
                              fdim_m,
                              config,
                              training)
    block_in_layer += 1
    with tf.variable_scope('searchst1_{:d}/{:s}_{:d}'.format(layer, 'block', block_in_layer)):
        block_ops = get_block_ops('search_st2')

        # Apply the layer function defining tf ops
        features43 = block_ops(layer,
                              inputs,
                              features4,
                              features3,
                              r,
                              fdim_h,
                              config,
                              training, hw, hw2)
    block_in_layer += 1
    with tf.variable_scope('reduce_{:d}/{:s}_{:d}'.format(layer, 'unary', block_in_layer)):
        block_ops = get_block_ops('unary')

        # Apply the layer function defining tf ops
        features43 = block_ops(layer,
                              inputs,
                              features43,
                              r,
                              128,
                              config,
                              training)

    with tf.variable_scope('up_{:d}/{:s}_{:d}'.format(layer, 'up43', block_in_layer)):
        block_ops = get_block_ops('nearest_upsample')

        # Apply the layer function defining tf ops
        features43 = block_ops(layer,
                              inputs,
                              features43,
                              r,
                              128,
                              config,
                              training)

    with tf.variable_scope('up_{:d}/{:s}_{:d}'.format(layer - 1, 'up32', block_in_layer)):
        block_ops = get_block_ops('nearest_upsample')

        # Apply the layer function defining tf ops
        features432 = block_ops(layer-1,
                              inputs,
                              features43,
                              r,
                              128,
                              config,
                              training)
    with tf.variable_scope('up_{:d}/{:s}_{:d}'.format(layer - 2, 'up21', block_in_layer)):
        block_ops = get_block_ops('nearest_upsample')

        # Apply the layer function defining tf ops
        features43 = block_ops(layer-2,
                              inputs,
                              features432,
                              r,
                              128,
                              config,
                              training)
    with tf.variable_scope('up_{:d}/{:s}_{:d}'.format(layer - 3, 'up10', block_in_layer)):
        block_ops = get_block_ops('nearest_upsample')

        # Apply the layer function defining tf ops
        features43 = block_ops(layer-3,
                              inputs,
                              features43,
                              r,
                              128,
                              config,
                              training)

    # features21 = tf.concat([features2, features1], axis=1)
    block_in_layer += 1
    with tf.variable_scope('searchst_{:d}/{:s}_{:d}'.format(layer-2, 'block', block_in_layer)):
        block_ops = get_block_ops('search_st4')

        # Apply the layer function defining tf ops
        features21 = block_ops(layer - 2,
                              inputs,
                              features2,
                              features1,
                              r * 0.25,
                              fdim_m,
                              config,
                              training, mw, mw2, features432, h2m)
    block_in_layer += 1
    with tf.variable_scope('reduce_{:d}/{:s}_{:d}'.format(layer - 2, 'unary', block_in_layer)):
        block_ops = get_block_ops('unary')

        # Apply the layer function defining tf ops
        features21 = block_ops(layer - 2,
                              inputs,
                              features21,
                              r * 0.25,
                              128,
                              config,
                              training)

    with tf.variable_scope('up_{:d}/{:s}_{:d}'.format(layer - 2, 'up21', block_in_layer)):
        block_ops = get_block_ops('nearest_upsample')

        # Apply the layer function defining tf ops
        features21 = block_ops(layer-2,
                              inputs,
                              features21,
                              r * 0.25,
                              128,
                              config,
                              training)
    with tf.variable_scope('up_{:d}/{:s}_{:d}'.format(layer - 2, 'up10', block_in_layer)):
        block_ops = get_block_ops('nearest_upsample')

        # Apply the layer function defining tf ops
        features21 = block_ops(layer-3,
                              inputs,
                              features21,
                              r * 0.25,
                              128,
                              config,
                              training)

    block_in_layer += 1
    with tf.variable_scope('reduce_{:d}/{:s}_{:d}'.format(layer - 4, 'unary', block_in_layer)):
        block_ops = get_block_ops('unary')

        # Apply the layer function defining tf ops
        features11 = block_ops(layer - 4,
                               inputs,
                               F[layer - 4],
                               r * 0.25 * 0.25,
                               16,
                               config,
                               training)

    block_in_layer += 1
    with tf.variable_scope('search_{:d}/{:s}_{:d}'.format(layer - 4, 'low_level', block_in_layer)):
        block_ops = get_block_ops('search_stl4')

        # Apply the layer function defining tf ops
        features11 = block_ops(layer - 4,
                               inputs,
                               features11,
                               r * 0.25 * 0.25,
                               16,
                               config,
                               training, lw2, features11, m2l)

    features = tf.concat([features11, features21, features43], axis=1)

    return features


def classification_head(features, config, dropout_prob):

    # Boolean of training
    training = dropout_prob < 0.99

    # Fully connected layer
    with tf.variable_scope('fc'):
        w = weight_variable([int(features.shape[1]), 1024])
        features = leaky_relu(batch_norm(tf.matmul(features, w),
                                         config.use_batch_norm,
                                         config.batch_norm_momentum,
                                         training))

    # Dropout
    with tf.variable_scope('dropout'):
        features = tf.nn.dropout(features, dropout_prob)

    # Softmax
    with tf.variable_scope('softmax'):
        w = weight_variable([1024, config.num_classes])
        b = bias_variable([config.num_classes])
        logits = tf.matmul(features, w) + b

    return logits


def segmentation_head(features, config, dropout_prob):

    # Boolean of training
    training = dropout_prob < 0.99

    # Unary conv (equivalent to fully connected for each pixel)
    with tf.variable_scope('head_unary_conv'):
        w = weight_variable([int(features.shape[1]), config.first_features_dim])
        features = conv_ops.unary_convolution(features, w)
        features = leaky_relu(batch_norm(features,
                                         config.use_batch_norm,
                                         config.batch_norm_momentum,
                                         training))

    # Softmax
    with tf.variable_scope('softmax'):
        w = weight_variable([config.first_features_dim, config.num_classes])
        b = bias_variable([config.num_classes])
        logits = conv_ops.unary_convolution(features, w) + b

    return logits

def mid_segmentation_head(inputs, features, config, dropout_prob):

    # Boolean of training
    training = dropout_prob < 0.99

    seg_logits = []
    for i in range(len(features)):
        # Unary conv (equivalent to fully connected for each pixel)
        with tf.variable_scope('head_unary_conv_{:}'.format(i)):
            w = weight_variable([int(features[i].shape[1]), config.first_features_dim])
            feature = conv_ops.unary_convolution(features[i], w)
            feature = leaky_relu(batch_norm(feature,
                                             config.use_batch_norm,
                                             config.batch_norm_momentum,
                                             training))

        # Softmax
        with tf.variable_scope('softmax_{:}'.format(i)):
            w = weight_variable([config.first_features_dim, config.num_classes])
            b = bias_variable([config.num_classes])
            logits = conv_ops.unary_convolution(feature, w) + b

        if i >0 :
            r = config.first_subsampling_dl * config.density_parameter * 2 ** (i*2)
            for j in range(i*2):
                with tf.variable_scope('up_sample_{:}'.format(i)):
                    block_ops = get_block_ops('nearest_upsample')

                    # Apply the layer function defining tf ops
                    logits = block_ops(j+1,
                                       inputs,
                                       logits,
                                       r*0.5**i,
                                       1,
                                       config,
                                       training)

        seg_logits+=[logits]
    return seg_logits[0], seg_logits[1], seg_logits[2]

def multi_segmentation_head(features, object_labels, config, dropout_prob):

    # Boolean of training
    training = dropout_prob < 0.99

    with tf.variable_scope('head_unary_conv'):

        # Get a feature for each point and for each object_class
        nC = len(config.num_classes)
        w = weight_variable([int(features.shape[1]), nC * config.first_features_dim])
        features = conv_ops.unary_convolution(features, w)
        features = leaky_relu(batch_norm(features,
                                         config.use_batch_norm,
                                         config.batch_norm_momentum,
                                         training))
        features = tf.reshape(features, [-1, nC, config.first_features_dim])
        features = tf.transpose(features, [1, 0, 2])

    with tf.variable_scope('softmax'):

        # Get a logit for each point and for each object_class
        maxC = np.max(config.num_classes)
        w = weight_variable([nC, config.first_features_dim, maxC])
        b = bias_variable([maxC])
        all_logits = conv_ops.unary_convolution(features, w) + b

        # Pool according to real object class
        nd_inds = tf.stack((object_labels, tf.range(tf.shape(object_labels)[0])))
        nd_inds = tf.transpose(nd_inds)
        logits = tf.gather_nd(all_logits, nd_inds)

    return logits


def classification_loss(logits, inputs):

    # Exclusive Labels cross entropy on each point
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs['labels'],
                                                                   logits=logits,
                                                                   name='xentropy')

    # Mean on batch
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def segmentation_loss(logits, inputs, batch_average=False):

    # Exclusive Labels cross entropy on each point
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs['point_labels'],
                                                                   logits=logits,
                                                                   name='xentropy')

    if not batch_average:
        # Option 1 : Mean on all points of all batch
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')

    else:
        # Option 2 : First mean on each batch, then mean (correspond to weighted sum with batch proportions)
        stacked_weights = inputs['batch_weights']
        return tf.reduce_mean(stacked_weights * cross_entropy, name='xentropy_mean')


def multi_segmentation_loss(logits, inputs, batch_average=False):

    # Exclusive Labels cross entropy
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=inputs['point_labels'],
                                                                   logits=logits,
                                                                   name='xentropy')

    if not batch_average:
        # Option 1 : Mean on all points of all batch
        return tf.reduce_mean(cross_entropy, name='xentropy_mean')

    else:
        # Option 2 : First mean on each batch, then mean (correspond to weighted sum with batch proportions)
        stacked_weights = inputs['batch_weights']
        return tf.reduce_mean(stacked_weights * cross_entropy, name='xentropy_mean')