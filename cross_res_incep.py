# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
'''
This is the resnet structure
'''
import numpy as np
from hyper_parameters import *
BN_EPSILON = 0.001

def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    if is_fc_layer is True:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)
    new_variables = tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer)

    return new_variables

def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True, initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h

def batch_normalization_layer(input_layer, dimension):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
    return bn_layer

def conv_bn_relu_layer(input_layer, filter_shape, stride):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''
    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    bn_layer = batch_normalization_layer(conv_layer, out_channel)
    output = tf.nn.relu(bn_layer)
    return output

def bn_relu_conv_layer(input_layer, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''
    in_channel = input_layer.get_shape().as_list()[-1]
    bn_layer = batch_normalization_layer(input_layer, in_channel)
    relu_layer = tf.nn.relu(bn_layer)
    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer


def dsr_attention(input_batch, in_ch):
    '''
    add dsr attention for the module
    '''
    size = input_batch.get_shape().as_list()[1]
    if size == 32:
        k1 = 7
        st1 = 4
        k2 = (size - k1) // st1 + 1
    if size == 16:
        k1 = 5
        st1 = 2
        k2 = (size - k1) // st1 + 1
    if size == 8:
        k1 = 3
        st1 = 2
        k2 = (size - k1) // st1 + 1

    with tf.variable_scope('squeeze1'):
        sq = bn_depthconv(input_batch, [k1, k1, in_ch, 1], st1)
    with tf.variable_scope('squeeze2'):
        sq = bn_depthconv(sq, [k2, k2, in_ch, 1], 1)
    ex = tf.nn.sigmoid(sq)
    att = tf.multiply(input_batch, ex)
    return att

def bn_depthconv(input_batch, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_batch: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''
    in_channel = input_batch.get_shape().as_list()[-1]
    bn_layer = batch_normalization_layer(input_batch, in_channel)
    #relu_layer = tf.nn.relu(bn_layer)
    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.depthwise_conv2d(bn_layer, filter, strides=[1, stride, stride, 1], padding='VALID')
    return conv_layer


def glpool(input_bn_S, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_bn_S: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''
    #in_channel = input_bn_S.get_shape().as_list()[-1]
    #bn_layer = batch_normalization_layer(input_bn_S, in_channel)
    #relu_layer = tf.nn.relu(bn_layer)
    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.depthwise_conv2d(input_bn_S, filter, strides=[1, stride, stride, 1], padding='VALID')
    return conv_layer

def inception_block(input_layer, output_channel):
    '''
    Defines a inception block in Net
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first inception block of the whole network
    :return: 4D tensor.
    '''
    """
    The left sub-structure
    Three layers, two shortcut connection
    """
    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_1_in_block'):
        conv11 = bn_relu_conv_layer(input_layer, [1, 1, output_channel, output_channel], 1)
    padded_input1 = conv11
    # the second layer
    with tf.variable_scope('conv1_2_in_block'):
        conv12 = bn_relu_conv_layer(conv11, [3, 3, output_channel, output_channel], 1)
    # the firt shorcut connection
    conv12 = input_layer + conv12
    # the third layer
    with tf.variable_scope('conv1_3_in_block'):
        conv13 = bn_relu_conv_layer(conv12, [1, 1, output_channel, output_channel], 1)
    output1 = conv13 + padded_input1

    """
    The Middle sub-structure
    Three layers, two shortcut connection
    """
    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv2_1_in_block'):
        conv21 = bn_relu_conv_layer(input_layer, [1, 1, output_channel, output_channel], 1)
    padded_input2 = conv21
    # the second layer
    with tf.variable_scope('conv2_2_in_block'):
        conv22 = bn_relu_conv_layer(conv21, [5, 5, output_channel, output_channel], 1)
    # the firt shorcut connection
    conv22 = input_layer + conv22
    # the third layer
    with tf.variable_scope('conv2_3_in_block'):
        conv23 = bn_relu_conv_layer(conv22, [1, 1, output_channel, output_channel], 1)
    output2 = conv23 + padded_input2

    """
    The right sub-structure
    Three layers, two shortcut connection
    """
    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv3_1_in_block'):
        conv31 = bn_relu_conv_layer(input_layer, [3, 3, output_channel, output_channel], 1)
    padded_input3 = conv31
    # the second layer
    with tf.variable_scope('conv3_2_in_block'):
        conv32 = bn_relu_conv_layer(conv31, [3, 3, output_channel, output_channel], 1)
    # the firt shorcut connection
    conv32 = input_layer + conv32
    # the third layer
    with tf.variable_scope('conv3_3_in_block'):
        conv33 = bn_relu_conv_layer(conv32, [3, 3, output_channel, output_channel], 1)
    output3 = conv33 + padded_input3

    output = output1 + output2 + output3
    
    return output

def cri_net(input_tensor_batch, n, reuse):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_inception_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''
    layers = []
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 16], 1)
        activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            conv1 = inception_block(layers[-1], 16)
            activation_summary(conv1)
            layers.append(conv1)
    
    with tf.variable_scope('pool1',reuse=reuse):
        pool1 = tf.nn.avg_pool(layers[-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool1 = tf.pad(pool1, [[0, 0], [0, 0], [0, 0], [8, 8]])
        activation_summary(pool1)
        layers.append(pool1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            conv2 = inception_block(layers[-1], 32)
            activation_summary(conv2)
            layers.append(conv2)

    with tf.variable_scope('pool2',reuse=reuse):
        pool2 = tf.nn.avg_pool(layers[-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool2 = tf.pad(pool2, [[0, 0], [0, 0], [0, 0], [16, 16]])
        activation_summary(pool2)
        layers.append(pool2)

    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            conv3 = inception_block(layers[-1], 64)
            activation_summary(conv3)
            layers.append(conv3)

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        global_pool = batch_normalization_layer(layers[-1], in_channel)
        # relu_layer = tf.nn.relu(bn_layer)
        # global_pool =  glpool(global_pool, [8, 8, 256, 1], 1)
        global_pool = tf.reduce_mean(global_pool, [1, 2])
        # assert global_pool.get_shape().as_list()[-1:] == [64]
        output = output_layer(global_pool, 100)
        activation_summary(output)
        layers.append(output)

    # with tf.variable_scope('pool3', reuse=reuse):
    #     pool3 = depthwise_conv(layers[-1], [7, 7, 1024, 1], 1)
    #     # pool3 = tf.nn.depthwise_conv2d(layers[-1], filter=[7, 7, 1024, 1], strides=[1, 1, 1, 1], padding='VALID')
    #     activation_summary(pool3)
    #     layers.append(pool3)

    # 1 x 1 x 1024
    # with tf.variable_scope("reshape", reuse=reuse):
    #     reshape = tf.reshape(layers[-1], [-1, 1024])
    #     activation_summary(reshape)
    #     layers.append(reshape)
    #     #print(reshape.get_shape().as_list())

    # if reuse is False:
    #     with tf.variable_scope('dropout'):
    #         dropout = tf.nn.dropout(layers[-1], 0.8)
    #         activation_summary(dropout)
    #         layers.append(dropout)

    # with tf.variable_scope('logits', reuse=reuse):
    #     logits = output_layer(layers[-1], NUM_CLASS)
    #     activation_summary(logits)
    #     layers.append(logits)



    return layers[-1]

def test_graph(train_dir='logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
    result = inference(input_tensor, 2, reuse=False)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)