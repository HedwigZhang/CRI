"""
Coder: Xingpeng Zhang
This file is to realize our new SGRISE module, which combining channel shufffle, grouping convoulution, 
residual inception and SE module
"""
#validation_step
import numpy as np
import tensorflow as tf
from hyperParameters import *
BN_EPSILON = 0.001
NUM_CLASS = 100

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

def output_layer(input_out_layer, num_labels):
    '''
    :param input_out_layer: 2D tensor, the input
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_out_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True, initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())
    fc_h = tf.matmul(input_out_layer, fc_w) + fc_b
    return fc_h

def batch_normalization_layer(input_batch, dimension):
    '''
    Helper function to do batch normalziation
    :param input_batch: 4D tensor
    :param dimension: input_batch.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    '''
    mean, variance = tf.nn.moments(input_batch, axes=[0, 1, 2])
    beta = tf.get_variable('beta', dimension, tf.float32, initializer=tf.constant_initializer(0.0, tf.float32))
    gamma = tf.get_variable('gamma', dimension, tf.float32, initializer=tf.constant_initializer(1.0, tf.float32))
    bn_layer = tf.nn.batch_normalization(input_batch, mean, variance, beta, gamma, BN_EPSILON)
    return bn_layer

def bn_relu_conv_layer_S(input_bn_S, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_bn_S: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''
    in_channel = input_bn_S.get_shape().as_list()[-1]
    bn_layer = batch_normalization_layer(input_bn_S, in_channel)
    relu_layer = tf.nn.relu(bn_layer)
    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer

def bn_relu_conv_layer_V(input_bn_V, filter_shape, stride):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_bn_V: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''
    in_channel = input_bn_V.get_shape().as_list()[-1]
    bn_layer = batch_normalization_layer(input_bn_V, in_channel)
    relu_layer = tf.nn.relu(bn_layer)
    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='VALID')
    return conv_layer

def channel_shuffle_layer(input_shuffle, num_groups):
    '''
    A helper function to realize channel shuffle
    :param input_shuffle: 4D tensor
    :param num_groups: 1D number, which denote the channel is shuffled into several group. 
    To relaize channle shuffle, matrix tranpose is the core
    :return: 4D tensor.
    '''
    num_input, height, width, channel = input_shuffle.shape.as_list()
    x_reshaped = tf.reshape(input_shuffle, [num_input, height, width, num_groups, channel // num_groups])
    # matrix tranpose
    x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
    output = tf.reshape(x_transposed, [num_input, height, width, channel])
    return output

# def dsr_attention(input_batch, in_ch):
#     '''
#     add dsr attention for the module
#     '''
#     size = input_batch.get_shape().as_list()[1]
#     if size == 28:
#         k1 = 7
#         st1 = 4
#         k2 = (size - k1) // st1 + 1
#     if size == 14:
#         k1 = 5
#         st1 = 2
#         k2 = (size - k1) // st1 + 1
#     if size == 7:
#         k1 = 3
#         st1 = 2
#         k2 = (size - k1) // st1 + 1

#     with tf.variable_scope('squeeze1'):
#         sq = bn_depthconv(input_batch, [k1, k1, in_ch, 1], st1)
#     with tf.variable_scope('squeeze2'):
#         sq = bn_depthconv(sq, [k2, k2, in_ch, 1], 1)
#     ex = tf.nn.sigmoid(sq)
#     att = tf.multiply(input_batch, ex)
#     return att

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


def bn_depthconv_S(input_batch, filter_shape, stride):
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
    conv_layer = tf.nn.depthwise_conv2d(bn_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer

def SGRISE(input_layer, branch1_size, branch2_size, branch3_size, branch4_size):
    """
    The detail of our designed SGRISE mudule
    :param input_layer: 4D tensor
    :param branch1_size: 1D number, the number of channels that pass through the first sub-network of SGRISE mudule
    :param branch2_size: 1D number, the number of channels that pass through the second sub-network of SGRISE mudule
    :param branch3_size: 1D number, the number of channels that pass through the third sub-network of SGRISE mudule
    :param branch4_size: 1D number, the number of channels that pass through the fourth sub-network of SGRISE mudule
    :return 4D tensor
    """
    # first, channel channel
    num_groups = 16
    #reduction_ratio = 4
    input_layer = channel_shuffle_layer(input_layer, num_groups)
    in_ch = input_layer.get_shape().as_list()[-1]
    '''
    The first sub-network
    three layers, two shortcut connections
    '''
    # get the input for the first sub-network based on the branch1_size
    input_1 = input_layer[:,:,:, 0 : branch1_size]
    # in1_ch = input_1.get_shape().as_list()[-1]
    # print(in1_ch)
    # the first conv layers in the first sub-network
    with tf.variable_scope('conv_1_1_block'):
        conv11 = bn_relu_conv_layer_S(input_1, [1, 1, branch1_size, branch1_size], 1)
    # keep as residual
    padded_input1 = conv11
    # the second layer in the first sub-structure
    with tf.variable_scope('conv_1_2_block'):
        conv12 = bn_relu_conv_layer_S(conv11, [3, 3, branch1_size, branch1_size], 1)
    # the firt shorcut connection in the first sub-structure
    conv12 = input_1 + conv12
    # the third layer in the first sub-structure
    with tf.variable_scope('conv_1_3_block'):
        conv13 = bn_relu_conv_layer_S(conv12, [1, 1, branch1_size, branch1_size], 1)
    # the firt shorcut connection in the first sub-structure
    output1 = conv13 + padded_input1
    ####### DSR module
    # with tf.variable_scope('dsr_left'):
    #     output1 = dsr_attention(output1, branch1_size)

    # the SE block in the first sub-structure
    # to calculate the maximum value of the element on every dimension of a tensor. Channel
    # with tf.variable_scope("GlobalAvgPool_1") :
    #     squeeze_1 = tf.reduce_mean(input_1, [1, 2])

    # # The first fully_connexted layers,
    # with tf.variable_scope("fully_connected_11"):
    #     excitation_1 = tf.layers.dense(inputs=squeeze_1, use_bias=True, units=branch1_size // reduction_ratio)
    #     excitation_1 = tf.nn.relu(excitation_1)

    # # the second fully_connexted layers,
    # with tf.variable_scope("fully_connected_12"):
    #     excitation_1 = tf.layers.dense(inputs=excitation_1, use_bias=True, units=branch1_size)
    #     excitation_1 = tf.nn.sigmoid(excitation_1)

    # # reshape the vector to a 4D densor
    # excitation_1 = tf.reshape(excitation_1, [-1, 1, 1, branch1_size])
    # ## In fact, this SE structure is to choose channel features, enhance useful channel 
    # # and restrain unuseful channel
    # output1 = output1 * excitation_1

    '''
    The second sub-network
    three layers, two shortcut connections
    '''
    # get the input for the second sub-network based on the branch2_size
    input_2 = input_layer[:,:,:, branch1_size : branch1_size + branch2_size]
    # in2_ch = input_2.get_shape().as_list()[-1]
    # print(in2_ch)
    # the first conv layers in the first sub-network
    with tf.variable_scope('conv_2_1_block'):
        conv21 = bn_relu_conv_layer_S(input_2, [1, 1, branch2_size, branch2_size], 1)
    # keep as residual
    padded_input2 = conv21
    # the second layer in the second sub-structure
    with tf.variable_scope('conv_2_2_block'):
        conv22 = bn_relu_conv_layer_S(conv21, [5, 5, branch2_size, branch2_size], 1)
    # the firt shorcut connection in the second sub-structure
    conv22 = input_2 + conv21
    # the third layer in the second sub-structure
    with tf.variable_scope('conv_2_3_block'):
        conv23 = bn_relu_conv_layer_S(conv22, [1, 1, branch2_size, branch2_size], 1)
    # the firt shorcut connection in the second sub-structure
    output2 = conv23 + padded_input2
    # with tf.variable_scope('dsr_middle'):
    #     output2 = dsr_attention(output2, branch2_size)
    # #the SE block in the second sub-structure
    # # to calculate the maximum value of the element on every dimension of a tensor. Channel
    # with tf.variable_scope("GlobalAvgPool_2") :
    #     squeeze_2 = tf.reduce_mean(input_2, [1, 2])

    # # The first fully_connexted layers in the second sub-structure
    # with tf.variable_scope("fully_connected_21"):
    #     excitation_2 = tf.layers.dense(inputs=squeeze_2, use_bias=True, units=branch2_size // reduction_ratio)
    #     excitation_2 = tf.nn.relu(excitation_2)

    # # the second fully_connexted layers in the second sub-structure
    # with tf.variable_scope("fully_connected_22"):
    #     excitation_2 = tf.layers.dense(inputs=excitation_2, use_bias=True, units=branch2_size)
    #     excitation_2 = tf.nn.sigmoid(excitation_2)

    # # reshape the vector to a 4D densor
    # excitation_2 = tf.reshape(excitation_2, [-1, 1, 1, branch2_size])
    # ## In fact, this SE structure is to choose channel features, enhance useful channel 
    # # and restrain unuseful channel
    # output2 = output2 * excitation_2

    '''
    The third sub-structure
    Three layers, two shortcut connection
    '''
    # get the input for the second sub-network based on the branch3_size
    input_3 = input_layer[:,:,:, branch1_size + branch2_size : branch1_size + branch2_size + branch3_size]
    # in3_ch = input_3.get_shape().as_list()[-1]
    # print(in3_ch)
    with tf.variable_scope('conv_3_1_block'):
        conv31 = bn_relu_conv_layer_S(input_3, [3, 3, branch3_size, branch3_size], 1)
    padded_input3 = conv31
    # the second layer in the third sub-structure
    with tf.variable_scope('conv_3_2_block'):
        conv32 = bn_relu_conv_layer_S(conv31, [3, 3, branch3_size, branch3_size], 1)
    # the firt shorcut connection in the third sub-structure
    conv32 = input_3 + conv32
    # the third layer in the third sub-structure
    with tf.variable_scope('conv_3_3_block'):
        conv33 = bn_relu_conv_layer_S(conv32, [3, 3, branch3_size, branch3_size], 1)
    # the second shorcut connection in the third sub-structure
    output3 = conv33 + padded_input3
    # with tf.variable_scope('dsr_right'):
    #     output3 = dsr_attention(output3, branch3_size)

    # #the SE block in the third sub-structure
    # # to calculate the maximum value of the element on every dimension of a tensor. Channel
    # with tf.variable_scope("GlobalAvgPool_3") :
    #     squeeze_3 = tf.reduce_mean(input_3, [1, 2])

    # # The first fully_connexted layers in the third sub-structure
    # with tf.variable_scope("fully_connected_31"):
    #     excitation_3 = tf.layers.dense(inputs=squeeze_3, use_bias=True, units=branch3_size // reduction_ratio)
    #     excitation_3 = tf.nn.relu(excitation_3)

    # # the second fully_connexted layers in the third sub-structure
    # with tf.variable_scope("fully_connected_32"):
    #     excitation_3 = tf.layers.dense(inputs=excitation_3, use_bias=True, units=branch3_size)
    #     excitation_3 = tf.nn.sigmoid(excitation_3)

    # # reshape the vector to a 4D densor
    # excitation_3 = tf.reshape(excitation_3, [-1, 1, 1, branch3_size])
    # ## In fact, this SE structure is to choose channel features, enhance useful channel 
    # # and restrain unuseful channel
    # output3 = output3 * excitation_3

    """
    The fourth sub-structure
    Three layers, two shortcut connection
    """
    # get the input for the second sub-network based on the branch4_size
    input_4 = input_layer[:,:,:, branch1_size + branch2_size + branch3_size : in_ch]
    # in4_ch = input_4.get_shape().as_list()[-1]
    # print(branch1_size + branch2_size + branch3_size)
    # print(in_ch)
    # print(in4_ch)
    with tf.variable_scope('conv_4_1_block'):
        conv41 = bn_relu_conv_layer_S(input_4, [3, 3, branch4_size, branch4_size], 1)
    padded_input4 = conv41
    # the second layer in the fourth sub-structure
    with tf.variable_scope('conv_4_2_block'):
        conv42 = bn_relu_conv_layer_S(conv41, [1, 1, branch4_size, branch4_size], 1)
    # the firt shorcut connection in the fourth sub-structure
    conv42 = input_4 + conv42
    # the third layer in the fourth sub-structure
    with tf.variable_scope('conv_4_3_block'):
        conv43 = bn_relu_conv_layer_S(conv42, [3, 3, branch4_size, branch4_size], 1)
    # the second shortcut connecction in the fourth sub-structure
    output4 = conv43 + padded_input4
    # with tf.variable_scope('dsr_right2'):
    #     output4 = dsr_attention(output4, branch4_size)
    
    # '''the SE block in the fourth sub-structure''' 
    # # to calculate the maximum value of the element on every dimension of a tensor. Channel
    # with tf.variable_scope("GlobalAvgPool_4") :
    #     squeeze_4 = tf.reduce_mean(input_4, [1, 2])

    # # The first fully_connexted layers,
    # with tf.variable_scope("fully_connected_41"):
    #     excitation_4 = tf.layers.dense(inputs=squeeze_4, use_bias=True, units=branch4_size // reduction_ratio)
    #     excitation_4 = tf.nn.relu(excitation_4)

    # # the second fully_connexted layers,
    # with tf.variable_scope("fully_connected_42"):
    #     excitation_4 = tf.layers.dense(inputs=excitation_4, use_bias=True, units=branch4_size)
    #     excitation_4 = tf.nn.sigmoid(excitation_4)

    # # reshape the vector to a 4D densor
    # excitation_4 = tf.reshape(excitation_4, [-1,1,1,branch4_size])
    # ## In fact, this SE structure is to choose channel features, enhance useful channel 
    # # and restrain unuseful channel
    # output4 = output4 * excitation_4
    '''
    contact the output of four sub-network to get the final output which has the same channel as the input
    '''
    output = tf.concat([output1, output2, output3, output4], 3)

    return output

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

def designdeNetwork(input_tensor_batch, reuse):
    '''
    The main function that defines the network.
    :param input_tensor_batch: 4D tensor
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''
    layers = []
    # The first conv layer with 3*3*16. From 32*32*3 to 32*32*32
    # 32 x 32 x 3
    # the first conv layer 
    with tf.variable_scope("conv1", reuse=reuse):
        conv1 = bn_relu_conv_layer_V(input_tensor_batch, [3, 3, 3, 64], 1)
        activation_summary(conv1)
        layers.append(conv1)
    # 30 x 30 x 64
    # the second conv layer 
    with tf.variable_scope("conv2", reuse=reuse):
        conv2 = bn_relu_conv_layer_V(layers[-1], [3, 3, 64, 256], 1)
        #conv2 = tf.nn.relu(conv2)
        activation_summary(conv2)
        layers.append(conv2)
    # 28 x 28 x 256
    '''
    three designed modules
    '''
    with tf.variable_scope("sgrise_3a", reuse=reuse):
        #sgrise_3a = SGRISE(layers[-1], 64, 64, 64, 64)
        sgrise_3a = SGRISE(layers[-1], 64, 64, 64, 64)
        activation_summary(sgrise_3a)
        layers.append(sgrise_3a)
    # 28 x 28 x 256
    with tf.variable_scope("sgrise_3b", reuse=reuse):
        # sgrise_3b = SGRISE(layers[-1], 48, 32, 96, 80)
        sgrise_3b = SGRISE(layers[-1], 48, 32, 96, 80)
        activation_summary(sgrise_3b)
        layers.append(sgrise_3b)
    # 28 x 28 x 256    
    with tf.variable_scope("sgrise_3c", reuse=reuse):
        # sgrise_3c = SGRISE(layers[-1], 32, 48, 80, 96)
        sgrise_3c = SGRISE(layers[-1], 32, 48, 80, 96)
        activation_summary(sgrise_3c)
        layers.append(sgrise_3c)
    # 28 x 28 x 256
    # avg pool
    # with tf.variable_scope('pool1', reuse=reuse):
    #     pool1 = tf.nn.avg_pool(layers[-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #     pool1 = tf.pad(pool1, [[0, 0], [0, 0], [0, 0], [128, 128]])
    #     activation_summary(pool1)
    #     layers.append(pool1)
    with tf.variable_scope("P1_p", reuse=reuse):
        p1 = bn_relu_conv_layer_S(layers[-1], [1, 1, 256, 512], 1)
        #conv2 = tf.nn.relu(conv2)
        activation_summary(p1)
        layers.append(p1)
    with tf.variable_scope("P1_dw", reuse=reuse):
        p1 = bn_depthconv_S(layers[-1], [3, 3, 512, 1], 2)
        #conv2 = tf.nn.relu(conv2)
        activation_summary(p1)
        layers.append(p1)
    # bn_depthconv(input_batch, filter_shape, stride)
    # 14 x 14 x 512
    '''
    three designed modules
    '''
    with tf.variable_scope("sgrise_4a", reuse=reuse):
        # sgrise_4a = SGRISE(layers[-1], 192, 208, 48, 64)
        sgrise_4a = SGRISE(layers[-1], 192, 208, 48, 64)
        activation_summary(sgrise_4a)
        layers.append(sgrise_4a)
    # 14 x 14 x 512        
    with tf.variable_scope("sgrise_4b", reuse=reuse):
        # sgrise_4b = SGRISE(layers[-1], 208, 48, 64, 192)
        sgrise_4b = SGRISE(layers[-1], 208, 48, 64, 192)
        activation_summary(sgrise_4b)
        layers.append(sgrise_4b)
    # 14 x 14 x 512
    with tf.variable_scope("sgrise_4c", reuse=reuse):
        # sgrise_4c = SGRISE(layers[-1], 48, 64, 192, 208)
        sgrise_4c = SGRISE(layers[-1], 48, 64, 192, 208)
        activation_summary(sgrise_4c)
        layers.append(sgrise_4c)
    # 14 x 14 x 512
    # avg pool
    # with tf.variable_scope('pool2', reuse=reuse):
    #     pool1 = tf.nn.avg_pool(layers[-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    #     pool1 = tf.pad(pool1, [[0, 0], [0, 0], [0, 0], [256, 256]])
    #     activation_summary(pool1)
    #     layers.append(pool1)
    with tf.variable_scope("P2_p", reuse=reuse):
        P2_p = bn_relu_conv_layer_S(layers[-1], [1, 1, 512, 1024], 1)
        #conv2 = tf.nn.relu(conv2)
        activation_summary(P2_p)
        layers.append(P2_p)
    with tf.variable_scope("P2_dw", reuse=reuse):
        P2_dw = bn_depthconv_S(layers[-1], [3, 3, 1024, 1], 2)
        #conv2 = tf.nn.relu(conv2)
        activation_summary(P2_dw)
        layers.append(P2_dw)
    # 7 x 7 x 1024
    '''
    three designed modules
    '''
    with tf.variable_scope("sgrise_5a", reuse=reuse):
        # sgrise_5a = SGRISE(layers[-1], 384, 384, 128, 128)
        sgrise_5a = SGRISE(layers[-1], 384, 384, 128, 128)
        activation_summary(sgrise_5a)
        layers.append(sgrise_5a)
    # 7 x 7 x 1024        
    with tf.variable_scope("sgrise_5b", reuse=reuse):
        # sgrise_5b = SGRISE(layers[-1], 384, 128, 128, 384)
        sgrise_5b = SGRISE(layers[-1], 384, 128, 128, 384)
        activation_summary(sgrise_5b)
        layers.append(sgrise_5b)
    # 7 x 7 x 1024
    with tf.variable_scope("sgrise_5c", reuse=reuse):
        # sgrise_5c = SGRISE(layers[-1], 128, 128, 384, 384)
        sgrise_5c = SGRISE(layers[-1], 128, 128, 384, 384)
        activation_summary(sgrise_5c)
        layers.append(sgrise_5c)
    # 7 x 7 x 1024
    # avg pool
    # with tf.variable_scope('pool3', reuse=reuse):
    #     pool3 = tf.nn.avg_pool(layers[-1], ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='VALID')
    #     activation_summary(pool3)
    #     layers.append(pool3)

    with tf.variable_scope('pool3', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        global_pool = batch_normalization_layer(layers[-1], in_channel)
        global_pool =  glpool(global_pool, [7, 7, 1024, 1], 1)
        global_pool = tf.reduce_mean(global_pool, [1, 2])
        activation_summary(global_pool)
        layers.append(global_pool)
    # 1 x 1 x 1024
    # with tf.variable_scope("reshape", reuse=reuse):
    #     reshape = tf.reshape(layers[-1], [-1, 1024])
    #     activation_summary(reshape)
    #     layers.append(reshape)
    #     #print(reshape.get_shape().as_list())

    if reuse is False:
        with tf.variable_scope('dropout'):
            dropout = tf.nn.dropout(layers[-1], 0.5)
            activation_summary(dropout)
            layers.append(dropout)

    with tf.variable_scope('logits', reuse=reuse):
        logits = output_layer(layers[-1], NUM_CLASS)
        activation_summary(logits)
        layers.append(logits)

    return layers[-1]    