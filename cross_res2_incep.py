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

def res2(input_batch, output_channel, k_size):
    '''
    the realization of res2net, which also add the channel shuffle block
    :param input_batch: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param k_size: int. kernel of convolution
    :return: 4D tensor.
    '''
    #num_groups = np.array([2, 4, 8, 16, 32])
    num_groups = tf.constant([2, 4, 8, 16, 32], tf.int32)
    num_shuffle = tf.random.shuffle(num_groups)
    input_batch = channel_shuffle_layer(input_batch, num_shuffle[0])
    group_ch = output_channel // 4
    input_1 = input_batch[:,:,:, 0 : group_ch]
    input_2 = input_batch[:,:,:, group_ch : 2 * group_ch]
    input_3 = input_batch[:,:,:, 2 * group_ch : 3 * group_ch]
    input_4 = input_batch[:,:,:, 3 * group_ch : 4 * group_ch]
    # input1_2_1 = subinput1_2[:,:,:, 0 : group_channel]
    with tf.variable_scope('conv_sub1'):
        conv_sub1 = bn_relu_conv_layer(input_1, [k_size, k_size, group_ch, group_ch], 1)

    ######### the second sub-group
    input_2 = input_2 + conv_sub1
    with tf.variable_scope('conv_sub2'):
        conv_sub2 = bn_relu_conv_layer(input_2, [k_size, k_size, group_ch, group_ch], 1)

    ######### the third sub-group
    input_3 = input_3 + conv_sub2
    with tf.variable_scope('conv_sub3'):
        conv_sub3 = bn_relu_conv_layer(input_3, [k_size, k_size, group_ch, group_ch], 1)

    ######### the fouth sub-group
    input_4 = input_4 + conv_sub3
    with tf.variable_scope('conv_sub4'):
        conv_sub4 = bn_relu_conv_layer(input_4, [k_size, k_size, group_ch, group_ch], 1)
        
    ########## concat the four sub-group to the output channel
    res2 = tf.concat([conv_sub1, conv_sub2, conv_sub3, conv_sub4], 3)

    return res2

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

def cross_res2_incep_block(input_layer, output_channel):
    '''
    Defines a inception block in Net
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :return: 4D tensor.
    '''
    ############################
    # The left sub-structure
    # Three layers, two shortcut connection
    ############################
    ###### The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_1_left'):
        conv11 = bn_relu_conv_layer(input_layer, [1, 1, output_channel, output_channel], 1)
    padded_input1 = conv11
    ####### the second layer
    with tf.variable_scope('conv1_2_left'):
        conv12 = res2(conv11, output_channel, 3)
    ####### the first shorcut connection
    conv12 = input_layer + conv12
    ####### the third layer
    with tf.variable_scope('conv1_3_left'):
        conv13 = bn_relu_conv_layer(conv12, [1, 1, output_channel, output_channel], 1)
    ####### the second shorcut connection
    output1 = conv13 + padded_input1
    ####### DSR module
    with tf.variable_scope('dsr_left'):
        output1 = dsr_attention(output1, output_channel)

    ##########################################
    ###### The Middle sub-structure
    ###### Three layers, two shortcut connection
    ###### The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv2_1_middle'):
        conv21 = bn_relu_conv_layer(input_layer, [1, 1, output_channel, output_channel], 1)
    padded_input2 = conv21
    ####### the second layer
    with tf.variable_scope('conv2_2_middle'):
        conv22 = res2(conv21, output_channel, 5)
    ####### the first shorcut connection
    conv22 = input_layer + conv22
    ####### the third layer
    with tf.variable_scope('conv2_3_middle'):
        conv23 = bn_relu_conv_layer(conv22, [1, 1, output_channel, output_channel], 1)
    ####### the second shorcut connection
    output2 = conv23 + padded_input2
    ####### DSR module
    with tf.variable_scope('dsr_middle'):
        output2 = dsr_attention(output2, output_channel)

    ###########################################
    #######  The right sub-structure
    #######  Three layers, two shortcut connection
    ######## The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv3_1_right'):
        conv31 = res2(input_layer, output_channel, 3)
    padded_input3 = conv31
    ######## the second layer
    with tf.variable_scope('conv3_2_right'):
        conv32 = res2(conv31, output_channel, 3)
    ######## the first shorcut connection
    conv32 = input_layer + conv32
    ######## the third layer
    with tf.variable_scope('conv3_3_right'):
        conv33 = res2(conv32, output_channel, 3)
    ######## the second shorcut connection
    output3 = conv33 + padded_input3
    ####### DSR module
    with tf.variable_scope('dsr_right'):
        output3 = dsr_attention(output3, output_channel)
    ####### final output
    output = output1 + output2 + output3
    
    return output

# def cross_res_incep_block(input_layer, output_channel, first_block=False):
#     '''
#     Defines a inception block in Net
#     :param input_layer: 4D tensor
#     :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
#     :param first_block: if this is the first inception block of the whole network
#     :return: 4D tensor.
#     '''
#     """
#     The left sub-structure
#     Three layers, two shortcut connection
#     """
#     # The first conv layer of the first residual block does not need to be normalized and relu-ed.
#     with tf.variable_scope('conv1_1_in_block'):
#         if first_block:
#             filter = create_variables(name='conv', shape=[1, 1, output_channel, output_channel])
#             conv11 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
#         else:
#             conv11 = bn_relu_conv_layer(input_layer, [1, 1, output_channel, output_channel], 1)
    
#     padded_input1 = conv11

#     # the second layer
#     with tf.variable_scope('conv1_2_in_block'):
#         conv12 = bn_relu_conv_layer(conv11, [3, 3, output_channel, output_channel], 1)

#     # the firt shorcut connection
#     conv12 = input_layer + conv12

#     # the third layer
#     with tf.variable_scope('conv1_3_in_block'):
#         conv13 = bn_relu_conv_layer(conv12, [1, 1, output_channel, output_channel], 1)

#     output1 = conv13 + padded_input1

#     """
#     The Middle sub-structure
#     Three layers, two shortcut connection
#     """
#     # The first conv layer of the first residual block does not need to be normalized and relu-ed.
#     with tf.variable_scope('conv2_1_in_block'):
#         if first_block:
#             filter = create_variables(name='conv', shape=[1, 1, output_channel, output_channel])
#             conv21 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
#         else:
#             conv21 = bn_relu_conv_layer(input_layer, [1, 1, output_channel, output_channel], 1)

#     padded_input2 = conv21

#     # the second layer
#     with tf.variable_scope('conv2_2_in_block'):
#         conv22 = bn_relu_conv_layer(conv21, [5, 5, output_channel, output_channel], 1)

#     # the firt shorcut connection
#     conv22 = input_layer + conv22

#     # the third layer
#     with tf.variable_scope('conv2_3_in_block'):
#         conv23 = bn_relu_conv_layer(conv22, [1, 1, output_channel, output_channel], 1)

#     output2 = conv23 + padded_input2

#     """
#     The right sub-structure
#     Three layers, two shortcut connection
#     """
#     # The first conv layer of the first residual block does not need to be normalized and relu-ed.
#     with tf.variable_scope('conv3_1_in_block'):
#         if first_block:
#             filter = create_variables(name='conv', shape=[3, 3, output_channel, output_channel])
#             conv31 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
#         else:
#             conv31 = bn_relu_conv_layer(input_layer, [3, 3, output_channel, output_channel], 1)

#     padded_input3 = conv31

#     # the second layer
#     with tf.variable_scope('conv3_2_in_block'):
#         conv32 = bn_relu_conv_layer(conv31, [3, 3, output_channel, output_channel], 1)

#     # the firt shorcut connection
#     conv32 = input_layer + conv32

#     # the third layer
#     with tf.variable_scope('conv3_3_in_block'):
#         conv33 = bn_relu_conv_layer(conv32, [3, 3, output_channel, output_channel], 1)

#     output3 = conv33 + padded_input3

#     output = output1 + output2 + output3
    
#     return output

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
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, 64], 1)
        activation_summary(conv0)
        layers.append(conv0)

    for i in range(n):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            # conv1 = cross_res_incep_block(layers[-1], 64)
            conv1 = cross_res2_incep_block(layers[-1], 64)
            activation_summary(conv1)
            layers.append(conv1)
    
    with tf.variable_scope('pool1',reuse=reuse):
        pool1 = tf.nn.avg_pool(layers[-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool1 = tf.pad(pool1, [[0, 0], [0, 0], [0, 0], [32, 32]])
        activation_summary(pool1)
        layers.append(pool1)

    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            conv2 = cross_res2_incep_block(layers[-1], 128)
            activation_summary(conv2)
            layers.append(conv2)

    with tf.variable_scope('pool2',reuse=reuse):
        pool2 = tf.nn.avg_pool(layers[-1], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        pool2 = tf.pad(pool2, [[0, 0], [0, 0], [0, 0], [64, 64]])
        activation_summary(pool2)
        layers.append(pool2)

    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            conv3 = cross_res2_incep_block(layers[-1], 256)
            activation_summary(conv3)
            layers.append(conv3)

    # with tf.variable_scope('fc', reuse=reuse):
    #     in_channel = layers[-1].get_shape().as_list()[-1]
    #     bn_layer = batch_normalization_layer(layers[-1], in_channel)
    #     # relu_layer = tf.nn.relu(bn_layer)
    #     #global_pool =  glpool(bn_layer, [8, 8, 256, 1], 1)
    #     global_pool = tf.reduce_mean(bn_layer, [1, 2])
    #     # assert global_pool.get_shape().as_list()[-1:] == [64]
    #     output = output_layer(global_pool, 100)
    #     activation_summary(output)
    #     layers.append(output)

    with tf.variable_scope('pool3', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel)
        pool3 = tf.nn.avg_pool(bn_layer, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
        #pool3 = depthwise_conv(layers[-1], [7, 7, 1024, 1], 1)
        pool3 = tf.reshape(pool3, [-1, 256])
        activation_summary(pool3)
        layers.append(pool3)

    # if reuse is False:
    #     with tf.variable_scope('dropout'):
    #         dropout = tf.nn.dropout(layers[-1], 0.5)
    #         activation_summary(dropout)
    #         layers.append(dropout)

    with tf.variable_scope('logits', reuse=reuse):
        logits = output_layer(layers[-1], 100)
        activation_summary(logits)
        layers.append(logits)

    return layers[-1]

# def test_graph(train_dir='logs'):
#     '''
#     Run this function to look at the graph structure on tensorboard. A fast way!
#     :param train_dir:
#     '''
#     input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
#     result = inference(input_tensor, 2, reuse=False)
#     init = tf.initialize_all_variables()
#     sess = tf.Session()
#     sess.run(init)
#     summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)