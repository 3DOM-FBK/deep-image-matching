# Filename: ops.py
# License: LICENSES/LICENSE_UVIC_EPFL
import numpy as np
from .tf_utils import gcn 

# From: https://github.com/shaohua0116/Group-Normalization-Tensorflow/blob/master/ops.py
def norm(x, norm_type, is_train, G=32, esp=1e-5):
    #
#   import tensorflow as tf
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    with tf.variable_scope('{}_norm'.format(norm_type)):
        if norm_type == 'none':
            output = x
        elif norm_type == 'bn':
            with tf.variable_scope("bn"):
                output = tf.layers.batch_normalization(
                    inputs=x,
                    center=False, scale=False,
                    training=is_train,
                    trainable=True,
                    axis=[-1],
                )
        elif norm_type == 'gn':
            # normalize
            # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
            x = tf.transpose(x, [0, 3, 1, 2])
            x_shp = tf.shape(x)
            N, C, H, W = x.get_shape().as_list()
            G = min(G, C)
            x = tf.reshape(x, [x_shp[0], G, int(C // G), x_shp[2], x_shp[3]])
            mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
            x = (x - mean) / tf.sqrt(var + esp)
            # per channel gamma and beta get_variable
            # gamma = tf.Variable(tf.constant(1.0, shape=[C]), dtype=tf.float32, name='gamma')
            # beta = tf.Variable(tf.constant(0.0, shape=[C]), dtype=tf.float32, name='beta')
            gamma = tf.get_variable('gamma', [C], dtype=tf.float32, initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable('beta', [C], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
            gamma = tf.reshape(gamma, [1, C, 1, 1])
            beta = tf.reshape(beta, [1, C, 1, 1])

            output = tf.reshape(x, [x_shp[0], x_shp[1], x_shp[2], x_shp[3]]) * gamma + beta
            # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
            output = tf.transpose(output, [0, 2, 3, 1])
        else:
            raise NotImplementedError
    return output

def tf_skew_symmetric(v):

    import tensorflow as tf

    zero = tf.zeros_like(v[:, 0])

    M = tf.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)

    return M

def tf_get_shape_as_list(x):

    return [_s if _s is not None else - 1 for _s in x.get_shape().as_list()]

def bn_act(linout, perform_gcn, perform_bn, activation_fn, is_training,
           data_format, config, weight=None):

    weight_output = None

    """ Perform batch normalization and activation """
    if data_format == "NHWC":
        axis = -1
    else:
        axis = 1

    # Global Context normalization on the input
    if perform_gcn:
        linout, weight_output = gcn(linout, weight, opt=config.gcn_opt)

    if perform_bn:
        linout = norm(linout, norm_type=config.bn_opt, is_train=is_training)

    if activation_fn is None:
        output = linout
    else:
        output = activation_fn(linout)

    return output, weight_output


def get_W_b_conv1d(in_channel, out_channel, ksize, dtype=None):

#   import tensorflow as tf
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    if dtype is None:
        dtype = tf.float32

    fanin = in_channel * ksize
    W = tf.get_variable(
        "weights", shape=[1, ksize, in_channel, out_channel], dtype=dtype,
        initializer=tf.truncated_normal_initializer(stddev=2.0 / fanin),
        # initializer=tf.random_normal_initializer(stddev=0.02),
    )
    b = tf.get_variable(
        "biases", shape=[out_channel], dtype=dtype,
        initializer=tf.zeros_initializer(),
    )
    # tf.summary.histogram("W", W)
    # tf.summary.histogram("b", b)

    return W, b


def conv1d_layer(inputs, ksize, nchannel, activation_fn, perform_bn,
                 perform_gcn, is_training, config, perform_kron=False,
                 padding="CYCLIC", data_format="NCHW",
                 act_pos="post", weight=None):

#   import tensorflow as tf
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    assert act_pos == "pre" or act_pos == "post"

    # Pad manually
    if padding == "CYCLIC":
        if ksize > 1:
            inputs = conv1d_pad_cyclic(
                inputs, ksize, 1, data_format=data_format)
        cur_padding = "VALID"
    else:
        cur_padding = padding

    in_shp = tf_get_shape_as_list(inputs)
    if data_format == "NHWC":
        in_channel = in_shp[-1]
        ksizes = [1, 1, ksize, 1]
    else:
        in_channel = in_shp[1]
        ksizes = [1, 1, 1, ksize]

    assert len(in_shp) == 4

    # # Lift with kronecker
    # if not is_first:
    #     inputs = tf.concat([
    #         inputs,
    #         kronecker_layer(inputs),
    #     ], axis=-1)

    pool_func = None
    self_ksize = ksize
    do_add = False

    # If pre activation
    if act_pos == "pre":
        inputs, weight_output = bn_act(inputs, perform_gcn, perform_bn, activation_fn,
                        is_training, data_format, config, weight)

    # Normal convolution
    with tf.variable_scope("self-conv"):
        W, b = get_W_b_conv1d(in_channel, nchannel, self_ksize)
        # Convolution in the valid region only
        linout = tf.nn.conv2d(
            inputs, W, [1, 1, 1, 1], cur_padding, data_format=data_format)
        linout = tf.nn.bias_add(linout, b, data_format=data_format)
    # Pooling Convolution for the summary route
    if pool_func is not None:
        with tf.variable_scope("neigh-conv"):
            if not do_add:
                linout = pool_func(
                    linout,
                    ksize=ksizes,
                    strides=[1, 1, 1, 1],
                    padding=cur_padding,
                    data_format=data_format,
                )
            else:
                W_n, b_n = get_W_b_conv1d(in_channel, nchannel, 1)
                # Convolution in the valid region only
                linout_n = tf.nn.conv2d(
                    inputs, W_n, [1, 1, 1, 1], "VALID", data_format=data_format
                )
                linout_n = tf.nn.bias_add(
                    linout_n, b_n, data_format=data_format)
                linout_n = pool_func(
                    linout_n,
                    ksize=ksizes,
                    strides=[1, 1, 1, 1],
                    padding=cur_padding,
                    data_format=data_format,
                )
                # # Crop original linout
                # if ksize > 1:
                #     if np.mod(ksize, 2) == 0:
                #         crop_st = ksize // 2 - 1
                #     else:
                #         crop_st = ksize // 2
                #         crop_ed = ksize // 2
                #     linout = linout[:, :, :, crop_st:-crop_ed]
                # Add to the original output
                linout = linout + linout_n

    # If post activation
    output = linout
    if act_pos == "post":
        output, weight_output = bn_act(linout, perform_gcn, perform_bn, activation_fn,
                        is_training, data_format, config, weight)

    return output, weight_output


def conv1d_resnet_block(inputs, ksize, nchannel, activation_fn, is_training, config,
                        midchannel=None, perform_bn=False, perform_gcn=False,
                        padding="CYCLIC", act_pos="post", data_format="NCHW", weight=None):

#   import tensorflow as tf
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

    # In case we want to do a bottleneck layer
    if midchannel is None:
        midchannel = nchannel

    # don't activate conv1 in case of midact
    conv1_act_fn = activation_fn
    if act_pos == "mid":
        conv1_act_fn = None
        act_pos = "pre"

    # Pass branch
    with tf.variable_scope("pass-branch"):
        # passthrough to be used when num_outputs != num_inputs
        in_shp = tf_get_shape_as_list(inputs)
        if data_format == "NHWC":
            in_channel = in_shp[-1]
        else:
            in_channel = in_shp[1]
        if in_channel != nchannel:
            cur_in = inputs
            # Simply change channels through 1x1 conv
            with tf.variable_scope("conv"):
                cur_in, weight_output = conv1d_layer(
                    inputs=inputs, ksize=1,
                    nchannel=nchannel,
                    activation_fn=None,
                    perform_bn=False,
                    perform_gcn=False,
                    is_training=is_training,
                    padding=padding,
                    data_format=data_format,
                    weight=weight,
                    config=config,
                )
            orig_inputs = cur_in
        else:
            orig_inputs = inputs

    # Conv branch
    with tf.variable_scope("conv-branch"):
        cur_in = inputs
        # Do bottle neck if necessary (Linear)
        if midchannel != nchannel:
            with tf.variable_scope("preconv"):
                cur_in, weight_output = conv1d_layer(
                    inputs=cur_in, ksize=1,
                    nchannel=nchannel,
                    activation_fn=None,
                    perform_bn=False,
                    perform_gcn=False,
                    is_training=is_training,
                    padding=padding,
                    data_format=data_format,
                    weight=weight,
                    config=config,
                )
                cur_in = activation_fn(cur_in)

        for i in range(config.num_inner):
            # Main convolution
            with tf.variable_scope("conv{}".format(i+1)):
                # right branch
                cur_in, weight_output = conv1d_layer(
                    inputs=cur_in, ksize=ksize,
                    nchannel=nchannel,
                    activation_fn=conv1_act_fn,
                    perform_bn=perform_bn,
                    perform_gcn=perform_gcn,
                    is_training=is_training,
                    padding=padding,
                    act_pos=act_pos,
                    data_format=data_format,
                    weight=weight,
                    config=config,
                )

        # Do bottle neck if necessary (Linear)
        if midchannel != nchannel:
            with tf.variable_scope("postconv"):
                cur_in, weight_output = conv1d_layer(
                    inputs=cur_in, ksize=1,
                    nchannel=nchannel,
                    activation_fn=None,
                    perform_bn=False,
                    perform_gcn=False,
                    is_training=is_training,
                    padding=padding,
                    data_format=data_format,
                    weight=weight,
                    config=config,
                )
                cur_in = activation_fn(cur_in)

    # Crop lb or rb accordingly
    if padding == "VALID" and ksize > 1:
        # Crop pass branch results
        if np.mod(ksize, 2) == 0:
            crop_st = ksize // 2 - 1
        else:
            crop_st = ksize // 2
            crop_ed = ksize // 2
            if data_format == "NHWC":
                orig_inputs = orig_inputs[:, :,  crop_st:-crop_ed, :]
            else:
                orig_inputs = orig_inputs[:, :, :, crop_st:-crop_ed]

    return cur_in + orig_inputs, weight_output

#
# ops.py ends here
