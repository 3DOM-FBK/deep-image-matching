# Filename: tf_utils.py
# License: LICENSES/LICENSE_APACHE

# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.reset_default_graph()

def pre_x_in(x_in, opt="4"):
    """
    Input
        x_in: B1K4
    Output
        X: B1KC
    config.pre_x_in = 9
    """
    if opt == "9":
        x_shp = tf.shape(x_in)
        xx = tf.transpose(tf.reshape(
            x_in, (x_shp[0], x_shp[2], 4)), (0, 2, 1))
        # Create the matrix to be used for the eight-point algorithm
        X = tf.transpose(tf.stack([
            xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
            xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
            xx[:, 0], xx[:, 1], tf.ones_like(xx[:, 0])
        ], axis=1), (0, 2, 1))
        X = X[:, None]
    elif opt == "3":
        X = x_in
    else:
        X = x_in 
    return X

def topk(x_in, weights, num_top_k, verbose=False):
    """
    x_in: BNK4
    weights: BNK
    num_top_k: int
    """
    # rate
    num_kp = tf.shape(weights)[2]
    # num_top_divider = config.num_top_divider
    # num_top_k = tf.to_int32(num_kp / num_top_divider)
    B = tf.shape(weights)[0]
    num_pairs = tf.shape(weights)[1]
    values, mask = tf.nn.top_k(
        weights, k=num_top_k, sorted=False)
    # mask [B, 1, K]
    B_ = tf.range(B)
    num_pairs_ = tf.range(num_pairs)
    num_top_k_ = tf.range(num_top_k)
    index0, index1, index2 = tf.meshgrid(
       B_, num_pairs_, num_top_k_, indexing="ij")
    # index[index0, index1, topk]
    index = tf.stack([index0, index1, mask], -1)
    x_in = tf.gather_nd(x_in, index)
    return x_in, index

def gcn(linout, weight=None, opt="vanilla"):
    """
    Global Context Normalization:
        linout: B1KC
        weight: B1K1, default None. Precomputed weight  
        opt: "vanilla" is CN for CNe, "reweight_vanilla_sigmoid_softmax" is ACN for ACNe
    """
    if opt == "vanilla":
        var_eps = 1e-3
        mean, variance = tf.nn.moments(linout, axes=[2], keepdims=True)
        linout = tf.nn.batch_normalization(
            linout, mean, variance, None, None, var_eps)
    elif opt == "reweight_vanilla_sigmoid_softmax":
        if weight is None:
            in_shp = [_s if _s is not None else - 1 for _s in linout.get_shape().as_list()]
            with tf.variable_scope("reweight"):
                in_channel = in_shp[-1]
                # get W and b for conv1d.
                out_channel, ksize = 2, 1 
                dtype = tf.float32
                fanin = in_channel * ksize
                W = tf.get_variable(
                    "weights", shape=[1, ksize, in_channel, out_channel], dtype=dtype,
                    initializer=tf.truncated_normal_initializer(stddev=2.0 / fanin),
                )
                b = tf.get_variable(
                    "biases", shape=[out_channel], dtype=dtype,
                    initializer=tf.zeros_initializer(),
                )
                
                cur_padding = "VALID"
                data_format = "NHWC"
                tf.summary.histogram("W_attention", W)
                tf.summary.histogram("b_attention", b)

                logits = tf.nn.conv2d(
                    linout, W, [1, 1, 1, 1], cur_padding, data_format=data_format)
                logits = tf.nn.bias_add(logits, b, data_format=data_format)
                softmax_logit = logits[..., :1]
                sigmoid_logit = logits[..., -1:]
                mask = tf.nn.sigmoid(sigmoid_logit)
                tf.add_to_collection("logit_attention", sigmoid_logit)
                tf.add_to_collection("logit_softmax_attention", softmax_logit)

                eps = 0
                weight = tf.exp(softmax_logit) * mask
                weight = weight / (tf.reduce_sum(weight, 2, keepdims=True) + eps) 
                tf.add_to_collection("attention", weight)
                tf.summary.histogram("attention", weight)
                # weight = log_tf_tensor(weight, name="attention")
        # mean: B11C
        mean = tf.reduce_sum(weight * linout, 2, keepdims=True)
        # variance: B1KC
        variance = tf.square(linout - mean)
        variance = tf.reduce_sum(weight * variance, 2, keepdims=True)
        var_eps = 1e-3
        tf.add_to_collection("preNorm", linout)
        linout = (linout - mean) / tf.sqrt(variance + var_eps)
        tf.add_to_collection("posNorm", linout)
    else:
        raise ValueError("Don't support this type of gcn function")
    return linout, weight
