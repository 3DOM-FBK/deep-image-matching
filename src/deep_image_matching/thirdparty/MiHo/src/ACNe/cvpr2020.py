# Filename: cvpr2020.py
# License: LICENSES/LICENSE_UVIC_EPFL
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .ops import conv1d_layer, conv1d_resnet_block


def build_graph(x_in, is_training, config, weight=None):
    vis_dict = {}

    activation_fn = tf.nn.relu

    x_in_shp = tf.shape(x_in)

    cur_input = x_in
    # print(cur_input.shape)
    idx_layer = 0
    numlayer = config.net_depth
    ksize = 1
    nchannel = config.net_nchannel
    # Use resnet or simle net
    act_pos = config.net_act_pos
    conv1d_block = conv1d_resnet_block

    # First convolution
    with tf.variable_scope("hidden-input"):
        vis_dict[cur_input.name] = cur_input
        cur_input, weight_output = conv1d_layer(
            inputs=cur_input,
            ksize=1,
            nchannel=nchannel,
            activation_fn=None,
            perform_bn=False,
            perform_gcn=False,
            is_training=is_training,
            act_pos="pre",
            data_format="NHWC",
            weight=weight,
            config=config,
        )
        # print(cur_input.shape)
        if weight_output is not None:
            vis_dict["attention {}".format(weight_output.name)] = weight_output
    for _ksize, _nchannel in zip(
            [ksize] * numlayer, [nchannel] * numlayer):
        scope_name = "hidden-" + str(idx_layer)
        with tf.variable_scope(scope_name):
            vis_dict[cur_input.name] = cur_input
            cur_input, weight_output = conv1d_block(
                inputs=cur_input,
                ksize=_ksize,
                nchannel=_nchannel,
                activation_fn=activation_fn,
                is_training=is_training,
                perform_bn=config.net_batchnorm,
                perform_gcn=config.net_gcnorm,
                act_pos=act_pos,
                data_format="NHWC",
                weight=weight,
                config=config,
            )
            # Apply pooling if needed
            # print(cur_input.shape)
            if weight_output is not None:
                vis_dict["attention_{}".format(weight_output.name)] = weight_output

        idx_layer += 1
    
    if config.nonlinearity_output:
        with tf.variable_scope("nonlinearity"):
            cur_input, weight_output = conv1d_layer(
                inputs=cur_input,
                ksize=_ksize,
                nchannel=_nchannel,
                activation_fn=activation_fn,
                is_training=is_training,
                perform_bn=True,
                perform_gcn=False,
                data_format="NHWC",
                weight=weight,
                config=config,
            )

    if config.weight_opt.startswith("sigmoid_softmax"):
        # logit for global attention
        with tf.variable_scope("logit_softmax"):
            logit_softmax, weight_output = conv1d_layer(
                inputs=cur_input,
                ksize=1,
                nchannel=1,
                activation_fn=None,
                is_training=is_training,
                perform_bn=False,
                perform_gcn=False,
                data_format="NHWC",
                weight=weight,
                config=config,
            )
            logit_softmax = tf.reshape(logit_softmax, (x_in_shp[0], x_in_shp[2]))
            vis_dict["logit_softmax"] = logit_softmax

    with tf.variable_scope("output"):
        # logit for local attention
        vis_dict[cur_input.name] = cur_input
        cur_input,weight_output = conv1d_layer(
            inputs=cur_input,
            ksize=1,
            nchannel=1,
            activation_fn=None,
            is_training=is_training,
            perform_bn=False,
            perform_gcn=False,
            data_format="NHWC",
            weight=weight,
            config=config,
        )
        #  Flatten
        if weight_output is not None:
            vis_dict["attention_{}".format(weight_output.name)] = weight_output
        cur_input = tf.reshape(cur_input, (x_in_shp[0], x_in_shp[2]))

    logits = cur_input
    # print(cur_input.shape)

    return logits, vis_dict


#
# cvpr2018.py ends here
