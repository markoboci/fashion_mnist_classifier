import tensorflow as tf
import tensorflow.contrib.slim as slim


def simple_model_1(inputs, dropout_prob=0.7, weight_decay=0, is_training=True):

    batch_norm_params={"decay": 0.995, "epsilon": 0.001}

    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='VALID'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):

                # 26 x 26 x 8
                net = slim.conv2d(inputs, 8, 3, scope='Conv2d_1')

                # 24 x 24 x 16
                net = slim.conv2d(net, 16, 3, scope='Conv2d_2')

                # 12 x 12 x 16
                net = slim.max_pool2d(net, 2, stride=2, scope='MaxPool_2')

                # 10 x 10 x 32
                net = slim.conv2d(net, 32, 3, scope='Conv2d_3')

                # 5 x 5 x 32
                net = slim.max_pool2d(net, 2, stride=2, scope='MaxPool_3')

                # 5 x 5 X 32
                net = slim.conv2d(net, 32, 2, padding='SAME', scope='Conv2d_4')

                # flatten
                net = slim.flatten(net)

                # dropout
                net = slim.dropout(net, dropout_prob)

                # fc 800 -> 128
                net = slim.fully_connected(net, 128)

                # dropout
                net = slim.dropout(net, dropout_prob)

                # fc 128 -> 10
                net = slim.fully_connected(net, 10, activation_fn=None, scope='prediction')

                return net