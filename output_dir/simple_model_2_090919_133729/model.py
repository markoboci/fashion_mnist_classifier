import tensorflow as tf
import tensorflow.contrib.slim as slim


def simple_model_2(inputs, dropout_prob=0.7, weight_decay=0, is_training=True):

    batch_norm_params={"decay": 0.995, "epsilon": 0.001}

    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='VALID'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):

                # 22 x 22 x 16
                net = slim.conv2d(inputs, 16, 7, scope='Conv2d_1')

                # 16 x 16 x 32
                net = slim.conv2d(net, 32, 7, scope='Conv2d_2')

                # 10 x 10 x 64
                net = slim.conv2d(net, 64, 7, scope='Conv2d_3')

                # 6 x 6 x 64
                net = slim.conv2d(net, 64, 5, scope='Conv2d_4')

                # 4 x 4 x 128
                net = slim.conv2d(net, 128, 3, scope='Conv2d_5')

                # flatten
                net = slim.flatten(net)

                # dropout
                net = slim.dropout(net, dropout_prob)

                # fc 2048 -> 128
                net = slim.fully_connected(net, 128)

                # dropout
                net = slim.dropout(net, dropout_prob)

                # fc 128 -> 10
                net = slim.fully_connected(net, 10, activation_fn=None, scope='prediction')

                return net