import tensorflow as tf
import tensorflow.contrib.slim as slim


def inception_a(net, scope=None):
    with tf.variable_scope(scope, 'inception_a', [net]):
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d], stride=1, padding='SAME'):

            with tf.variable_scope('tower_1'):
                tower_1_conv_1 = slim.conv2d(net, 16, 1, scope='Conv2d_1x1')
                tower_1_conv_2 = slim.conv2d(tower_1_conv_1, 16, 3, scope='Conv2d_3x3_1')
                tower_1_conv_3 = slim.conv2d(tower_1_conv_2, 16, 3, scope='Conv2d_3x3_2')
            with tf.variable_scope('tower_2'):
                tower_2_conv_1 = slim.conv2d(net, 16, 1, scope='Conv2d_1x1')
                tower_2_conv_2 = slim.conv2d(tower_2_conv_1, 16, 3, scope='Conv2d_3x3')
            with tf.variable_scope('tower_3'):
                tower_3_avgpool_1 = slim.avg_pool2d(net, 3, scope='AvgPool_3x3')
                tower_3_conv_2 = slim.conv2d(tower_3_avgpool_1, 16, 3, scope='Conv2d_3x3')
            with tf.variable_scope('tower_4'):
                tower_4_conv_1= slim.conv2d(net, 16, 1, scope='Conv2d_1x1')

            return tf.concat([tower_1_conv_3, tower_2_conv_2, tower_3_conv_2, tower_4_conv_1], 3)


def reduction_a(net, scope=None):
    with tf.variable_scope(scope, 'reduction_a', [net]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=2, padding='SAME'):

            with tf.variable_scope('tower_1'):
                tower_1_conv_1 = slim.conv2d(net, 32, 1, stride=1, scope='Conv2d_1x1')
                tower_1_conv_2 = slim.conv2d(tower_1_conv_1, 32, 3, stride=1, scope='Conv2d_3x3_1')
                tower_1_conv_3 = slim.conv2d(tower_1_conv_2, 32, 3, scope='Conv2d_3x3_2')
            with tf.variable_scope('tower_2'):
                tower_2_conv_1 = slim.conv2d(net, 32, 3, scope='Conv2d_3x3')
            with tf.variable_scope('tower_3'):
                tower_3_maxpool_1 = slim.max_pool2d(net, 3, scope='MaxPool_3x3')

            return tf.concat([tower_1_conv_3, tower_2_conv_1, tower_3_maxpool_1], 3)


def inception_v3(inputs, dropout_prob=0.7, weight_decay=0, is_training=True):

    batch_norm_params={"decay": 0.995, "epsilon": 0.001}

    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='VALID'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):

                # 26 x 26 x 8
                net = slim.conv2d(inputs, 16, 3, scope='Conv2d_1')

                # 24 x 24 x 16
                net = slim.conv2d(net, 16, 3, scope='Conv2d_2')

                # 24 x 24 x 64
                net = inception_a(net)

                # 12 x 12 x 96
                net = reduction_a(net)

                # 6 x 6 x 96
                net = reduction_a(net)

                # 6 x 6 x 48
                net = slim.conv2d(net, 48, 1, scope='Conv2d_3')

                # flatten
                net = slim.flatten(net)

                # dropout
                net = slim.dropout(net, dropout_prob)

                # fc 1728 -> 128
                net = slim.fully_connected(net, 128)

                # dropout
                net = slim.dropout(net, dropout_prob)

                # fc 128 -> 10
                net = slim.fully_connected(net, 10, activation_fn=None, scope='prediction')

                return net