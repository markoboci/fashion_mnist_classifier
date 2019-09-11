import tensorflow as tf
import tensorflow.contrib.slim as slim


def conv_block(net, num_channels=64, scope=None):
    with tf.variable_scope(scope, 'conv_block', [net]):
        with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):

            with tf.variable_scope('tower_1'):
                branch_1_conv_1 = slim.conv2d(net, num_channels, 1, scope='Conv2d_1x1_1')
                branch_1_conv_2 = slim.conv2d(branch_1_conv_1, num_channels, 3, scope='Conv2d_3x3')
                branch_1_conv_3 = slim.conv2d(branch_1_conv_2, num_channels, 1, scope='Conv2d_1x1_2')
            with tf.variable_scope('tower_2'):
                branch_2_conv_1 = slim.conv2d(net, num_channels, 1, scope='Conv2d_1x1')

            return branch_1_conv_3 + branch_2_conv_1


def identity_block(net, scope=None):
    num_channels=net.shape[-1]
    with tf.variable_scope(scope, 'conv_block', [net]):
        with slim.arg_scope([slim.conv2d], stride=1, padding='SAME'):
            branch_conv_1 = slim.conv2d(net, num_channels, 1, scope='Conv2d_1x1_1')
            branch_conv_2 = slim.conv2d(branch_conv_1, num_channels, 3, scope='Conv2d_3x3')
            branch_conv_3 = slim.conv2d(branch_conv_2, num_channels, 1, scope='Conv2d_1x1_2')

            return net + branch_conv_3



def resnet(inputs, dropout_prob=0.7, weight_decay=0, is_training=True):

    batch_norm_params={"decay": 0.995, "epsilon": 0.001}

    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], stride=1, padding='VALID'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):

                # 26 x 26 x 8
                net = slim.conv2d(inputs, 16, 3, scope='Conv2d_1')

                # 24 x 24 x 32
                net = slim.conv2d(net, 32, 3, scope='Conv2d_2')

                # 24 x 24 x 64
                net = conv_block(net)

                # 24 x 24 x 64
                net = slim.repeat(net, 2, identity_block)

                # 12 x 12 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='SAME', scope='MaxPool_3')

                # 12 x 12 x 128
                net = conv_block(net, num_channels=128)

                # 12 x 12 x 128
                net = slim.repeat(net, 2, identity_block)

                # 6 x 6 x 128
                net = slim.max_pool2d(net, 3, stride=2, padding='SAME', scope='MaxPool_4')

                # flatten
                net = slim.flatten(net)

                # dropout
                net = slim.dropout(net, dropout_prob)

                # fc 4608 -> 128
                net = slim.fully_connected(net, 128)

                # dropout
                net = slim.dropout(net, dropout_prob)

                # fc 128 -> 10
                net = slim.fully_connected(net, 10, activation_fn=None, scope='prediction')

                return net