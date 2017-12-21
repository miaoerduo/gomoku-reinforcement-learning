# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib import slim as slim

def QNet(inputs, width, is_training=True, reuse=False, scope="QNet"):
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.convolution, slim.fully_connected],
                            weights_initializer=slim.initializers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.01),
                            biases_initializer=tf.zeros_initializer(),
                            biases_regularizer=slim.l2_regularizer(0.01),
                            ):
            net = slim.convolution(
                inputs=inputs,
                num_outputs=32,
                kernel_size=5,
                stride=1,
                activation_fn=tf.nn.relu,
                padding="VALID",
                scope="conv1"
            )
            net = slim.batch_norm(net, is_training=is_training)
            net = slim.convolution(
                inputs=net,
                num_outputs=64,
                kernel_size=5,
                stride=1,
                activation_fn=tf.nn.relu,
                padding="VALID",
                scope="conv2"
            )
            net = slim.batch_norm(net, is_training=is_training)
            net = slim.flatten(net)
            net = slim.fully_connected(
                inputs=net,
                num_outputs=512,
                activation_fn=tf.nn.relu,
                scope="fc1",
            )
            net = slim.batch_norm(net, is_training=is_training)
            net = slim.fully_connected(
                inputs=net,
                num_outputs=512,
                activation_fn=tf.nn.relu,
                scope="fc2"
            )
            net = slim.batch_norm(net, is_training=is_training)
            net = slim.fully_connected(
                inputs=net,
                num_outputs=width ** 2,
                activation_fn=None,
                scope="fc3"
            )
            net = tf.reshape(net, (-1, width, width))
            return net