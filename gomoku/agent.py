# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class Agent(object):
    """
    玩家代理，用来进行玩家的策略模拟
    """

    def __init__(self):
        """
        初始化
        """
        self._board_size = None
        self._board = None
        self._predict = None
        self._loss = None
        self._optimizer = None
        self._label = None
        self._lr = 0.01
        self._is_training = False

    @classmethod
    def trans_pos(cls, n, width):
        return (n // width, n % width)

    def make_predict(self, boards, sess):
        """
        预测落子，可以同时对一次或者多次棋局进行预测
        :param boards:  棋局
        :param sess:    会话
        :return:
        """
        boards = np.array(boards)
        shape = boards.shape
        shape_length = len(shape)
        assert shape_length == 3 or shape_length == 4, 'the boards shape need to be 3 or 4'
        if shape_length == 3:
            boards = boards.reshape((1, shape[0], shape[1], shape[2]))
        raw_predict_map = sess.run([self._predict],
                                   feed_dict={
                                       self._board: boards,
                                   })[0]
        # 找到当前可以落子的位置
        available_pos = (boards.sum(axis=-1) == 0)
        available_predict_map = raw_predict_map.copy()
        available_predict_map[available_pos == False] = float('-inf')
        max_predict_index = available_predict_map.reshape((available_predict_map.shape[0], -1)).argmax(axis=1)
        max_predict_index = [Agent.trans_pos(k, self._board_size) for k in max_predict_index]
        max_predict = [raw_predict_map[k][max_predict_index[k]] for k in range(len(max_predict_index))]

        if shape_length == 3:
            return max_predict_index[0], max_predict[0], raw_predict_map[0]
        else:
            return max_predict_index, max_predict, raw_predict_map

    def build_model(self, board_size, is_training=True, reuse=False, name="agent", learning_rate=0.01):
        """
        创建模型
        :param board_size:
        :param is_training:
        :param reuse:
        :param name:
        :param learning_rate:
        :return:
        """
        from .network import QNet
        self._board_size = board_size
        self._board = tf.placeholder(tf.float32, shape=(None, board_size, board_size, 2))
        self._label = tf.placeholder(tf.float32, shape=(None, board_size, board_size))
        self._predict = QNet(self._board, width=board_size, is_training=is_training, reuse=reuse, scope=name)
        if learning_rate is not None:
            self._lr = learning_rate

        self._is_training = is_training
        if is_training:
            self._loss = tf.reduce_mean(tf.reduce_sum(tf.square(self._label - self._predict), axis=[1, 2]))
            self._optimizer = tf.train.GradientDescentOptimizer(self._lr).minimize(self._loss)

    def train(self, data, label, sess, display=True):
        if isinstance(self._lr, tf.Tensor):
            _, lr, loss = sess.run([self._optimizer, self._lr, self._loss], feed_dict={
                self._board: np.array(data),
                self._label: np.array(label)
            })
        else:
            _, loss = sess.run([self._optimizer, self._loss], feed_dict={
                self._board: np.array(data),
                self._label: np.array(label)
            })
            lr = self._lr
        if display:
            print('lr: {} loss: {}'.format(lr, loss))
