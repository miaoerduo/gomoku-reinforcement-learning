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
        self._board_size = None  # 棋盘大小
        self._board = None  # 棋盘输入
        self._predict = None  # 预测结果
        self._loss = None  # 损失函数
        self._optimizer = None  # 优化器
        self._label = None  # 标签
        self._lr = 0.01  # 学习率
        self._is_training = False  # 是否训练模式

    @classmethod
    def trans_pos(cls, n, width):
        """
        根据下标计算落子的位置
        :param n:       下标
        :param width:   棋盘大小
        :return:
        """
        return n // width, n % width

    def make_predict(self, boards, sess):
        """
        预测落子，可以同时对一次或者多次棋局进行预测
        :param boards:  棋局
        :param sess:    会话
        :return:
        """

        # 对于棋局，无论是一局还是多局，都按照多局的策略去预测
        boards = np.array(boards)
        shape = boards.shape
        shape_length = len(shape)
        assert shape_length == 3 or shape_length == 4, 'the boards shape need to be 3 or 4'
        if shape_length == 3:
            boards = boards.reshape((1, shape[0], shape[1], shape[2]))

        raw_predict_map = sess.run([self._predict], feed_dict={self._board: boards})[0]

        # 找到当前可以落子的位置
        available_pos = (boards.sum(axis=-1) == 0)
        available_predict_map = raw_predict_map.copy()
        available_predict_map[False == available_pos] = float('-inf')
        max_predict_index = available_predict_map.reshape((available_predict_map.shape[0], -1)).argmax(axis=1)
        max_predict_index = [Agent.trans_pos(k, self._board_size) for k in max_predict_index]

        max_predict = [raw_predict_map[it][max_idx] for it, max_idx in enumerate(max_predict_index)]

        if shape_length == 3:
            return max_predict_index[0], max_predict[0], raw_predict_map[0]
        else:
            return max_predict_index, max_predict, raw_predict_map

    def build_model(self, board_size, is_training=True, reuse=False, name="agent", learning_rate=0.01):
        """
        创建模型
        :param board_size:      棋盘大小
        :param is_training:     是否训练
        :param reuse:           是否复用
        :param name:            名称
        :param learning_rate:   学习率
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

    def train(self, data, label, sess):
        """
        训练一个batch，返回当前的学习率和loss
        :param data:    训练数据
        :param label:   训练标签
        :param sess:    会话
        :return:
        """
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

        return lr, loss
