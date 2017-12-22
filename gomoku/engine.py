# -*- coding: utf-8 -*-

import numpy as np


class GomokuEngine(object):
    """
    五子棋游戏引擎，可以完成五子棋的各种状态的模拟，包含了棋盘，落子，胜利等数据
    """

    def __init__(self, board_size, board=None):
        """
        构造函数
        :param board_size:  棋盘大小
        """
        self._board = None
        self._board_size = None
        self._always1st = False
        self._actor = 0
        self._history = []
        self._winner = -1
        self._is_over = False
        self._available = None
        self.reset(board_size, board)

    def reset(self, board_size=None, board=None, copy=False, is_over=False, winner=-1):
        """
        重置游戏状态，为了简化初始化操作，不进行游戏是否终止的判定
        :param board_size:  棋盘大小
        :param board:       棋盘
        :param copy:        引用或是拷贝棋盘
        :param is_over:     游戏是否结束(reset不自动检查)
        :param winner:      获胜玩家(reset不自动检查)
        :return:
        """
        if board_size is not None:
            self._board_size = board_size
        if board is None:
            self._board = np.zeros((self._board_size, self._board_size, 2))
        else:
            self._board = board
            if copy:
                self._board = self._board.copy()
        self._available = (self._board.sum(-1) == 0)
        self._actor = 0 if self._board[:,:,0].sum() == self._board[:,:,1].sum() else 1
        self._history = []
        self._winner = winner
        self._is_over = is_over
        return self._check_shape()

    def _check_shape(self):
        """
        检查参数的形状
        :return:
        """
        if self._board_size <= 5:
            return False
        if self._board.shape != (self._board_size, self._board_size, 2):
            return False
        return True

    def make_move(self, action):
        """
        落子
        :param action:  元组，表示落子的位置
        :return:
        """

        # 游戏已经结束了
        if self._is_over:
            return False

        # 已经有子
        if not self._available[action]:
            return False

        self._board[action][self._actor] = 1
        self._available[action] = False
        self._history.append((action, self._actor))
        if self._is_win(self._actor):
            self._winner = self._actor
            self._is_over = True
        if not self._available.any():
            self._is_over = True

        self._actor = 1 - self._actor
        return True

    def _is_win(self, actor):
        """
        玩家是否获胜
        :param actor:   玩家id，简化判断的复杂度
        :return:
        """
        # 行
        for row in range(self._board_size):
            for col in range(self._board_size - 4):
                # 行
                horizontal = self._board[row, col: col+5, actor]
                if (horizontal == 1).all():
                    return True

        # 列
        for col in range(self._board_size):
            for row in range(self._board_size - 4):
                vertical = self._board[row: row+5, col, actor]
                if (vertical == 1).all():
                    return True

        # 斜线方向
        for row in range(self._board_size - 4):
            for col in range(self._board_size - 4):

                # 斜 左上到右下
                diagonal1 = np.array([self._board[row+k, col+k, actor] for k in range(5)])
                if (diagonal1 == 1).all():
                    return True

                # 斜 右上到左下
                diagonal2 = np.array([self._board[row+k, col+4-k, actor] for k in range(5)])
                if (diagonal2 == 1).all():
                    return True
        return False

    def undo(self, step=1):
        """
        悔棋
        :param step:    悔棋的步数
        :return:
        """
        if step != 0 and len(self._history) > 0:
            self._winner = -1
            self._is_over = False

        while step != 0 and len(self._history) > 0:
            undo_action, undo_actor = self._history.pop()
            self._board[undo_action][undo_actor] = 0
            self._available[undo_action] = True
            self._actor = undo_actor
            step -= 1

    def display(self):
        """
        根据历史记录打印棋谱，如果是通过reset的棋盘，则可能没有完整的历史，打印会有问题
        :return:
        """
        display_board = np.zeros((self._board_size, self._board_size), np.int32)

        step = 1
        for action, actor in self._history:
            display_board[action] = step * (1 if actor == 0 else -1)
            step += 1

        num_len = len(str(self._board_size * self._board_size))
        content = "+" + ("-" * ((num_len + 1) * self._board_size - 1)) + "+\n"
        for row in range(len(display_board)):
            content += "|"
            for col in range(len(display_board[0])):
                if display_board[row][col] > 0:
                    color_format = '\033[0;31m{:^' + str(num_len) + '}\033[0m|'
                else:
                    color_format = '\033[0;32m{:^' + str(num_len) + '}\033[0m|'
                content += color_format.format(abs(display_board[row][col]) if display_board[row][col] != 0 else "")
            content += "\n"
            content += "+" + ("-" * ((num_len + 1) * self._board_size - 1)) + "+\n"

        print("player \033[0;31m0\033[0m  player \033[0;32m1\033[0m ")
        print(content)

    def get_board(self):
        """
        获取棋局
        :return:
        """
        return self._board

    def get_actor(self):
        """
        获取该落子的玩家id
        :return:
        """
        return self._actor

    def get_winner(self):
        """
        获取赢家id，如果没有人获胜，则返回-1
        :return:
        """
        return self._winner

    def get_available(self):
        """
        获取棋盘可用的区域，返回棋盘相同大小的bool矩阵，True表示可以落子
        :return:
        """
        return self._available

    def is_over(self):
        """
        游戏是否结束
        :return:
        """
        return self._is_over

