# -*- coding: utf-8 -*-

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

import tensorflow as tf
from tensorflow.contrib import slim

from gomoku import engine, agent


BOARD_SIZE = 11
ITER_NUM = 10000
SAVE_INTERVAL = 1000
BATCH_SIZE = 1024
WIN_REWARD = 10
LOSS_REWARD = -20
EVEN_REWARD = -0.2
KEEP_REWARD = -0.4
GAMMA1 = 0.99
GAMMA2 = GAMMA1 * GAMMA1
LAMBDA = 0.8
BASE_LEARNING_RATE = 0.01
BOARD_HISTORY_LIMIT = 1000
BOARD_HISTORY_BASE_NUM = 1024

pos_map = np.array(range(BOARD_SIZE * BOARD_SIZE)).reshape((BOARD_SIZE, BOARD_SIZE))


def get_available_pos(board):
    return pos_map[board.sum(-1) == 0]

def get_reward(game):
    if not game.is_over():
        # 游戏继续
        return KEEP_REWARD
    elif game.get_winner() == -1:
        # 平局
        return EVEN_REWARD
    else:
        return WIN_REWARD

def main():



    game = engine.GomokuEngine(board_size=BOARD_SIZE)
    player = agent.Agent()


    global_step = slim.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(
        BASE_LEARNING_RATE, global_step,
        1000, 0.96, staircase=True)
    player.build_model(
        board_size=BOARD_SIZE, is_training=True,
        reuse=False, name="agent",
        learning_rate=learning_rate)

    board_history = []  # 历史棋局

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:

        sess.run([tf.global_variables_initializer()])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # 构造初始训练数据
        while len(board_history) < BOARD_HISTORY_BASE_NUM:
            game.reset()
            while True:
                available_pos = get_available_pos(game.get_board())
                action = agent.Agent.trans_pos(np.random.choice(available_pos), BOARD_SIZE)
                game.make_move(action)
                if game.is_over():
                    game.display()
                    break
                board = game.get_board().copy()
                if game.get_actor() == 1:
                    board = board[:,:,::-1]
                board_history.append(board)

        game_list = [engine.GomokuEngine(board_size=BOARD_SIZE) for _ in range(BATCH_SIZE)]

        for step in range(ITER_NUM):
            train_data = []
            train_label = []

            # 每次迭代进行一次完整的游戏模拟，之后从历史数据中找到预测结果

            # 一次完整的游戏模拟
            game.reset()
            one_game_board = []             # 棋局
            one_game_action = []            # 行为
            one_game_score_map = []         # 预测结果
            one_game_reward = []            # 奖励

            while True:

                board = game.get_board()
                if game.get_actor() == 1:
                    board = board[:, :, ::-1]
                action, score, score_map = player.make_predict(board, sess)

                one_game_board.append(board.copy())
                one_game_score_map.append(score_map)

                if np.random.rand() < 0.1:
                    available_pos = get_available_pos(game.get_board())
                    action = agent.Agent.trans_pos(np.random.choice(available_pos), BOARD_SIZE)

                one_game_action.append(action)
                game.make_move(action)
                reward = get_reward(game)
                one_game_reward.append(reward)

                if game.is_over():
                    game.display()
                    break

            # 构造样本
            one_game_train_data = []
            one_game_train_label = []
            step_num = len(one_game_board)
            for _st in range(step_num):
                board = one_game_board[_st]
                action = one_game_action[_st]
                reward = one_game_reward[_st]
                score_map = one_game_score_map[_st]
                one_game_train_data.append(board)
                if _st < step_num - 2:
                    Q = reward - GAMMA1 * one_game_reward[_st + 1] + GAMMA2 * one_game_reward[_st + 2]
                elif _st == (step_num - 2):
                    # 倒数第二步
                    Q = reward - GAMMA1 * one_game_reward[_st + 1]
                else:
                    # 最后一步
                    Q = reward
                score_map[action] = LAMBDA * score_map[action] + (1 - LAMBDA) * Q
                one_game_train_label.append(score_map)

            # 注意，这里默认我们每次一局产生的数据小于BATCH的大小
            one_game_train_step_count = len(one_game_train_label)
            assert one_game_train_step_count < BATCH_SIZE

            choice_idx = np.random.randint(0, len(board_history), BATCH_SIZE - one_game_train_step_count)
            history_train_data = [board_history[k] for k in choice_idx]
            actions, scores, score_maps = player.make_predict(history_train_data, sess)

            rival_game_idx = []
            history_reward = np.zeros((BATCH_SIZE - one_game_train_step_count, 3))
            for game_idx in range(len(history_train_data)):
                game = game_list[game_idx]
                game.reset(BOARD_SIZE, history_train_data[game_idx], copy=True)
                game.make_move(action)
                reward = get_reward(game)
                history_reward[game_idx][0] = reward
                if not game.is_over():
                    rival_game_idx.append(game_idx)

            # 第二步
            hist_2rd_data = [game_list[k].get_board()[:,:,::-1] for k in rival_game_idx]
            rival_actions, rival_scores, _ = player.make_predict(hist_2rd_data, sess)

            my_game_idx = []

            for idx in range(len(rival_game_idx)):
                game_idx = rival_game_idx[idx]
                game = game_list[game_idx]
                game.make_move(rival_actions[idx])
                history_reward[game_idx, 1] = get_reward(game)
                if not game.is_over():
                    my_game_idx.append(game_idx)
            # 第三步
            hist_3nd_data = [game_list[k].get_board() for k in my_game_idx]
            my_actions, my_scores, _ = player.make_predict(hist_3nd_data, sess)
            for idx in range(len(my_game_idx)):
                game_idx = my_game_idx[idx]
                game = game_list[game_idx]
                game.make_move(my_actions[idx])
                history_reward[game_idx, 2] = get_reward(game)

            history_train_label = score_maps
            for idx in range(len(history_train_label)):
                Q = history_reward[idx, 0] - GAMMA1 * history_reward[idx, 1] + GAMMA2 * history_reward[idx, 2]
                history_train_label[actions[idx]] = LAMBDA * history_train_label[actions[idx]] \
                                                    + (1 - LAMBDA) * Q
            board_history.extend(one_game_train_data)
            for _ in range(len(board_history) - BOARD_HISTORY_LIMIT):
                board_history.pop(0)

            one_game_train_data.extend(history_train_data)
            one_game_train_label.extend(history_train_label)

            player.train(one_game_train_data, one_game_train_label, sess, display=True)















        #
        #
        # for step in range(ITER_NUM):
        #     if (step % SAVE_INTERVAL) == 0:
        #         saver.save(sess, './models/gomoku', global_step=step)
        #
        #     train_data = []
        #     train_label = []
        #
        #
        #
        #     while True:
        #
        #
        #         game.reset()
        #         actor = 0
        #
        #         while True:
        #
        #             if np.random.rand() < 0.1:
        #                 available_pos = get_available_pos(game.get_board())
        #                 action = agent.Agent.trans_pos(np.random.choice(available_pos), BOARD_SIZE)
        #                 game.make_move(action)
        #                 actor = 1 - actor
        #                 continue
        #
        #                 max_predict_index, _, _ = player.make_predict(game.get_board(), sess)
        #
        #
        #
        #
        #
        #
        #
        #     while len(train_data[0]) < BATCH_SIZE or len(train_data[1]) < BATCH_SIZE:
        #
        #         # 开始一局
        #         game.reset_game()
        #         print('game: {}'.format(game_idx))
        #
        #         # 选择先手
        #         curr_actor = np.random.randint(2)
        #
        #         while True:
        #             # 当前选手下棋
        #             if np.random.rand() < 0.1:
        #                 # 随机下
        #                 avai = pos_map[game.available == 0]
        #                 _action = np.random.choice(avai)
        #                 action = (_action // WIDTH, _action % WIDTH)
        #                 game.make_move(action=action, actor=curr_actor)
        #                 if game_idx % 100 == 0:
        #                     print('game: {}'.format(game_idx))
        #                     print(display_game(game.board))
        #                 if game.is_win(actor=curr_actor) or len(avai) == 1:
        #                     break
        #                 curr_actor = 1 - curr_actor
        #                 continue
        #
        #             # 正常下
        #             action, max_score, raw_score_map = agents[curr_actor].make_predict(game, sess)
        #
        #             my_input_board = game.board.copy()
        #             game.make_move(action=action, actor=curr_actor)
        #             if game_idx % 100 == 0:
        #                 print('game: {}'.format(game_idx))
        #                 print(display_game(game.board))
        #
        #             # 计算奖励
        #             reward = game.get_reward(actor=curr_actor,
        #                                      win_reward=WIN_REWARD,
        #                                      loss_reward=LOSS_REWARD,
        #                                      even_reward=EVEN_REWARD,
        #                                      keep_reward=KEEP_REWARD)
        #
        #             if reward[curr_actor] == WIN_REWARD or reward[curr_actor] == EVEN_REWARD:
        #                 # 获胜，则说明这一步下的很正确，因此为了鼓励该结果，优化当前步的得分为奖励值
        #                 # 平局也是相同的操作
        #                 label = raw_score_map.copy()
        #                 label[action] = LABMDA * label[action] + (1 - LABMDA) * reward[curr_actor]
        #                 train_data[curr_actor].append(my_input_board)
        #                 train_label[curr_actor].append(label)
        #                 break
        #
        #             # 没有结束对局，说明游戏还需要继续，这时候的该步棋的评价十分复杂
        #             # 首先，如果棋局对于对手来说很棘手，则说明我们下的好，反之亦然
        #             # 同时，在对手下完一步之后，我们仍能够有比较好的局势
        #
        #             # 对手的策略
        #
        #             # 保存当前棋局
        #             curr_board = game.board.copy()
        #             curr_available = game.available.copy()
        #             curr_board_size = game.board_size
        #
        #             rival_action, rival_max_score, rival_raw_score_map = agents[1 - curr_actor].make_predict(game, sess)
        #
        #             game.make_move(action=rival_action, actor=1 - curr_actor)
        #             # 计算对手的奖励
        #             rival_reward = game.get_reward(actor=1 - curr_actor,
        #                                            win_reward=WIN_REWARD,
        #                                            loss_reward=LOSS_REWARD,
        #                                            even_reward=EVEN_REWARD,
        #                                            keep_reward=KEEP_REWARD)
        #
        #             if rival_reward[curr_actor] == LOSS_REWARD or rival_reward[curr_actor] == EVEN_REWARD:
        #                 # 我方的两种终结的状态：失败或者和局，该步的奖励就是所得到的奖励
        #                 label = raw_score_map.copy()
        #                 label[action] = LABMDA * label[action] + (1 - LABMDA) * rival_reward[curr_actor]
        #                 train_data[curr_actor].append(my_input_board)
        #                 train_label[curr_actor].append(label)
        #                 # 状态重置回刚下一步的状态，把下棋的权利让给对方
        #                 game.reset_status(curr_board_size, curr_board, curr_available)
        #                 curr_actor = 1 - curr_actor
        #                 continue
        #
        #             # 游戏还在继续，我方再下一步
        #             fur_action, fur_max_score, fur_raw_score_map = agents[curr_actor].make_predict(game, sess)
        #
        #             # 根据增强学习的迭代规则
        #             # 总奖励 = 本回自己的奖励 + 敌方的状态 + 下一步的状态
        #             label = raw_score_map.copy()
        #             Q = reward[curr_actor] - GAMMA1 * rival_max_score + GAMMA2 * fur_max_score
        #             label[action] = LABMDA * label[action] + (1 - LABMDA) * Q
        #             # print(label[action], raw_score_map[action])
        #             train_data[curr_actor].append(my_input_board)
        #             train_label[curr_actor].append(label)
        #
        #             # 状态重置回刚下一步的状态，把下棋的权利让给对方
        #             game.reset_status(curr_board_size, curr_board, curr_available)
        #             curr_actor = 1 - curr_actor
        #
        #         game_idx = game_idx + 1
        #
        #     # 通过上述的不断模拟，我们可以得到大量的训练数据，随后对两个代理进行训练
        #     for idx in range(2):
        #         choice = np.random.choice(len(train_data[idx]), BATCH_SIZE)
        #         train_data_batch = np.array(train_data[idx])[choice]
        #         train_label_batch = np.array(train_label[idx])[choice]
        #         agents[idx].fit(train_data_batch=train_data_batch, train_label_batch=train_label_batch, sess=sess,
        #                         display=True)


if __name__ == '__main__':
    main()
