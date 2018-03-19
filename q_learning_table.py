import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        pd.set_option('display.expand_frame_repr', False)
        pd.set_option('line_width', 1000)
        pd.set_option('display.max_rows', 100000)
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    # def learn(self, s, a, r, s_):
    #     self.check_state_exist(s_)
    #     self.check_state_exist(s)
    #
    #     q_predict = self.q_table.ix[s, a]
    #
    #     if s_ != 'terminal':
    #         q_target = r + self.gamma * self.q_table.ix[s_, :].max()
    #     else:
    #         q_target = r  # next state is terminal
    #
    #     # update
    #     self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def learn(self, memory):
        gamma = 0.99
        alpha = 0.5
        total_reward_t = 0

        # print(str(self.q_table))

        while (memory.len() > 0):

            (state, action, reward) = memory.pop()
            self.check_state_exist(str(state))

            total_reward_t = gamma * total_reward_t  # 時間割引率をかける
            # Q関数を更新
            self.q_table.ix[str(state), action] = self.q_table.ix[str(state), action] + alpha * (
                        reward + total_reward_t - self.q_table.ix[str(state), action])

            # 無効なアクションはTotalに入れない
            if reward == -999:
                reward = 0

            total_reward_t = total_reward_t + reward  # ステップtより先でもらえた報酬の合計を更新
            # print("state:" + str(state) + ",action:" + str(action) + ",totalReward:" + str(total_reward_t))

        return self.q_table

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # print("state" + str(state))
            # append new state to q table
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))
