# coding=utf-8
import math
import os.path
import random
from collections import deque

import numpy as np
import pandas as pd
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from pysc2.lib.actions import FUNCTIONS
from sklearn.cluster import KMeans

from location_utils import transformLocation
from objects import OBJECTS
from objects import SIZES
from q_learning_table import QLearningTable

_NO_OP = FUNCTIONS.no_op.id
_SELECT_POINT = FUNCTIONS.select_point.id
_IDLE_SELECT_WORKER_POINT = FUNCTIONS.select_idle_worker.id
_BUILD_PYLON = FUNCTIONS.Build_Pylon_screen.id
_BUILD_GATEWAY = FUNCTIONS.Build_Gateway_screen.id
_BUILD_ASSIMILATOR = FUNCTIONS.Build_Assimilator_screen.id
_TRAIN_ZELOT = FUNCTIONS.Train_Zealot_quick.id
_BUILD_SUPPLY_DEPOT = FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = FUNCTIONS.Build_Barracks_screen.id
_SELECT_ARMY = FUNCTIONS.select_army.id
_ATTACK_MINIMAP = FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = FUNCTIONS.Harvest_Gather_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_POWER = features.SCREEN_FEATURES.power.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SHIELDS = features.SCREEN_FEATURES.unit_shields.index

_MINI_MAP_PLAYER_RELATIVE = features.MINIMAP_FEATURES.player_relative.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

DATA_FILE = 'basic_protoss_agent_data'

ACTION_DO_NOTHING = 'donothing'
ACTION_DIG_MINERAL = 'ACTIONDIGMINERAL'
ACTION_DIG_GASS = 'ACTIONDIGASS'
ACTION_BUILD_PYLON = 'ACTIONBUILDPYLON'
ACTION_BUILD_ASSIMILATOR = 'ACTIONBUILDASSIMILATOR'
ACTION_BUILD_GATEWAY = 'ACTIONBUILDGATEWAY'
ACTION_BUILD_CYBERNETICS_CORE = 'ACTIONBUILDCYBERNETICSCORE'
ACTION_TRAIN_PROBE = 'ACTIONTRAINPROBE'
ACTION_TRAIN_ZELOAT = 'ACTIONBUILDZELOAT'
ACTION_TRAIN_STALKER = 'ACTIONBUILDSTALKER'
ACTION_ATTACK = 'attack'
ACTION_ATTACK_TOP_LEFT = 'attack_16_16'
ACTION_ATTACK_TOP_RIGHT = 'attack_48_16'
ACTION_ATTACK_BOTTOM_LEFT = 'attack_16_48'
ACTION_ATTACK_BOTTOM_RIGHT = 'attack_43_53'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_PYLON,
    ACTION_TRAIN_PROBE,
    ACTION_BUILD_ASSIMILATOR,
    ACTION_BUILD_GATEWAY,
    ACTION_BUILD_CYBERNETICS_CORE,
    ACTION_TRAIN_ZELOAT,
    ACTION_TRAIN_STALKER,
    # ACTION_DIG_MINERAL,
    # ACTION_DIG_GASS,
    ACTION_ATTACK_TOP_LEFT,
    ACTION_ATTACK_TOP_RIGHT,
    ACTION_ATTACK_BOTTOM_LEFT,
    ACTION_ATTACK_BOTTOM_RIGHT
]

# 各種建物の個数を管理するクラスを作成する

# for mm_x in range(0, 64):
#     for mm_y in range(0, 64):
#         if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
#             smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 24))
#
KILL_UNIT_REWARD = 0.2


class BasicProtossAgent(base_agent.BaseAgent):
    def __init__(self):
        super(BasicProtossAgent, self).__init__()

        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_action = None
        self.previous_state = None

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0

        self.nx_y = None
        self.nx_x = None

        self.memory = Memory()

        self.smart_action = None
        self.move = 0
        self.x = 0
        self.y = 0

    def splitAction(self, action_id):
        smart_action = smart_actions[action_id]

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return smart_action, x, y

    def step(self, obs):

        if obs.last():
            # print(str(self.previous_state))
            reward = obs.reward  # if obs.reward is not 0 else 0.2
            self.memory.push(self.previous_state, self.previous_action, reward)

            while True:
                try:
                    if os.path.isfile(DATA_FILE + '.gz'):
                        self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

                    self.qlearn.learn(self.memory)
                    self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')
                    break
                except EOFError:
                    print("EOFError")
            self.previous_action = None
            self.previous_state = None

            self.WriteResult(reward)

            return actions.FunctionCall(_NO_OP, [])

        unit_type = obs.observation['screen'][_UNIT_TYPE]

        if obs.first():

            player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

            self.nx_y, self.nx_x = (unit_type == OBJECTS.Nexus).nonzero()

            while True:
                try:
                    if os.path.isfile(DATA_FILE + '.gz'):
                        self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

                        print(str(self.qlearn.q_table))
                        break
                except EOFError:
                    print("EOFError")

        if self.previous_action is not None and self.move == 0:
            self.memory.push(self.previous_state, self.previous_action, self.reward)
            self.reward = 0

        # current_status生成
        nx_y, nx_x = (unit_type == OBJECTS.Nexus).nonzero()
        nx_count = 1 if nx_y.any() else 0

        pylon_y, pylon_x = (unit_type == OBJECTS.Pylon).nonzero()
        pylon_count = int(round(len(pylon_y) / SIZES.Pylon))

        gw_y, gw_x = (unit_type == OBJECTS.Gateway).nonzero()
        gw_count = int(round(len(gw_y) / SIZES.Gateway))

        vespene_y, vespene_x = (unit_type == OBJECTS.VespeneGeyser).nonzero()
        vespene_geyser_count = int(math.ceil(len(vespene_y) / 97))

        assimilator_y,assimilator_x = (unit_type == OBJECTS.Assimilator).nonzero()
        assimilator_count = int(math.ceil(len(assimilator_y) / 112))

        cc_y, _ = (unit_type == OBJECTS.CyberneticsCore).nonzero()

        players = Players(obs)
        # players.print_all()

        current_state = [
            self.base_top_left,
            nx_count,
            pylon_count,
            gw_count,
            cc_y.any(),
            players.food_used_by_workers,
            min(int(players.minerals / 100), 1),
            min(int(players.vespene / 25), 1),
            min(int(players.army_count), 10),
        ]

        isChangedState = False
        for i in range(len(current_state)):

            if self.previous_state is not None and current_state[i] != self.previous_state[i]:
                isChangedState = True

        if isChangedState:
            print(current_state)

        # hot_squares = np.zeros(4)
        # enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        # for i in range(0, len(enemy_y)):
        #     y = int(math.ceil((enemy_y[i] + 1) / 32))
        #     x = int(math.ceil((enemy_x[i] + 1) / 32))
        #
        #     hot_squares[((y - 1) * 2) + (x - 1)] = 1
        #
        # if not self.base_top_left:
        #     hot_squares = hot_squares[::-1]
        #
        # for i in range(0, 4):
        #     current_state[i + 4] = hot_squares[i]

        rl_action = self.qlearn.choose_action(str(current_state))

        # 前回情報を記録
        self.previous_state = current_state
        self.previous_action = rl_action

        # actionを決定
        if self.move == 0:
            self.smart_action = smart_actions[rl_action]
            self.x = 0
            self.y = 0
            if '_' in self.smart_action:
                self.smart_action, self.x, self.y = self.smart_action.split('_')

            print("action:" + str(self.smart_action))

        # print('smart_action:{}'.format(smart_action))

        # ScreenFeatures
        player_relative = obs.observation['minimap'][_MINI_MAP_PLAYER_RELATIVE]
        power = np.array(obs.observation['screen'][_POWER], dtype='bool')
        shield = np.array(obs.observation['screen'][_SHIELDS], dtype='bool')

        if self.isBuildAction(self.smart_action):

            if self.move == 0:
                # if players.idle_worker_count > 0:
                #     return actions.FunctionCall(_IDLE_SELECT_WORKER_POINT, [_NOT_QUEUED])

                unit_y, unit_x = (unit_type == OBJECTS.Probe).nonzero()
                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]
                    self.move = 1
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            else:
                self.move = 0

            if _BUILD_PYLON in obs.observation['available_actions'] \
                    and self.smart_action == ACTION_BUILD_PYLON and pylon_count < 5:

                top, left, bottom, right = self.get_no_building_position(vespene_geyser_count, vespene_x, vespene_y)
                noBuildingArea = np.zeros([84, 84], dtype=bool)
                noBuildingArea[top:bottom, left:right] = True

                logical_and = np.logical_or(np.logical_or(noBuildingArea, shield), power)
                targets = np.where(logical_and == False)

                target = self.get_random_position(targets[0], targets[1])

                if targets is not None:
                    return actions.FunctionCall(_BUILD_PYLON, [_NOT_QUEUED, target])

            elif _BUILD_GATEWAY in obs.observation['available_actions'] \
                    and self.smart_action == ACTION_BUILD_GATEWAY and gw_count < 4:
                unit_y, unit_x = (unit_type == OBJECTS.Pylon).nonzero()
                if unit_y.any():
                    target = None

                    top, left, bottom, right = self.get_no_building_position(vespene_geyser_count, vespene_x, vespene_y)

                    xor = np.logical_xor(power, shield)
                    xor[top:bottom, left:right] = False
                    targets = np.where(xor)

                    if len(targets[0]) != 0:
                        i = random.randrange(0, len(targets[0]) - 1)
                        target = [targets[0][i], targets[1][i]]
                        # print("target ({0},{1})".format(targets[0][i], targets[1][i]))

                    if target is not None:
                        return actions.FunctionCall(_BUILD_GATEWAY, [_NOT_QUEUED, target])

            elif _BUILD_ASSIMILATOR in obs.observation['available_actions'] \
                    and self.smart_action == ACTION_BUILD_ASSIMILATOR and assimilator_count < 2:

                target = self.get_random_position(vespene_x, vespene_y)
                if target is not None:
                    return actions.FunctionCall(_BUILD_ASSIMILATOR, [_NOT_QUEUED, target])

            elif FUNCTIONS.Build_CyberneticsCore_screen.id in obs.observation['available_actions'] \
                    and self.smart_action == ACTION_BUILD_CYBERNETICS_CORE and 0 < gw_count:
                cc_y, cc_x = (unit_type == OBJECTS.CyberneticsCore).nonzero()
                if not cc_y.any():

                    # todo VesperとAssimilatorに対応できるようにする
                    top, left, bottom, right = self.get_no_building_position(vespene_geyser_count, vespene_x, vespene_y)

                    xor = np.logical_xor(power, shield)
                    xor[top:bottom, left:right] = False
                    targets = np.where(xor)

                    if len(targets[0]) != 0:
                        i = random.randrange(0, len(targets[0]) - 1)
                        target = [targets[0][i], targets[1][i]]
                        # print("target ({0},{1})".format(targets[0][i], targets[1][i]))
                        return actions.FunctionCall(FUNCTIONS.Build_CyberneticsCore_screen.id, [_NOT_QUEUED, target])
        elif self.smart_action == ACTION_DIG_GASS or self.smart_action == ACTION_DIG_MINERAL:

            # todo まだできていない
            shields_ = obs.observation['screen'][_SHIELDS]
            mineral_y, mineral_x = (shields_ == OBJECTS.MineralField).nonzero()

            if len(mineral_y) > 0:
                i = random.randrange(0, len(mineral_y))
                target = [mineral_x[i], mineral_y[i]]
                # print("target" + str(target))
                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

        elif (self.smart_action == ACTION_TRAIN_ZELOAT or self.smart_action == ACTION_TRAIN_STALKER) and 0 < gw_count:

            if self.move == 0 and gw_y.any():
                self.move = 1
                return self.select_all_action(gw_x, gw_y)
            else:
                self.move = 0

            if FUNCTIONS.Train_Stalker_quick.id in obs.observation['available_actions'] \
                    and self.smart_action == ACTION_TRAIN_STALKER:
                return actions.FunctionCall(FUNCTIONS.Train_Stalker_quick.id, [_QUEUED])

            if _TRAIN_ZELOT in obs.observation['available_actions'] \
                    and self.smart_action == ACTION_TRAIN_ZELOAT:
                return actions.FunctionCall(_TRAIN_ZELOT, [_QUEUED])

        elif self.smart_action == ACTION_TRAIN_PROBE:
            if self.move == 0 and nx_y.any():
                self.move = 1
                return self.select_all_action(nx_x, nx_y)
            else:
                self.move = 0

            if FUNCTIONS.Train_Probe_quick.id in obs.observation['available_actions']:
                return actions.FunctionCall(FUNCTIONS.Train_Probe_quick.id, [_QUEUED])

        elif self.smart_action == ACTION_ATTACK and 0 < players.army_count:

            if self.move == 0 and _SELECT_ARMY in obs.observation['available_actions']:
                self.move = 1
                return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
            else:
                self.move = 0

            if obs.observation['single_select'][0][0] != OBJECTS.Probe \
                    and (
                    len(obs.observation['multi_select']) != 0 and obs.observation['multi_select'][0][
                0] != OBJECTS.Probe) \
                    and _ATTACK_MINIMAP in obs.observation["available_actions"]:

                target_y, target_x = (player_relative == 4).nonzero()
                if target_y.any() and players.army_count > 5:
                    i = random.randrange(0, len(target_y))
                    return actions.FunctionCall(_ATTACK_MINIMAP,
                                                [_NOT_QUEUED, [min(target_x[i] + 2, 64), min(target_y[i] + 2, 64)]])
                else:
                    return actions.FunctionCall(_ATTACK_MINIMAP,
                                                [_NOT_QUEUED, transformLocation(self, int(self.x), int(self.y))])

        if self.smart_action == ACTION_DO_NOTHING:
            return actions.FunctionCall(_NO_OP, [])

        self.reward = -1
        return actions.FunctionCall(_NO_OP, [])

    @staticmethod
    def WriteResult(reward):
        f = open('result.txt', 'a')
        f.write(str(reward) + "\n")
        f.close()

    def isBuildAction(self, smart_action):
        return smart_action == ACTION_BUILD_PYLON or smart_action == ACTION_BUILD_GATEWAY \
               or smart_action == ACTION_BUILD_ASSIMILATOR or smart_action == ACTION_BUILD_CYBERNETICS_CORE

    @staticmethod
    def get_random_position(x, y):
        target = None
        if len(x) != 0:
            i = random.randrange(0, len(x))
            target = [x[i], y[i]]
        return target

    @staticmethod
    def select_all_action(x, y):
        i = random.randint(0, len(y) - 1)
        target = [x[i], y[i]]
        return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

    @staticmethod
    def get_no_building_position(vespene_geyser_count, vespene_x, vespene_y):

        # VespeneGeyserをクラスタリング
        units = []
        for i in range(0, len(vespene_y)):
            units.append((vespene_x[i], vespene_y[i]))

        kmeans = KMeans(n_clusters=vespene_geyser_count)
        kmeans.fit(units)

        # Todo Assimilatorを立てるとVespeneGeyser消える！
        if len(kmeans.cluster_centers_) >= 2:
            # 2つのVespeneGeyserの位置を特定
            vespene1_x = int(kmeans.cluster_centers_[0][0])
            vespene1_y = int(kmeans.cluster_centers_[0][1])
            vespene2_x = int(kmeans.cluster_centers_[1][0])
            vespene2_y = int(kmeans.cluster_centers_[1][1])
            top = vespene2_y if vespene1_y > vespene2_y else vespene1_y
            bottom = vespene1_y if vespene1_y > vespene2_y else vespene2_y
            right = vespene1_x if vespene1_x > vespene2_x else vespene2_x
            left = vespene2_x if vespene1_x > vespene2_x else vespene1_x

            return top, left, bottom, right

        return 0, 0, 0, 0


class Players:
    def __init__(self, obs):
        i = 0
        self.player_id = obs.observation['player'][0]
        self.minerals = obs.observation['player'][1]
        self.vespene = obs.observation['player'][2]
        self.food_used = obs.observation['player'][3]
        self.food_cap = obs.observation['player'][4]
        self.food_used_by_army = obs.observation['player'][5]
        self.food_used_by_workers = obs.observation['player'][6]
        self.idle_worker_count = obs.observation['player'][7]
        self.army_count = obs.observation['player'][8]
        self.warp_gate_count = obs.observation['player'][9]
        self.larva_count = obs.observation['player'][10]

    def print_all(self):
        print("player_id:{0}\n"
              "minerals:{1}\n"
              "vespene:{2}\n"
              "food_used:{3}\n"
              "food_cap:{4}\n"
              "food_used_by_army:{5}\n"
              "food_used_by_workers:{6}\n"
              "idle_worker_count:{7}\n"
              "army_count:{8}\n"
              "warp_gate_count:{9}\n"
              "larva_count:{10}\n".format(self.player_id, int(self.minerals / 100), int(self.vespene / 25),
                                          self.food_used, self.food_cap, self.food_used_by_army,
                                          self.food_used_by_workers,
                                          self.idle_worker_count, self.army_count, self.warp_gate_count,
                                          self.larva_count))


class Memory:
    def __init__(self, max_size=300000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward):
        self.buffer.append((str(state), action, reward))

    def pop(self):
        return self.buffer.pop()  # 最後尾のメモリを取り出す

    def len(self):
        return len(self.buffer)
