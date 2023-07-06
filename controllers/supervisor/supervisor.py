# John Modl UROP Spring 2023: Towards Cooperative Intelligence in Multi-Agent Systems using Deep RL
# Adapted from the deepbots and deepworlds repositories: https://github.com/aidudezzz/deepbots
#                                                        https://github.com/aidudezzz/deepworlds
# Official citation:
#   @InProceedings{10.1007/978-3-030-49186-4_6,
#       author="Kirtas, M.
#       and Tsampazis, K.
#       and Passalis, N.
#       and Tefas, A.",
#       title="Deepbots: A Webots-Based Deep Reinforcement Learning Framework for Robotics",
#       booktitle="Artificial Intelligence Applications and Innovations",
#       year="2020",
#       publisher="Springer International Publishing",
#       address="Cham",
#       pages="64--75",
#       isbn="978-3-030-49186-4"
#   }
#----------------------------------------------------

import gym
import numpy as np
from deepbots.supervisor import CSVSupervisorEnv
from deepbots.supervisor.wrappers import KeyboardPrinter

import utilities as utils
from models.networks import DDPG
import os
OBSERVATION_SPACE = 10
ACTION_SPACE = 2

DIST_SENSORS_MM = {'min': 0, 'max': 1023}
EUCL_MM = {'min': 0, 'max': 1.5}
ACTION_MM = {'min': -1, 'max': 1}
ANGLE_MM = {'min': -np.pi, 'max': np.pi}


class FindTargetSupervisor(CSVSupervisorEnv):
    def __init__(self, robot, target):
        super(FindTargetSupervisor, self).__init__(emitter_name='emitter',
                                                   receiver_name='receiver')
        self.observations = OBSERVATION_SPACE

        self.robot_name = robot
        self.target_name = target
        self.robot = self.getFromDef(robot)
        self.target = self.getFromDef(target)
        self.find_threshold = 0.05
        self.steps = 0
        self.steps_threshold = 1500
        self.message = []
        self.should_done = False

        self.pre_distance = None
        
        self.is_solved = False
        self.distance = utils.get_distance_from_target(self.robot, self.target)
        self.count = 0

    def get_default_observation(self):
        return [0 for i in range(OBSERVATION_SPACE)]

    def get_observations(self):
        message = self.handle_receiver()
        observation = []
        self.message = []
        if message is not None:
            for i in range(len(message)):

                self.message.append(float(message[i]))

                observation.append(
                    utils.normalize_to_range(float(message[i]),
                                             DIST_SENSORS_MM['min'],
                                             DIST_SENSORS_MM['max'], 0, 1))

            distance_from_target = utils.get_distance_from_target(
                self.robot, self.target)
            self.message.append(distance_from_target)
            distance_from_target = utils.normalize_to_range(
                distance_from_target, EUCL_MM['min'], EUCL_MM['max'], 0, 1)
            observation.append(distance_from_target)

            angle_from_target = utils.get_angle_from_target(self.robot, self.target, is_abs=False)
            self.message.append(angle_from_target)
            angle_from_target = utils.normalize_to_range(angle_from_target,
                                                       ANGLE_MM['min'],
                                                       ANGLE_MM['max'], 0, 1)
            observation.append(angle_from_target)

        else:
            observation = [0 for i in range(OBSERVATION_SPACE)]

        self.observation = observation

        return self.observation

    def get_reward(self, action):
        if (self.message is None or len(self.message) == 0
                or self.observation is None):
            return 0

        rf_values = np.array(self.message[:8])
        
        reward = 0

        # (1) Take too many steps           ### Capped at 500 ###
        # if self.steps > self.steps_threshold:
        #     return -10
        # reward -= (self.steps / self.steps_threshold)

        # (2) Reward according to distance
        target_ = self.target
            
        if self.pre_distance == None:
            self.pre_distance = utils.get_distance_from_target(self.robot, target_)
        else:
            cur_distance = utils.get_distance_from_target(self.robot, target_)
            reward += (self.distance - cur_distance)
            # print('current reward', reward)
            self.pre_distance = cur_distance
            
        # (3) Find the target
        if utils.get_distance_from_target(self.robot, self.target) < self.find_threshold:
            reward = reward + 10 + (1500-self.steps)*1

        # (4) Action 1 (gas) or Action 0 (turning) should <= 1.5
        # if np.abs(action[1]) > 1.5 or np.abs(action[0]) > 1.5:
        #     if self.steps > 10:
        #         self.should_done = True
        #     print('it is too fast')
        #     return -1

        # (5) Stop or Punish the agent when the robot is getting to close to obstacle
        if np.max(rf_values) > 300:
            if self.steps > 10:
                self.should_done = True
            return -10
        elif np.max(rf_values) > 200:
            # reward -= 0.5
            return reward-(0.05*np.max(rf_values))+5
        
        return reward-self.steps*0.0001

    def is_done(self):
        self.steps += 1
        distance = utils.get_distance_from_target(self.robot, self.target)

        if distance < self.find_threshold:
            print("======== + Solved + ========")
            self.is_solved = True
            self.count = self.count + 1
            return True

        if self.steps > self.steps_threshold or self.should_done:
            return True

        return False

    def reset(self):
        self.steps = 0
        self.should_done = False
        self.pre_distance = None
        self.is_solved = False

        return super().reset()

    def get_info(self):
        pass


def create_path(path):
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)     
    else:
        print ("Successfully created the directory %s " % path)

if __name__ == '__main__':
    create_path("./models/saved/ddpg/")
    create_path("./exports/")

    supervisor_pre = FindTargetSupervisor('robot', 'target')
    supervisor_env = KeyboardPrinter(supervisor_pre)
    agent = DDPG(lr_actor=0.00025,
                lr_critic=0.00025,
                input_dims=[10],
                gamma=0.99,
                tau=0.001,
                env=supervisor_env,
                batch_size=8,
                layer1_size=400,
                layer2_size=300,
                layer3_size=200,
                n_actions=2,
                load_models=True,
                save_dir='./models/saved/ddpg/')
    # Load from checkpoint                              ### Used for testing ###
    # agent.load_models(lr_critic=0.00025, lr_actor=0.00025, 
    #                 input_dims=[10], 
    #                 layer1_size=400,
    #                 layer2_size=300, 
    #                 layer3_size=200, 
    #                 n_actions=2, 
    #                 load_dir='./models/saved/ddpg/')
    score_history = []

    # np.random.seed(0)
    n_episode = 1000
    for i in range(n_episode+1):
        done = False
        score = 0
        obs = list(map(float, supervisor_env.reset()))
        supervisor_pre.is_solved = False
        first_iter = True

        # if score_history == [] or np.mean(score_history[-50:])<0.5 or score_history[-1] < 5:
        #     print("================= TRAINING =================")
        #     while not done:
        #         if (not first_iter):
        #             act = agent.choose_action_train(obs).tolist()
        #         else:
        #             first_iter = False
        #             act = [0, 0]
        #
        #         new_state, reward, done, info = supervisor_env.step(act)
        #         # print('1,2,3,4',new_state,'\n', reward,'\n', done, info)
        #         agent.remember(obs, act, reward, new_state, int(done))
        #         agent.learn()
        #         print('single step reward',reward)
        #         score += reward
        #
        #         obs = list(map(float, new_state))
        #         # print('new_state',obs)
        # else:
        print("================= TESTING =================")
        while not done:
            if (not first_iter):
                act = agent.choose_action_test(obs).tolist()
            else:
                first_iter = False
                act = [0, 0]

            new_state, _, done, _ = supervisor_env.step(act)
            obs = list(map(float, new_state))
            

        score_history.append(score)
        fp = open("./exports/Episode-score.txt","a")
        fp.write(str(score)+'\n')
        fp.close()
        print("===== Episode", i, "score %.2f" % score,
            "50 game average %.2f" % np.mean(score_history[-50:]))
        print('succeeded episodes',supervisor_pre.count)

        agent.save_models()