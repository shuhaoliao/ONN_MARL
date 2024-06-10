import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp
from ray.rllib.env import MultiAgentEnv
from gym.spaces import Box
import os

class ONNEnvMulti:
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, worker_index):
        self.worker_index = worker_index
        # ONN parameters
        self.max_reward = 0
        self.directory = './10_6_pattern'
        self.digit_arrays = {}
        for filename in os.listdir(self.directory):
            if filename.endswith('.npy'):
                array = np.load(os.path.join(self.directory, filename))
                key = int(filename[:-4])
                self.digit_arrays[key] = array.flatten()
        self.digit_arrays_save = {
            0: self.digit_arrays[5],  # 5
            1: self.digit_arrays[7],  # 7
            2: self.digit_arrays[9],  # 9
            3: self.digit_arrays[0],  # 0
            4: self.digit_arrays[2],  # 2
            5: self.digit_arrays[1],  # 1
            # 6: self.digit_arrays[3],  # 3
            # 7: self.digit_arrays[4],  # 4
            # 8: self.digit_arrays[6],  # 6
            # 9: self.digit_arrays[8],  # 8
        }
        self.num_elements_to_change = 10
        self.MAX_STEP = 1000
        self.step_count = 0
        self.t_plot = 200
        self.n = 60
        self.max_S0 = np.zeros((self.n, self.n))
        self.S_0 = np.zeros((self.n, self.n))
        for i in range(len(self.digit_arrays_save)):
            self.S_0 = self.S_0 + \
                np.outer(self.digit_arrays_save[i], self.digit_arrays_save[i])
        self.S_0 = self.S_0 / self.n
        self.delta = 0.01 * np.pi
        self.S = np.copy(self.S_0)
        self.S_0_pre = np.copy(self.S_0)
        self.save_num = len(self.digit_arrays_save)

        # configure spaces
        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=(self.n * self.n,), dtype=np.float64)
        self.observation_space = spaces.Box(
            low=-np.inf, high=+np.inf, shape=(self.n, self.n), dtype=np.float64)

    def choose_pattern(self, pattern):
        # choose a pattern point
        xi_0 = pattern
        # compute lambda
        lambda_ = np.zeros(self.n)
        for i in range(self.n):
            for j in range(self.n):
                lambda_[i] += self.S_0[i, j] * (xi_0[i] * xi_0[j])
            lambda_[i] = 1 / lambda_[i]

        # multiply S by lambda, as in ODE
        lambda_temp = np.tile(lambda_, (self.n, 1)).T
        self.S = lambda_temp * self.S_0


    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def add_random_noise(self, arr):
        xi_0 = np.copy(arr)
        random_numbers = np.random.uniform(-0.5,
                                           0.5, size=self.num_elements_to_change)
        indices = np.random.choice(
            xi_0.size, size=self.num_elements_to_change, replace=False)
        xi_0[indices] = xi_0[indices] + random_numbers
        return xi_0

    def get_obs(self):
        obs = {}
        for i in range(self.n):
            name = f'neural_{i}'
            obs_t = np.copy(self.S_0)
            #mask obs
            obs_t[i, :] = 0
            obs_t[:, i] = 0
            obs[name] = obs_t
        return obs

    def get_reward(self, rew):
        rewards = {}
        for i in range(self.n):
            name = f'neural_{i}'
            rewards[name] = rew/self.n
        return rewards

    def _set_action(self, agent_id, action):
        self.S_0[agent_id, :] += action
        self.S_0[:, agent_id] += action
        

    def step(self, actions):
        self.step_count += 1
        rew_ini = 5.0*3
        reward = rew_ini

        for i in range(self.n):
            name = f'neural_{i}'
            self._set_action(i, actions[name])

        self.S_0 = self.S_0/self.n

        for i in range(self.save_num):
            # choose a pattern point

            self.choose_pattern(self.digit_arrays_save[i])

            changed_pattern = np.abs(
                self.digit_arrays_save[i] - self.digit_arrays_save[i][0])/2

            if np.any(np.abs(self.S) > 1000):
                reward = reward - 40*3
            else:
                # set initial condition to be perturbation of the xi_0 we set before
                # xi_0 = np.array([1, 1, 1, 1, 1, -1, -1, -1, 1, 0.5, 1, 1, -1, -1, -1, 1,
                #                  1, 1, 1, 1])
                # xi_0 = self.digit_arrays[i]
                xi_0 = self.add_random_noise(self.digit_arrays_save[i])
                # compute phase profile based on initial condition
                phi_0 = 0.5 * np.pi * (xi_0 - 1)

                def odefun(t, x):
                    return np.sum(self.S * np.sin(-np.subtract.outer(x, x) + self.delta), axis=1)

                result = solve_ivp(
                    odefun, [0, self.t_plot], phi_0, t_eval=np.linspace(0, self.t_plot, 200))

                TOUT = result.t
                YOUT = result.y.T

                out = YOUT[-1, :] - YOUT[-1, :][0]
                for i in range(len(out)):
                    fi = out[i]
                    tmp = np.abs(fi) // np.pi % 2 + \
                        (np.abs(fi) % np.pi) / np.pi
                    reward_tmp = -np.sum(np.abs(tmp - changed_pattern[i]))
                    reward = reward + reward_tmp

        obs = self.get_obs()
        if reward > self.max_reward:
            np.save('/home/work1/ONN/multi_npy/s60tmp/workid_'+str(self.worker_index)+'best_reward_'+str(self.save_num)+'p_n'+str(self.num_elements_to_change)+'.npy', self.S_0)
            self.max_reward = reward
            self.max_S0 = np.copy(self.S_0)
            print('save best reward', self.max_reward)
        if reward < -rew_ini:
            done = True
        elif self.step_count > self.MAX_STEP:
            done = True
        else:
            done = False
        # print('reward', reward)
        reward = self.get_reward(reward)

        return obs, reward, {'__all__': done}, {}

    def reset(self):
        self.S_0 = np.zeros((self.n, self.n))
            for i in range(self.save_num):
                self.S_0 = self.S_0 + \
                    np.outer(
                        self.digit_arrays_save[i], self.digit_arrays_save[i])
            self.S_0 = self.S_0 / self.n
        # print('reset')
        obs = self.get_obs()
        return obs

    def render(self, mode='human'):
        return None

    def close(self):
        return None
    
class EnvProxy(MultiAgentEnv):

    def __init__(self, env_config):
        super().__init__()
        self.worker_index = -1
        self.n_neurals = 60
        if env_config is None:
            self.worker_index = -1
            self.n_neurals = 60
        else:
            self.worker_index = env_config.worker_index
            print('worker_index', self.worker_index)
            self.n_neurals = env_config['n_neural']
        
        self._agent_ids = [f'neural_{i}' for i in range(self.n_neurals)]
        self._env = ONNEnvMulti(self.worker_index)
        self.obs_space = Box(low=-np.inf, high=+np.inf, shape=(self.n_neurals , self.n_neurals), dtype=np.float64)
        self.action_space = Box(low=-0.5, high=0.5, shape=(self.n_neurals,), dtype=np.float64)

    def reset(self):
        return self._env.reset()

    def step(self, actions):
        s, r, d, i = self._env.step(actions)
        return s, r, d, i
    
_test_env =  EnvProxy(None)
obs_space = _test_env.obs_space
act_space = _test_env.action_space