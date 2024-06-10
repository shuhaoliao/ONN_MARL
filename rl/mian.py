from stable_baselines3.common.env_checker import check_env
import gym
from gym import spaces
import numpy as np
from scipy.integrate import solve_ivp
import multiprocessing
from stable_baselines3 import PPO
import wandb
import matplotlib.pyplot as plt
from datetime import datetime
# Single Agent


class ONNEnvsingle(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):

        # ONN parameters
        self.max_reward = 0
        self.digit_arrays = {
            0: np.array([-1.0, 1, 1, -1,
                         1, -1, -1, 1,
                         1, -1, -1, 1,
                         1, -1, -1, 1,
                         -1, 1, 1, -1]),

            1: np.array([-1.0, -1, 1, -1,
                         -1, 1, 1, -1,
                         -1, -1, 1, -1,
                         -1, -1, 1, -1,
                         -1, -1, 1, -1]),

            2: np.array([1.0, 1, 1, 1,
                         -1, -1, -1, 1,
                         1, 1, 1, 1,
                         1, -1, -1, -1,
                         1, 1, 1, 1]),

            3: np.array([1.0, 1, 1, 1,
                         -1, -1, -1, 1,
                         1, 1, 1, 1,
                         -1, -1, -1, 1,
                         1, 1, 1, 1]),

            4: np.array([1.0, -1, 1, -1,
                         1, -1, 1, -1,
                         1, 1, 1, 1,
                         -1, -1, 1, -1,
                         -1, -1, 1, -1]),

            5: np.array([1.0, 1, 1, 1,
                         1, -1, -1, -1,
                         1, 1, 1, 1,
                         -1, -1, -1, 1,
                         1, 1, 1, 1]),

            6: np.array([1.0, 1, 1, 1,
                         1, -1, -1, -1,
                         1, 1, 1, 1,
                         1, -1, -1, 1,
                         1, 1, 1, 1]),

            7: np.array([1.0, 1, 1, 1,
                         -1, -1, -1, 1,
                         -1, -1, -1, 1,
                         -1, -1, -1, 1,
                         -1, -1, -1, 1]),

            8: np.array([1.0, 1, 1, 1,
                         1, -1, -1, 1,
                         1, 1, 1, 1,
                         1, -1, -1, 1,
                         1, 1, 1, 1]),

            9: np.array([1.0, 1, 1, 1,
                         1, -1, -1, 1,
                         1, 1, 1, 1,
                         -1, -1, -1, 1,
                         1, 1, 1, 1]),
        }
        self.digit_arrays_save = {
            0: np.array([1.0, 1, 1, 1,
                         1, -1, -1, -1,
                         1, 1, 1, 1,
                         -1, -1, -1, 1,
                         1, 1, 1, 1]),  # 5

            1: np.array([1.0, 1, 1, 1,
                         -1, -1, -1, 1,
                         -1, -1, -1, 1,
                         -1, -1, -1, 1,
                         -1, -1, -1, 1]),  # 7
            2: np.array([1.0, 1, 1, 1,
                         1, -1, -1, 1,
                         1, 1, 1, 1,
                         -1, -1, -1, 1,
                         1, 1, 1, 1]),  # 9
            3: np.array([-1.0, 1, 1, -1,
                         1, -1, -1, 1,
                         1, -1, -1, 1,
                         1, -1, -1, 1,
                         -1, 1, 1, -1]),  # 0
            4: np.array([1.0, 1, 1, 1,
                         -1, -1, -1, 1,
                         1, 1, 1, 1,
                         1, -1, -1, -1,
                         1, 1, 1, 1]),  # 2
            5: np.array([-1, -1, 1, -1,
                         -1, 1, 1, -1,
                         -1, -1, 1, -1,
                         -1, -1, 1, -1,
                         -1, -1, 1, -1]),  # 1
            # 6: np.array([1, 1, 1, 1,
            #              -1, -1, -1, 1,
            #              1, 1, 1, 1,
            #              -1, -1, -1, 1,
            #              1, 1, 1, 1]),  # 3
            # 7: np.array([1, -1, 1, -1,
            #              1, -1, 1, -1,
            #              1, 1, 1, 1,
            #              -1, -1, 1, -1,
            #              -1, -1, 1, -1]),  # 4
        }
        self.t_plot = 200
        self.n = 20
        self.max_S0 = np.zeros((self.n, self.n))
        self.S_0 = np.zeros((self.n, self.n))
        for i in range(len(self.digit_arrays_save)):
            self.S_0 = self.S_0 + \
                np.outer(self.digit_arrays_save[i], self.digit_arrays_save[i])
        self.S_0 = self.S_0 * 1/20
        self.delta = 0.01 * np.pi
        self.S = np.copy(self.S_0)
        self.S_0_pre = np.copy(self.S_0)
        self.save_num = len(self.digit_arrays_save)

        # configure spaces
        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=(self.n, self.n), dtype=np.float64)
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

    def convert_to_binary(self, arr):
        assert len(arr) == self.n, "Input array must have a length of 20."

        mean_value = np.mean(arr)
        arr_binary = np.where(arr - mean_value > 0, 1, -1)

        return arr_binary

    def seed(self, seed=None):
        if seed is None:
            np.random.seed(1)
        else:
            np.random.seed(seed)

    def add_random_noise(self, arr):
        xi_0 = np.copy(arr)
        num_elements_to_change = 7
        random_numbers = np.random.uniform(-0.5,
                                           0.5, size=num_elements_to_change)
        indices = np.random.choice(
            xi_0.size, size=num_elements_to_change, replace=False)
        xi_0[indices] = xi_0[indices] + random_numbers
        return xi_0

    def _set_action(self, agent_id, action):
        # 0-19: add 0.1, 20-39: add -0.1more
        if action >= self.n:
            self.S_0[agent_id][action - self.n] -= 0.1
            self.S_0[action - self.n][agent_id] -= 0.1

        else:
            self.S_0[agent_id][action] += 0.1
            self.S_0[action][agent_id] += 0.1

    def step(self, action):
        rew_ini = 5.0
        reward = rew_ini

        self.S_0 = self.S_0 + (action + np.transpose(action))/2

        for i in range(self.save_num):
            # choose a pattern point

            self.choose_pattern(self.digit_arrays_save[i])

            changed_pattern = np.abs(
                self.digit_arrays_save[i] - self.digit_arrays_save[i][0])/2

            if np.any(np.abs(self.S) > 1000):
                reward = reward - 40
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
                # draw

                def draw_grid1(arr, ax):
                    assert len(
                        arr) == 20, "Input array must have a length of 20."

                    grid = arr.reshape(5, 4)

                    cmap = plt.cm.gray_r
                    norm = plt.Normalize(-1, 1)

                    ax.imshow(grid, cmap=cmap, norm=norm)

                    # Set the ticks and labels
                    ax.set_xticks(np.arange(0, 4, 1))
                    ax.set_yticks(np.arange(0, 5, 1))
                    ax.set_xticklabels(np.arange(1, 5, 1))
                    ax.set_yticklabels(np.arange(1, 6, 1))

                    # Draw grid lines
                    ax.set_xticks(np.arange(-0.5, 3.5, 1), minor=True)
                    ax.set_yticks(np.arange(-0.5, 4.5, 1), minor=True)
                    ax.grid(which='minor', color='black',
                            linestyle='-', linewidth=2)

                out = YOUT[-1, :] - YOUT[-1, :][0]
                for i in range(len(out)):
                    fi = out[i]
                    tmp = np.abs(fi) // np.pi % 2 + \
                        (np.abs(fi) % np.pi) / np.pi
                    reward_tmp = -np.sum(np.abs(tmp - changed_pattern[i]))
                    reward = reward + reward_tmp

        obs = np.copy(self.S_0)
        wandb.log({"reward": reward})
        if reward > self.max_reward:
            np.save('best_reward_6p_n7111.npy', self.S_0)
            self.max_reward = reward
            self.max_S0 = np.copy(self.S_0)
        if reward < -rew_ini:
            done = True
        else:
            done = False

        # print(reward)
        return obs, reward, done, {}

    def reset(self):
        self.S_0 = np.zeros((self.n, self.n))
        for i in range(self.save_num):
            self.S_0 = self.S_0 + \
                np.outer(
                    self.digit_arrays_save[i], self.digit_arrays_save[i])
        self.S_0 = self.S_0 * 1 / 20
        # print('reset')
        obs = np.copy(self.S_0)
        return obs

    def render(self, mode='human'):
        return None

    def close(self):
        return None


wandb.init(project="ONN_single_U", name="t10w-5_6p_noise7111")
env = ONNEnvsingle()
# check_env(env)
obs = env.reset()
model = PPO("MlpPolicy", env, verbose=1)
############################################
# model = PPO.load("onn_single_t10w-5_6p_noise6")
# model.set_env(env)
############################################
model.learn(total_timesteps=100_000)
model.save("onn_single_t10w-5_6p_noise7")
print('training done')
