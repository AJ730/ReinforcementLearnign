import random
import sys

import numpy as np

NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 500
DEFAULT_DISCOUNT = 0.9
EPSILON = 0.7
LEARNINGRATE = 0.1


class QLearner():
    """
    Q-learning agent
    num_states = env.observation_space
    num_action = env.discount.space
    """

    def __init__(self, num_states, num_actions, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE, adaptive_exploitation = False):
        self.done = False
        self.name = "agent1"
        self.observation_space = num_states
        self.action_space = num_actions
        self.discount = discount
        self.learning_rate = learning_rate
        self.Q = np.zeros([num_states, num_actions])
        self.episodes = NUM_EPISODES
        self.episode_length = MAX_EPISODE_LENGTH
        self.episilon = EPSILON
        self.timesteps = []
        self.reward = -10
        self.adaptive_exploitation = adaptive_exploitation
        self.goal_dictionary = []


    def reset_episode(self):
        """
        Here you can update some of the statistics that could be helpful to maintain
        """

        pass

    def process_experience(self, state, action, next_state, reward, done):
        """
        Update the Q-value based on the state, action, next state and reward.
        """



        self.done = done

        if done:
            self.reward = reward
            self.Q[state, action] = (1 - self.learning_rate) * self.Q[state, action] + self.learning_rate * reward
            if (self.adaptive_exploitation):
                self.episilon -= 0.00001
            return

        self.Q[state, action] = (1 - self.learning_rate) * self.Q[state, action] + \
                                 self.learning_rate * (
                                         reward + self.discount * np.max(self.Q[next_state]))



        #
        # print(self.Q)

    def select_action(self, state):
        """
        Returns an action, selected based on the current state
        """

        if np.all(np.array(self.Q[state, :]) == np.array(self.Q[state, :])[0]):
            return self.get_random_action()

        if random.random()  < self.episilon:
            return self.get_random_action()
        else:
            return self.get_best_action(state)

    def get_random_action(self):
        return np.random.choice(self.action_space)

    def get_best_action(self, state):
        return np.argmax(self.Q[state])

    def report(self, steps, episode):
        """
        Function to print useful information, printed during the main loop
        """

        print(self.reward)
        if self.done and self.reward == 10:
            self.goal_dictionary.append((episode, steps))
            self.timesteps.append(steps)


    def get_timestamp(self):
        return self.timesteps


    def get_goal_dictionary(self):
        return self.goal_dictionary
