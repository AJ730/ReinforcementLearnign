import random

import numpy as np

NUM_EPISODES = 1
MAX_EPISODE_LENGTH = 500
DEFAULT_DISCOUNT = 0.9
EPSILON = 0.05
LEARNINGRATE = 0.1


class QLearner():
    """
    Q-learning agent
    num_states = env.observation_space
    num_action = env.discount.space
    """

    def __init__(self, num_states, num_actions, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE):
        self.name = "agent1"
        self.observation_space = num_states
        self.action_space = num_actions
        self.discount = discount
        self.learning_rate = learning_rate
        self.Q = np.zeros([num_states, num_actions])
        self.episodes = NUM_EPISODES
        self.episode_length = MAX_EPISODE_LENGTH
        self.episilon = EPSILON
        self.counter = 0

    def reset_episode(self):
        """
        Here you can update some of the statistics that could be helpful to maintain
        """

        pass

    def process_experience(self, state, action, next_state, reward, done):
        """
        Update the Q-value based on the state, action, next state and reward.
        """
        if done:
            self.Q[next_state, action] = (1 - self.learning_rate) * self.Q[state, action] + self.learning_rate * reward

        self.Q[next_state, action] = (1 - self.learning_rate) * self.Q[state, action] + \
                                 self.learning_rate * (
                                         reward + self.discount * np.max(self.Q[state,:]))

        print(self.Q)

    def select_action(self, state):
        """
        Returns an action, selected based on the current state
        """
        # if np.any(self.Q == 0):
        #     return self.get_random_action()

        if random.uniform(0, 1)  < self.episilon:
            return self.get_random_action()
        else:
            return self.get_best_action(state)

    def get_random_action(self):
        return np.random.choice(self.action_space)

    def get_best_action(self, state):
        return np.argmax(self.Q[state, :])

    def report(self):
        """
        Function to print useful information, printed during the main loop
        """
        print("---")
