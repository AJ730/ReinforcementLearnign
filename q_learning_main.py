import gym
import numpy as np

import simple_grid
from q_learning_skeleton import *


def act_loop(env, agent, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_episode()

        print('---episode %d---' % episode)
        renderit = False
        if episode % 10 == 0:
            renderit = True

        for t in range(MAX_EPISODE_LENGTH):
            if renderit:
                env.render()
            printing = False
            if t % 500 == 499:
                printing = True

            if printing:
                print('---stage %d---' % t)
                agent.report(t + 1, episode)
                print("state:", state)

            action = agent.select_action(state)
            new_state, reward, done, info = env.step(action)
            if printing:
                print("act:", action)
                print("reward=%s" % reward)

            agent.process_experience(state, action, new_state, reward, done)
            state = new_state
            if done:
                print("Episode finished after {} timesteps".format(t + 1))
                env.render()
                agent.report(t + 1, episode)
                break

    env.close()


# Find the best action by looping through each actions and finding the best q value
def argmax(env, V, pi, action, s, gamma):
    e = np.zeros(env.nA)
    # Go through each action
    for a in range(env.nA):
        q = 0
        P = np.array(env.P[s][a])
        (x, y) = np.shape(P)
        # Transition probability P(s' given s , a)
        for i in range(x): # iteratre through possible states
            s_ = int(P[i][1])
            p = P[i][0]
            r = P[i][2]

            q += p * (r + gamma * V[s_]) # calculate q(s give n a)
            e[a] = q

    m = np.argmax(e)
    pi[s][m] = 1 # best pi( a given s)
    action[s] = m # index with maximum value

    return pi

#This code has been adapted from https://towardsdatascience.com/value-iteration-to-solve-openai-gyms-frozenlake-6c5e7bf0a64d

def bellman_optimality_update(env, V, s, discount):
    pi = np.zeros((env.nS, env.nA)) # this is also based on the book, but we did not have the utilityu in q learning, so new method was msde
    e = np.zeros(env.nA) # action maximizing current value

    for a in range(env.nA): # iterate through all actions
        q = 0
        P = np.array(env.P[s][a])
        (x, y) = np.shape(P)

        for i in range(x): # transition probability (ps' given s, a)
            s_ = int(P[i][1])
            p = P[i][0] # Transition probability (p (s' given s, a)
            r = P[i][2]
            q += p * (r + discount * V[s_])
            e[a] = q

    m = np.argmax(e)
    pi[s][m] = 1

    value = 0
    for a in range(env.nA):  # Iterate through every possible action
        u = 0
        P = np.array(env.P[s][a])
        (x, y) = np.shape(P)
        for i in range(x): # Iterate through every possible state
            s_ = int(P[i][1])
            p = P[i][0]
            r = P[i][2]

            u += p * (r + discount * V[s_])  # Calculate state value

        value += pi[s, a] * u

    V[s] = value

    return V[s]


def value_iteration(env, gamma, theta):  # Uses psuedo code in the book
    V = np.zeros(env.nS)  # Initialize v with 0s
    while True:
        delta = 0
        for s in range(env.s):  # Go through all states
            v = V[s]
            bellman_optimality_update(env, V, s, gamma)  # Update state_value
            delta = max(delta, abs(v - V[s]))  # Convergence check

        if (delta < theta):
            break

    pi = np.zeros((env.nS, env.nA))
    action = np.zeros((env.nS))
    for s in range(env.nS):
        pi = argmax(env, V, pi, action, s, gamma)  # Find best policy

    return V, pi, action


if __name__ == "__main__":
    # env = simple_grid.DrunkenWalkEnv(map_name="walkInThePark")
    env = simple_grid.DrunkenWalkEnv(map_name="theAlley")
    num_a = env.action_space.n

    if (type(env.observation_space) == gym.spaces.discrete.Discrete):
        num_o = env.observation_space.n
    else:
        raise ("Qtable only works for discrete observations")

    discount = DEFAULT_DISCOUNT
    ql = QLearner(num_o, num_a, discount, adaptive_exploitation=True)  # <- QTable
    act_loop(env, ql, NUM_EPISODES)

    ql.get_timestamp().sort()

    dict = ql.get_goal_dictionary()
    sorted_dict = sorted(dict, key=lambda t: t[1])

    answer = [sorted_dict[0]]

    for a in answer:
        print("episode with minimum steps", a[0])

    print("minimum steps:", ql.get_timestamp()[0])

    # V, pi, action =  value_iteration(env, 0.9, 0.0001)
    # print(action.reshape(6, 8))

    V, pi, action = value_iteration(env, 0.9, 0.0001)
    print(np.reshape(action, (1, 13)))
    print(V)
