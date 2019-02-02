import numpy as np


def sarsa_lambda(policy, gamma, alpha, lambda_):

    def episode_fn(state, step, i, Q):
        e = np.zeros_like(Q)
        action = policy(Q, i, state)
        done = False
        while not done:
            next_state, reward, done, info = step(action)
            next_action = policy(Q, i, next_state)
            td_err = (reward + gamma * Q[next_state, next_action]
                      - Q[state, action])
            e[state, action] += 1.0
            Q += e * alpha(i) * td_err
            e *= lambda_ * gamma
            state = next_state
            action = next_action

        return Q

    return episode_fn
