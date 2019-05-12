import numpy as np


def planning(n, Q, R, T, alpha, gamma):
    Q = Q.copy()
    for _ in range(n):
        visited_states = np.where(np.sum(T, axis=(1, 2)) > 0)[0]
        state = np.random.choice(visited_states)
        actions_taken = np.where(np.sum(T[state, :, :], axis=1) > 0)[0]
        action = np.random.choice(actions_taken)
        p = T[state, action, :] / T[state, action, :].sum()
        next_state = np.random.choice(Q.shape[0], p=p)
        reward = R[state, action, next_state]
        done = np.all(T[next_state, :, :] == 0)
        td_err = (reward + gamma * Q[next_state, :].max() * (not done)
                  - Q[state, action])
        Q[state, action] += alpha * td_err

    return Q


def dyna_q(policy, gamma, alpha, n_planning):
    def episode_function(state, step, i, Q, R, T):
        done = False
        while not done:
            action = policy(Q, i, state)
            next_state, reward, done, info = step(action)
            T[state, action, next_state] += 1
            r_err = reward - R[state, action, next_state]
            R[state, action, next_state] += r_err / T[state, action, next_state]
            td_err = (reward + gamma * Q[next_state, :].max() * (not done)
                      - Q[state, action])
            Q[state, action] += alpha(i) * td_err
            state = next_state

            Q = planning(n_planning, Q, R, T, alpha(i), gamma)

        return Q, R, T

    return episode_function
