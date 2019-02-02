def sarsa(policy, gamma, alpha):

    def episode_fn(state, step, i, Q):
        action = policy(Q, i, state)
        done = False
        while not done:
            next_state, reward, done, info = step(action)
            next_action = policy(Q, i, next_state)
            td_err = (reward + gamma * Q[next_state, next_action]
                      - Q[state, action])
            Q[state, action] += alpha(i) * td_err
            state = next_state
            action = next_action

        return Q

    return episode_fn
