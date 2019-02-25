def dyna_q(policy, gamma, alpha):
    def episode_function(state, step, i, Q, p):
        action = policy(Q, i, state)
        done = False
        while not done:
            next_state, reward, done, info = step(action)

        return Q, p

    return episode_function
