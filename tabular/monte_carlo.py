"""
This method of learning the state value function v(s) is basically straight
from the definition of v(s) as the expected reward following the policy from
s to the end of the episode. We use random simulation to collect samples to
estimate the expectation value.
"""


def monte_carlo(policy, alpha):
    def episode(state, step, i, Q):
        done = False
        experiences = []
        while not done:
            action = policy(Q, i, state)
            next_state, reward, done, info = step(action)
            experiences.append((state, action, reward, next_state, done))
            state = next_state

        states, actions, rewards, _, __ = zip(*experiences)

        for j, state in enumerate(states):
            old_Q = Q[state][actions[j]]
            Q[state][actions[j]] = old_Q + alpha(i) * (sum(rewards[j:]) - old_Q)
        return Q

    return episode
