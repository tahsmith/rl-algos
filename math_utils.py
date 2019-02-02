import numpy as np


def constant(x):
    def f(*_, **__):
        return x

    return f


def clipped_harmonic_sequence(x0, a, xmin):
    """
    Harmonic series of the form x0 / (a * i + 1)
    """
    def f(i):
        return max(
            x0 / (a * i + 1),
            xmin
        )

    return f


def clipped_exp_decay(x0, t, xmin):
    def f(i):
        return max(
            x0 * np.exp(-i * t),
            xmin
        )

    return f


def epsilon_greedy(action_size, eps):
    choices = np.arange(action_size)

    def policy_probs(Q, eps, state):
        probs = np.ones(action_size) * eps / action_size
        probs[np.argmax(Q[state])] = 1 - eps + eps / action_size
        return probs

    def policy(Q, i, state):
        return np.random.choice(choices,
                                p=policy_probs(Q, eps(i), state))

    return policy