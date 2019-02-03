import numpy as np


def constant(x):
    def f(*_, **__):
        return x

    return f


def harmonic_sequence(x0, a, xmin):
    """
    Harmonic series of the form (x0 - xmin) / (a * i + 1) + xmin
    """

    def f(i):
        return (x0 - xmin) / (a * i + 1) + xmin

    return f


def exp_decay(x0, t, xmin):
    def f(i):
        return (x0 - xmin) * np.exp(-i * t) + xmin

    return f


def epsilon_greedy_policy_probs(Q, eps, state):
    action_size = len(Q[0])
    probs = np.ones(action_size) * eps / action_size
    probs[np.argmax(Q[state])] = 1 - eps + eps / action_size
    return probs


def epsilon_greedy(eps):
    def policy(Q, i, state):
        return np.random.choice(
            len(Q[state]),
            p=epsilon_greedy_policy_probs(Q, eps(i), state)
        )

    return policy


def softmax(x):
    x -= np.max(x)
    y = np.exp(x)
    return y / np.sum(y)


def softmax_policy(t):
    def policy(Q, i, state):
        return np.random.choice(
            len(Q[state]),
            p=softmax(Q[state] / t(i))
        )

    return policy


def moving_avg(window, y):
    avg_x = np.arange(y.shape[0] - window + 1) + window - 1
    avg = np.convolve(y, np.ones(window) / window, mode='valid')
    return avg_x, avg
