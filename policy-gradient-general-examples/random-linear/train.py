import torch

device = torch.device('cpu:0')


def randn(shape, var, mean):
    return torch.randn(*shape) * var + mean


def sample_data(n):
    def fn(m, b, batch_size):
        noise_var = 0.0
        noise_mean = 0.0

        x = randn((batch_size, n), 10, 0)
        noise = randn((batch_size, 1), noise_var, noise_mean)

        return x, x.matmul(m) + b + noise

    return fn


def loss_fn(estimate, true):
    return torch.pow(estimate - true, 2)


def normal(shape, mean_initial, stddev_initial):
    mean = (torch.ones(shape) * mean_initial)
    mean = mean.clone().detach().requires_grad_(True)
    stddev = (torch.ones(shape) * stddev_initial)
    stddev = stddev.clone().detach().requires_grad_(True)
    return [mean, stddev], torch.distributions.Normal(loc=mean, scale=stddev)


class Estimator:
    def __init__(self, n_features):
        self.m_vars, self.m_dist = normal((1, n_features), 0.0, 1.0)
        self.b_vars, self.b_dist = normal((1,), 0.0, 1.0)

        self.params = sum([self.m_vars, self.b_vars], [])
        self.dists = [self.m_dist, self.b_dist]

    def estimate(self, x):
        m = self.m_dist.sample((1,))[0]
        b = self.b_dist.sample((1,))[0]
        y_estimate = x.matmul(m.t()) + b

        return y_estimate, (self.m_dist.log_prob(m).sum()
                            + self.b_dist.log_prob(x).sum())

    def estimate_mean(self, x):
        m = self.m_dist.mean
        b = self.b_dist.mean
        y_estimate = x.matmul(m.t()) + b

        return y_estimate


def train():
    learning_rate = 1e-2
    steps = int(1e6)
    batch_size = 1000

    m_true = torch.tensor([-1., -1.])
    b_true = 1.0

    estimator = Estimator(2)

    sample_data_1d = sample_data(2)

    opt = torch.optim.Adam(params=estimator.params, lr=learning_rate)

    for i in range(steps):
        opt.zero_grad()

        losses = []
        policy_vals = []

        for _ in range(batch_size):
            x_true, y_true = sample_data_1d(m_true, b_true, 1)

            y_estimate, log_prob = estimator.estimate(x_true)

            loss = loss_fn(y_estimate, y_true)
            loss = loss.detach()
            losses.append(loss)
            policy_vals.append(log_prob)

        losses = torch.tensor(losses)
        loss_mean = torch.std(losses)
        loss_std = torch.mean(losses)
        policy_loss = sum((loss - loss_mean) * p / loss_std
                          for loss, p in zip(losses, policy_vals))

        policy_loss.backward()

        if ((i + 1) % 10) == 0:
            test_x, test_y = sample_data_1d(m_true, b_true, batch_size)
            test_estimate = estimator.estimate_mean(test_x)
            loss = torch.mean(loss_fn(test_estimate, test_y))
            print('loss', loss)
            print('grad', [x.grad for x in estimator.params])
            print('mean', [x.mean for x in estimator.dists])
            print('std', [x.variance for x in estimator.dists])

        opt.step()


def main():
    train()


if __name__ == '__main__':
    main()
