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


def train():
    learning_rate = 1e-2
    steps = int(1e6)
    batch_size = 1000

    model = torch.nn.Linear(2, 1)

    m_true = torch.tensor([-1., -1.])
    b_true = 1.0

    m_vars, m_dist = normal((1,2), 0.0, 1.0)
    b_vars, b_dist = normal((1,), 0.0, 1.0)

    dists = [
        m_dist, b_dist
    ]

    params = m_vars + b_vars

    sample_data_1d = sample_data(2)

    opt = torch.optim.Adam(params=params, lr=learning_rate)

    for i in range(steps):
        opt.zero_grad()

        losses = []
        policy_vals = []

        for _ in range(batch_size):
            x_true, y_true = sample_data_1d(m_true, b_true, 1)

            m_trial = m_dist.sample((1,))[0]
            b_trial = b_dist.sample((1,))[0]

            model.weight.data = m_trial
            model.bias.data = b_trial

            y_estimate = model(x_true)

            loss = loss_fn(y_estimate, y_true)
            loss = loss.detach()
            losses.append(loss)
            policy_vals.append(m_dist.log_prob(m_trial).sum()
                               + b_dist.log_prob(b_trial).sum())

        losses = torch.tensor(losses)
        loss_mean = torch.std(losses)
        loss_std = torch.mean(losses)
        policy_loss = sum((loss - loss_mean) * p / loss_std
                          for loss, p in zip(losses, policy_vals))

        policy_loss.backward()

        if ((i + 1) % 10) == 0:
            test_x, test_y = sample_data_1d(m_true, b_true, batch_size)
            model.weight.data = m_dist.mean.data
            model.bias.data = b_dist.mean.data

            test_estimate = model(test_x)
            loss = torch.mean(loss_fn(test_estimate, test_y))
            print('loss', loss)
            print('grad', m_dist.mean.grad, b_dist.mean.grad)
            print('mean', m_dist.mean.data, b_dist.mean.data)
            print('std', m_dist.variance.data, b_dist.variance.data)

        opt.step()


def main():
    train()


if __name__ == '__main__':
    main()
