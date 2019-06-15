import torch

device = torch.device('cpu:0')


def randn(shape, var, mean):
    return torch.randn(*shape) * var + mean


def sample_data(m, b, batch_size):
    noise_var = 0.0
    noise_mean = 0.0

    x = randn((batch_size, 1), 10, 0)
    noise = randn((batch_size, 1), noise_var, noise_mean)

    return x, m * x + b + noise


def loss_fn(estimate, true):
    return torch.pow(estimate - true, 2)


class Normal:
    def __init__(self):
        self.var = torch.tensor([10.0], requires_grad=True)
        self.mean = torch.tensor([0.0], requires_grad=True)

    def prob(self, x):
        return torch.exp(self.log_prob(x))

    def log_prob(self, x):
        diff = (x - self.mean) / self.var
        return diff * diff

    def sample(self, shape):
        return randn(shape, self.var.detach(), self.mean.detach())


def train():
    learning_rate = 1e-2
    steps = int(1e6)
    batch_size = 1000

    model = torch.nn.Linear(1, 1)

    m_true = 0.0
    b_true = 10.0

    m_dist = Normal()
    b_dist = Normal()

    params = [
        m_dist.mean, m_dist.var, b_dist.mean, b_dist.var
    ]

    opt = torch.optim.Adam(params=params, lr=learning_rate)

    for i in range(steps):
        opt.zero_grad()

        losses = []
        policy_vals = []

        for _ in range(batch_size):
            x_true, y_true = sample_data(m_true, b_true, 1)

            m_trial = m_dist.sample((1,))
            b_trial = b_dist.sample((1,))

            model.weight.data = m_trial.reshape(1, 1)
            model.bias.data = b_trial

            y_estimate = model(x_true)

            loss = loss_fn(y_estimate, y_true)
            loss = loss.detach()
            losses.append(loss)
            policy_vals.append(m_dist.log_prob(m_trial)
                               + b_dist.log_prob(b_trial))

        losses = torch.tensor(losses)
        loss_mean = torch.std(losses)
        loss_std = torch.mean(losses)
        policy_loss = - sum((x - loss_mean) * y / loss_std
                            for x, y in zip(losses, policy_vals))
        policy_loss.backward()

        if ((i + 1) % 10) == 0:
            print('loss', torch.mean(loss), 'policy', policy_loss)
            print('grad', m_dist.mean.grad, b_dist.mean.grad)
            print('mean', m_dist.mean.data, b_dist.mean.data)
            print('std', m_dist.var.data, b_dist.var.data)

        opt.step()


def main():
    train()


if __name__ == '__main__':
    main()
