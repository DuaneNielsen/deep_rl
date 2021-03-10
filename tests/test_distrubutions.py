import torch
from distributions import ScaledTanhTransformedGaussian, TanhTransformedGaussian
from matplotlib import pyplot as plt
import pytest


def test_rsample_log_probs():
    mu = torch.full((1, ), fill_value=-3.0817)
    scale = torch.full((1, ), fill_value=1.5147)
    x = ScaledTanhTransformedGaussian(mu, scale, min=-2.0, max=2.0)

    for i in range(10000):
        a = x.rsample()
        p = x.log_prob(a)
        assert torch.isnan(p).any() == False


@pytest.mark.skip('test plot')
def test_plot_distribution():
    x = torch.linspace(-2.0, 2.0, 100)
    mu = torch.full((100, ), fill_value=0.5)
    scale = torch.full((100, ), fill_value=0.5)
    d = ScaledTanhTransformedGaussian(mu, scale, min=-2.0, max=2.0)
    y = torch.exp(d.log_prob(x))
    plt.plot(x, y)
    plt.show()


@pytest.mark.skip('test plot')
def test_rsample_distribution():

    mu = torch.full((400, ), fill_value=0.0)
    scale = torch.full((400, ), fill_value=0.8)
    d = ScaledTanhTransformedGaussian(mu, scale, min=-2.0, max=2.0)
    y = d.rsample()
    plt.hist(y.numpy(), bins=50)
    plt.show()