import math
import torch
from torch.nn.functional import softplus
from torch.distributions import constraints, TransformedDistribution, Normal
from torch.distributions.transforms import Transform, AffineTransform


class TanhTransform(Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to

    .. code-block:: python

        ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])

    However this might not be numerically stable, thus it is recommended to use `TanhTransform' instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = constraints.real
    codomain = constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/blob/master/tensorflow_probability/python/bijectors/tanh.py#L69-L80
        return 2. * (math.log(2.) - x - softplus(-2. * x))


class TanhTransformedGaussian(TransformedDistribution):
    """
    A gaussian projected through tanh

    Args:
        mu: mean
        scale: std deviation
    """
    def __init__(self, mu, scale):
        self.mu, self.scale = mu, scale
        base_dist = Normal(mu, scale)
        transforms = [TanhTransform(cache_size=1)]
        super(TanhTransformedGaussian, self).__init__(base_dist, transforms)

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        """ not implemented """
        return None

    def enumerate_support(self, expand=True):
        """ not implemented """
        pass

    def entropy(self):
        """ not implemented """
        pass


class ScaledTanhTransformedGaussian(TransformedDistribution):
    """
    ScaledTanhTransformed Gaussian
    Ensures that the probability mass is saturated between min and max
    Dream to Control: Learning Behaviors by Latent Imagination https://arxiv.org/abs/1912.01603

    Args:
        mu: mean
        scale: deviation
        min: lower bound of distribution
        max: upper bound of distribution
    """
    def __init__(self, mu, scale, min=-1.0, max=1.0):
        self.mu, self.scale = mu, scale
        self.min, self.max = min, max
        base_dist = Normal(mu, scale)
        transforms = [TanhTransform(cache_size=1), AffineTransform(loc=0, scale=(max - min)/2.0)]
        super().__init__(base_dist, transforms)

    def rsample(self, *args, **kwargs):
        sample = super().rsample(*args, **kwargs)
        eps = torch.finfo(sample.dtype).eps
        return sample.clamp(min=self.min+eps, max=self.max-eps)

    def sample(self, *args, **kwargs):
        sample = super().sample(*args, **kwargs)
        eps = torch.finfo(sample.dtype).eps
        return sample.clamp(min=self.min+eps, max=self.max-eps)

    @property
    def mean(self):
        return self.mu

    @property
    def variance(self):
        """ not accurate don't use"""
        return self.base_dist.variance()

    def enumerate_support(self, expand=True):
        """ not accurate don't use"""
        return self.base_dist.support()

    def entropy(self):
        """ not accurate don't use"""
        return self.base_dist.entropy()