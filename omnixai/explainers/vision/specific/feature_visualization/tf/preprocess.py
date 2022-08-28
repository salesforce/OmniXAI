import tensorflow as tf
from omnixai.preprocessing.base import TransformBase


class RandomBlur(TransformBase):
    """
    Blurs image with randomly chosen Gaussian blur
    """

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def fit(self, x):
        return self

    def transform(self, x):
        pass

    def invert(self, x):
        raise RuntimeError("`RandomBlur` doesn't support the `invert` function.")
