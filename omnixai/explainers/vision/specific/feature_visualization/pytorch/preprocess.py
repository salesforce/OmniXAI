import torchvision
from omnixai.preprocessing.base import TransformBase


class RandomBlur(TransformBase):
    """
    Blurs image with randomly chosen Gaussian blur
    """

    def __init__(self, kernel_size, sigma=(0.1, 2.0)):
        super().__init__()
        self.transformer = torchvision.transforms.GaussianBlur(
            kernel_size, sigma=sigma)

    def fit(self, x):
        return self

    def transform(self, x):
        return self.transformer(x)

    def invert(self, x):
        raise RuntimeError("`RandomBlur` doesn't support the `invert` function.")
