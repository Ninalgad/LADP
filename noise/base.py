from abc import ABC, abstractmethod


class NoisingTransform(ABC):
    def __init__(self, sd=1):
        self.sd = sd

    def apply(self, image, mask):
        background = image == -6
        z, y = self.add_noise(image, mask)
        z[background] = -6
        return z, y

    @abstractmethod
    def add_noise(self, image, mask):
        raise NotImplementedError("Please Implement this method")
