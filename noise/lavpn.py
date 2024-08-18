import numpy as np
from noise.base import NoisingTransform
from noise.utils import one_hot2dist


class LAVPN(NoisingTransform):
    def __init__(self, sd=1, s0=0.08, s1=0.3):
        super(LAVPN, self).__init__(sd=sd)
        self.s0, self.s1 = s0, s1

    def apply(self, image, mask=None):
        dist = one_hot2dist(mask)
        dist = - dist.copy().astype("float32")
        dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-5)

        phi = dist * (self.s1 - self.s0) + self.s0

        epsilon = np.random.normal(scale=self.sd, size=image.shape)
        z = np.cos(phi) * image + np.sin(phi) * epsilon
        v = np.cos(phi) * epsilon.copy() - np.sin(phi) * image.copy()

        return z, v
