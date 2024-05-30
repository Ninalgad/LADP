import numpy as np
from base import PreTransform


class DenoisingAutoEncoder(PreTransform):
    def __init__(self, sd=1):
        super(DenoisingAutoEncoder, self).__init__('DAE')
        self.sd = sd

    def apply(self, image, mask, dist):
        background = image <= -6.
        y = np.random.normal(scale=self.sd, size=image.shape)

        noisy_image = image.copy() + y.copy()
        noisy_image[background] = -6.
        y[background] = -6.

        return noisy_image, y
