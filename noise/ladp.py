import numpy as np
from base import PreTransform


class LADP(PreTransform):
    def __init__(self, s0=.8, s1=1.5, gamma=0.95):
        super(LADP, self).__init__('LADP')
        self.s0 = s0
        self.s1 = s1
        self.gamma = gamma

    def apply(self, image, mask, dist):
        # sample mixture parameters
        # snippet from: https://github.com/ZhangxinruBIT/CarveMix/blob/main/Task100_ATLASwithCarveMix/Simple_CarveMix.py
        c = np.random.beta(1, 1)  # [0,1] create distance
        c = (c - 0.5) * 2  # [-1.1]
        m = np.min(dist)
        if c > 0:
            lam = c * m / 2  # λl = -1/2|min(dis_array)|
        else:
            lam = c * m
        mask = (dist < lam).astype('float32')  # create M
        # end of snippet

        # generate noise
        background = image <= -6.
        y = np.random.normal(scale=1, size=image.shape)

        # mix noise with image
        z = y.copy()
        z = z * (1 - mask) * self.s0 + z * mask * self.s1
        noisy_image = pow(self.gamma, .5) * image.copy() + pow(1 - self.gamma, .5) * z
        noisy_image[background] = -6.
        y[background] = -6.

        return noisy_image, y
