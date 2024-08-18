import numpy as np
from scipy.ndimage import distance_transform_edt as eucl_distance


def one_hot2dist(seg: np.ndarray, resolution=(1, 1),
                 dtype='float32') -> np.ndarray:
    num_channels = seg.shape[-1]

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(num_channels):
        posmask = seg[:, :, k].astype('bool')

        if posmask.any():
            negmask = ~posmask
            res[:, :, k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
    return res
