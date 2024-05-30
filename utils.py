import numpy as np
from glob import glob
from scipy.ndimage import distance_transform_edt as eucl_distance


def one_hot2dist(seg: np.ndarray, resolution=(1, 1),
                 dtype='float32') -> np.ndarray:
    K = seg.shape[-1]

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[:, :, k].astype('bool')

        if posmask.any():
            negmask = ~posmask
            res[:, :, k] = eucl_distance(negmask, sampling=resolution) * negmask \
                - (eucl_distance(posmask, sampling=resolution) - 1) * posmask
    return res


def _path2id(path):
    return path.split('_')[-3][:3]


def recover_ids(data_dir):
    train_ids = [_path2id(f) for f in glob(str(data_dir / "BONBID2023_Train/3LABEL/*.mha"))]
    train_ids = np.array(sorted(train_ids))
    return train_ids
