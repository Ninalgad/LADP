import numpy as np
from medpy.io import load
from torch.utils.data import Dataset
from scipy import ndimage
from utils import one_hot2dist


MAP_STATISTICS = {"1ADC_ss": (1350.2495, 428.13467), "2Z_ADC": (0.34669298, 2.487756)}


class ImageDataset(Dataset):
    def __init__(self, img_ids, data_dir, inp_size, transform=None):
        self.x, self.y, self.d = load_data_bonbidhie2023(img_ids, data_dir, inp_size)
        self.n = len(self.x)
        self.transform = transform

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        batch = {}
        image, label, dist = self.x[idx], self.y[idx], self.d[idx]

        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)

            image = transformed["image"]
            label = transformed["mask"]
            del transformed

        batch['image'] = np.transpose(image, (2, 0, 1))
        batch['label'] = np.clip(np.transpose(label, (2, 0, 1)), 0, 1)
        batch['dist'] = np.transpose(dist, (2, 0, 1))
        return batch


class DenosingDataset(Dataset):
    def __init__(self, img_ids, data_dir, inp_size, noising_transform, transform=None):
        self.x, self.y, self.d = load_data_bonbidhie2023(img_ids, data_dir, inp_size)
        self.noise_transform = noising_transform
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        image, label = self.x[idx], self.y[idx]
        transformed_image = self.transform(image=image)['image']
        image = np.clip(transformed_image, -6, image.max())

        image, noise = self.noise_transform.apply(image, label, dist=self.d[idx])

        if self.transform is not None:
            transformed = self.transform(image=image, mask=noise)
            image = transformed["image"]
            noise = transformed["mask"]
            del transformed

        batch = dict()
        batch['image'] = np.transpose(image, (2, 0, 1))
        batch['label'] = np.transpose(noise, (2, 0, 1))
        batch['dist'] = np.transpose(self.d[idx], (2, 0, 1))
        return batch


def preprocess(img, input_type=None, target_size=None):
    if target_size is not None:
        # mask prevents blending with the background after scaling
        mask = img != 0

        # resize to 'target_size'
        n = img.shape[-1]
        s = target_size / img.shape[1]
        img = ndimage.zoom(img, (s, s, 1), cval=0.0)
        mask = ndimage.zoom(mask.astype('uint8'), (s, s, 1), cval=0.0)

        # ensure background is constant
        img = img * mask.astype(img.dtype)

        assert n == img.shape[-1]

    # set channels last by default
    img = np.transpose(img, [2, 0, 1])

    if input_type is not None:
        # normalise using precomputed stats
        m, s = MAP_STATISTICS[input_type]
        idx = img != 0
        img[idx] = (img[idx] - m) / pow(s, 1.2)

        # mask out background values to -6
        idx = np.logical_not(idx)
        img[idx] = -6
        img = np.clip(img, -6, 6)

    img = np.expand_dims(img, -1)
    return img


# loading and preprocessing
def load_inputs(idx, data_dir, target_size=None, return_meta=False, channels_first=False):
    ss_adc = data_dir / f"BONBID2023_Train/1ADC_ss/MGHNICU_{idx}-VISIT_01-ADC_ss.mha"
    zadc = data_dir / f"BONBID2023_Train/2Z_ADC/Zmap_MGHNICU_{idx}-VISIT_01-ADC_smooth2mm_clipped10.mha"

    ss_adc, _ = load(ss_adc)
    ss_adc = preprocess(ss_adc, '1ADC_ss', target_size)

    zadc, h = load(zadc)
    zadc = preprocess(zadc, '2Z_ADC', target_size)

    img = np.concatenate([ss_adc, zadc, zadc], axis=-1)
    if channels_first:
        img = np.transpose(img, (0, 3, 1, 2))

    if return_meta:
        return img, h
    return img


def load_label(idx, data_dir, target_size=None, return_meta=False, channels_first=False):
    label = data_dir / f"BONBID2023_Train/3LABEL/MGHNICU_{idx}-VISIT_01_lesion.mha"

    label, h = load(label)
    label = preprocess(label, target_size=target_size)
    if channels_first:
        label = np.transpose(label, (0, 3, 1, 2))
    if return_meta:
        return label, h
    return label


def load_data_bonbidhie2023(img_ids, data_dir, inp_size):
    # load and preprocess
    x = [load_inputs(x, data_dir, inp_size) for x in img_ids]
    y = [load_label(x, data_dir, inp_size) for x in img_ids]
    d = [one_hot2dist(t) for t in y]

    # concat
    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    d = np.concatenate(d, axis=0)

    return x, y, d
