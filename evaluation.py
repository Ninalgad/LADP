from data import load_inputs, load_label
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F
import numpy as np


def dice(y_pred, y_true, k=1):
    y_pred = y_pred.astype('float32')
    y_true = y_true.astype('float32')
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + k)


def evaluate(model, validation_ids, data_dir, config, device):
    model.eval()
    preds, tars = [], []
    for idx in validation_ids:

        # use the volume as the batch
        x = load_inputs(idx, data_dir=data_dir, config=config, channels_first=True)
        x = torch.tensor(x, dtype=torch.float32).to(device)
        p = F.sigmoid(model(x)).detach().cpu().numpy()
        y = load_label(idx, data_dir=data_dir, config=config, channels_first=True)

        # scale back to original size
        s = y.shape[-1] / int(config['image_size'])
        p = zoom(p, (1, 1, s, s))

        preds.append(p)
        tars.append(y)

    # find best_thresh
    best_score, best_thresh = -1, 0
    min_ = min([p.min() for p in preds])
    max_ = max([p.max() for p in preds])
    for t in np.linspace(min_, max_, num=int(config['num_thresh_sweep_steps'])):
        scores = []
        for (p, y) in zip(preds, tars):
            pt = (p > t).astype('float32')
            scores.append(dice(pt, y))

        scores_avg = np.mean(scores)
        if scores_avg > best_score:
            best_score = scores_avg
            best_thresh = t

    return best_score, best_thresh
