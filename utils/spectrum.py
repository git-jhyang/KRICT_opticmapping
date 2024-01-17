import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.constants import constants as C

wave_constant = C.h * C.c * 1e9 / C.e

def read_spectrum(idx, root='/mnt/DATA/2D/pl_postech/spectrum', transpose_y=True):
    if not isinstance(idx, str):
        idx = str(idx)
    if not idx.endswith('.txt'):
        idx = idx + '.txt'
    if os.path.isfile(idx):
        mat = np.loadtxt(idx).T
    else:
        mat = np.loadtxt(os.path.join(root, idx)).T
    x = mat[0]
    ys = mat[1:]
    r = np.sqrt(ys.shape[0]).astype(int)
    ys = ys.reshape(r,r,-1)
    if transpose_y:
        # channel on first dim
        ys = ys.transpose(2, 0, 1)
    return x, ys

def save_spectrum(x, ys, fn, root='/mnt/DATA/2D/pl_postech/spectrum'):
    x = x.reshape(-1, 1)
    n = x.shape[0]
    i = ys.shape.index(n)
    if i == 0:
        ys = ys.reshape(n, -1)
    else:
        ys = ys.reshape(-1, n).T
    mat = np.hstack([x, ys])
    np.savetxt(os.path.join(root, fn), mat, fmt='%.5f')

def get_mask(x1, x2):
    x1 = x1.reshape(-1)
    x2 = x2.reshape(-1)
    if x1.shape[0] > x2.shape[0]:
        v1 = x1.reshape(-1, 1)
        v2 = x2.reshape(1, -1)
    else:
        v1 = x2.reshape(-1, 1)
        v2 = x1.reshape(1, -1)
    mask = np.zeros(v1.shape[0]).astype(bool)
    mask[np.argmin(np.abs(v1 - v2), axis=0)] = True
    return mask

def norm_spectrum(ys: np.ndarray, 
                  ref: np.ndarray = None, 
                  vmin:float = None, 
                  vmax:float = None, 
                  n: int = 2000, 
                  min_count: int = 3, 
                  mode:str = 'histogram',
                  num_bit = 256,
                  extend_bit = True,
                  ):
    if ref is None:
        ref = ys.copy()
    if mode == 'histogram':
        flat = ref.reshape(-1)
        count, bins = np.histogram(flat, bins=n)
        bins = (bins[1:] + bins[:-1]) * 0.5
        vmin = bins[count > min_count][0]
        vmax = bins[count > min_count][-1]
    elif mode == 'minmax':
        vmin = vmin if vmin is not None else np.min(ref)
        vmax = vmax if vmax is not None else np.max(ref)
    else:
        raise ValueError('Not supported normalization', mode)

    if extend_bit and num_bit != 0:
        vmax = np.max([vmax, vmin + num_bit - 1])
    y_norm = np.clip((ys - vmin) / (vmax - vmin), 0, 1)
    if num_bit != 0:
        y_norm = (y_norm * (num_bit - 1)).astype(int).astype(float) / 255.0
    return vmin, vmax, y_norm

def denorm_spectrum(ys, vmin, vmax):
    return ys.copy() * (vmax - vmin) + vmin

def transpose_spectrum(ys, axis=0):
    ys = ys.squeeze()
    shape = ys.shape
    if len(shape) == 2:
        if axis != 0:
            ys = ys.T
            shape = ys.shape
        r = np.sqrt(shape[1]).astype(int)
        ys = ys.reshape(-1, r, r)
    elif len(shape) == 3:
        if axis == 2 or shape[0] == shape[1]:
            ys.transpose(2,0,1)
    elif len(shape) == 4:
        i = np.argmax(shape)
        c = np.argmin(shape)
        mask = np.ones(4, dtype=bool)
        mask[i] = False
        mask[c] = False
        r = np.where(mask)[0].tolist()
        ys = ys.transpose(i, *r, c)
    else:
        raise ValueError('Invalid data shape', ys.shape)
    return ys

def spectrum_psnrs(target, prediction, channel=0):
    if len(target.shape) != len(prediction.shape) or np.sum(np.array(target.shape) != np.array(prediction.shape)) != 0:
        raise ValueError('Input shape must be same', target.shape, prediction.shape)
    tgt = transpose_spectrum(target)
    pred = transpose_spectrum(prediction)
    if len(tgt.shape) == 4:
        tgt = tgt[..., channel]
        pred = pred[..., channel]
    return np.array([psnr(t, p, data_range=1) for t, p in zip(tgt, pred)])
    
def spectrum_ssims(target, prediction, channel=0):
    if len(target.shape) != len(prediction.shape) or np.sum(np.array(target.shape) != np.array(prediction.shape)) != 0:
        raise ValueError('Input shape must be same', target.shape, prediction.shape)

    tgt = transpose_spectrum(target)
    pred = transpose_spectrum(prediction)
    if len(tgt.shape) == 4:
        tgt = tgt[..., channel]
        pred = pred[..., channel]
    return np.array([ssim(t, p, win_size=3, data_range=1, channel_axis=None) for t, p in zip(tgt, pred)])

def spectrum_correlations(target, prediction, channel=0):
    if len(target.shape) != len(prediction.shape) or np.sum(np.array(target.shape) != np.array(prediction.shape)) != 0:
        raise ValueError('Input shape must be same', target.shape, prediction.shape)

    tgt = transpose_spectrum(target)
    pred = transpose_spectrum(prediction)
    if len(tgt.shape) == 4:
        tgt = tgt[..., channel]
        pred = pred[..., channel]
    n = tgt.shape[0]
    tgt = tgt.reshape(n, -1).T
    pred = pred.reshape(n, -1).T
    corr = np.zeros(tgt.shape[0])
    mask = (np.var(tgt, axis=1) > 1e-15) & (np.var(pred, axis=1) > 1e-15)
    corr[mask] = np.array([np.corrcoef(t, p)[0,1] for t, p in zip(tgt[mask], pred[mask])])
    return corr

wave_unit_converters = {
    'nm': {
        'nm'   : lambda x: x,
        'cm-1' : lambda x: np.piecewise(x, [x != 0, x == 0], [lambda x: 1e7/x, np.inf]),
        'eV'   : lambda x: np.piecewise(x, [x != 0, x == 0], [lambda x: wave_constant/x, np.inf]),
        'meV'  : lambda x: np.piecewise(x, [x != 0, x == 0], [lambda x: 1e3*wave_constant/x, np.inf]),
    },
    'eV': {
        'nm'   : lambda x: np.piecewise(x, [x != 0, x == 0], [lambda x: wave_constant/x, np.inf]),
        'cm-1' : lambda x: 1e7*x/wave_constant,
        'eV'   : lambda x: x,
        'meV'  : lambda x: 1e3*x,
    },
    'cm-1': {
        'nm'   : lambda x: np.piecewise(x, [x != 0, x == 0], [lambda x: 1e7/x, np.inf]),
        'cm-1' : lambda x: x,
        'eV'   : lambda x: wave_constant*x*1e-7,
        'meV'  : lambda x: wave_constant*x*1e-4,
    }
}