import numpy as np
import matplotlib.pyplot as plt
from .spectrum import denorm_spectrum
import os, pickle

def postprocessing(data):
    vmin = data['vmin']
    vmax = data['vmax']
    x = data['x']
    pred = data['pred']
    bic = data['bic']
 
def plot_map(ax, x, ys, loc, tol=0.03, grid=None, readout='max', **kwargs):
    shape = ys.shape
    if len(shape) == 2:
        i = (np.where(np.array(shape) == x.shape[0])[0] + 1)
        if i == 0: ys = ys.T
        ni = np.sqrt(ys.shape[0]).astype(int)
    elif len(shape) == 3:
        if shape[0] != shape[1]:
            ys = ys.transpose(1, 2, 0)
        ys = ys.reshape(-1, x.shape[0])
        ni = shape[1]
    else:
        raise ValueError('Invalid shape of ys (2-dim or 3-dim)', shape)
    mask = np.abs(x - loc) < tol
    mat = eval(f'np.{readout}(ys[:, mask], axis=1).reshape(ni, ni)')
    
    im = ax.pcolormesh(mat, **kwargs)
    if grid:
        ax.set_xticks([(i + 1) * ni / 4 for i in range(1,4)], labels=[])
        ax.set_yticks([(i + 1) * ni / 4 for i in range(1,4)], labels=[])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.grid(ls='--', color=[1,1,1], lw=0.5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    return im

def plot_maxmap(ax, x, ys, xrange=None, yrange=None, grid=None, cax=None, 
                aspect_ratio=0.2, **kwargs):
    shape = ys.shape
    if len(shape) == 2:
        i = (np.where(np.array(shape) == x.shape[0])[0] + 1)
        if i == 0: ys = ys.T
        ni = np.sqrt(ys.shape[0]).astype(int)
    elif len(shape) == 3:
        if shape[0] != shape[1]:
            ys = ys.transpose(1, 2, 0)
        ys = ys.reshape(-1, x.shape[0])
        ni = shape[1]
    else:
        raise ValueError('Invalid shape of ys (2-dim or 3-dim)', shape)
    idxs = np.argmax(ys, axis=1)
    cmap = plt.get_cmap('rainbow')
    i_min = np.min(idxs)
    i_max = np.max(idxs)
    if xrange is not None:
        i_min = np.argmin(np.abs(x - xrange[0]))
        i_max = np.argmin(np.abs(x - xrange[1]))
    rgb = cmap((idxs - i_min + 1e-5)/(i_max - i_min + 1e-4))
    ys_max = np.max(ys, axis=1)
    if yrange is not None:
        y_min, y_max = yrange
    else:
        y_min = np.min(ys_max)
        y_max = np.max(ys_max)        
    if y_max - y_min > 100:
        if y_max > 1e6:
            y_max = (y_max // 1e5 + 1) * 1e5
        elif y_max > 2950:
            y_max = (y_max // 1e3 + 1) * 1e3
        else:
            y_max = (y_max // 100 + 1) * 100
        if y_min > 1e6:
            y_min = (y_min // 1e5) * 1e5
        else:
            y_min = (y_min // 100) * 100
    level = (ys_max - y_min + 1e-5)/(y_max - y_min + 1e-4)
    level[level > 1] = 1
    level[level < 0] = 0
    rgb[level < 0.5] = level[level < 0.5].reshape(-1,1) * rgb[level < 0.5] * 2
    rgb[level < 0.5, -1] = 1
    rgb[level > 0.5, -1] = 1.7 - level[level > 0.5] * 1.6
    im = ax.imshow(rgb.reshape(ni,ni,-1), **kwargs)
    
    if grid:
        ax.set_xticks([(i + 1) * ni / 4 for i in range(1,4)], labels=[])
        ax.set_yticks([(i + 1) * ni / 4 for i in range(1,4)], labels=[])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.grid(ls='--', color=[1,1,1], lw=0.5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    
    ax.invert_yaxis()
    if cax is not None:
        idxs = np.linspace(1e-5,1-1e-5,200)
        lvls = np.linspace(1e-5,1-1e-5,100)
        rgb = cmap(idxs)[:,:3]
        mat = np.vstack([[np.hstack([rgb*l,[[1]]*200]) for l in lvls],
                         [np.hstack([rgb, [[1-l*1.6/1.7]]*200]) for l in lvls]])
        xticks = np.linspace(0,200, 4)
        xticklabels = [f'{v:.0f} nm' for v in np.linspace(xrange[0], xrange[1],4)]

        if y_max - y_min > 100:
            if y_max > 1e7:
                y_max = '{:.0f}M'.format(y_max*1e-6)
            elif y_max > 1e5:
                y_max = '{:.1f}M'.format(y_max*1e-6)
            elif y_max >= 2900:
                y_max = '{:.0f}k'.format(y_max*1e-3)
            else:
                y_max = '{:.1f}k'.format(y_max*1e-3)
            if y_min >= 1e3:
                y_min = '{:.0f}k'.format(y_min*1e-3)
            elif y_min < 1:
                y_min = '0'
            else:
                y_min = '{:.1f}k'.format(y_min*1e-3)
        else:
            y_min, y_max = f'{y_min:.0f}', f'{y_max:.0f}'
        if aspect_ratio < 1:
            cax.imshow(mat, aspect=aspect_ratio, zorder=99)
            cax.set_xticks(xticks-1)
            cax.set_xticklabels(xticklabels)
            cax.set_yticks([0,200])
            cax.set_yticklabels(labels=[y_min, y_max])
        else:
            cax.imshow(mat.transpose(1,0,2), aspect=aspect_ratio, zorder=99)
            cax.set_yticks(xticks)
            cax.set_yticklabels(xticklabels)
            cax.set_xticks([0,200])
            cax.set_xticklabels(labels=[y_min, y_max])            
        cax.yaxis.tick_right()
        cax.invert_yaxis()
    return im