import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .spectrum import norm_spectrum, denorm_spectrum

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
    
    im = ax.imshow(mat, **kwargs)
    if grid:
        ax.set_xticks([(i + 1) * ni / 4 for i in range(4)], labels=[])
        ax.set_yticks([(i + 1) * ni / 4 for i in range(4)], labels=[])
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

def super_resolution_summary(x, y, x_inp=None, y_inp=None, 
                             locs=[624, 634, 644, 654], intensity_range=None, 
                             cmap=mpl.cm.inferno, figsize=(4,4), fontsize=14,
                             scalebar='horizontal'):
    xs = [x]
    ys = [y]
    vmin, vmax, _ = norm_spectrum(y, num_bit=0)
    if intensity_range is not None:
        vmin, vmax = intensity_range
    if vmax - vmin > 5000:
        vmin = int(vmin / 1000) * 1000
        vmax = int(vmax / 1000) * 1000 + (1000 if vmax % 1000 != 0 else 0)

    lbls = ['SR']
    if x_inp is not None and y_inp is not None:
        xs = [x_inp, x]
        ys = [y_inp, y]
        lbls = ['Orig.', 'SR']

    f, axs = plt.subplots(len(xs), len(locs), figsize=(figsize[0]*len(locs), figsize[1]*len(xs)))
    axs = axs.reshape(len(xs), len(locs))
    for i, (_x, _y, lbl) in enumerate(zip(xs, ys, lbls)):
        _, n1, n2 = _y.shape
        axs[i, 0].set_ylabel(f'{lbl}: {n1}x{n2}', fontsize=fontsize)

        for j, loc in enumerate(locs):
            k = np.argmin(np.abs(_x - loc))
            if i == 0:
                axs[0,j].set_title(f'{_x[k]:.2f}nm', fontsize=fontsize)
            im = axs[i,j].imshow(_y[k], vmin=vmin, vmax=vmax, cmap=cmap)
    for ax in axs.reshape(-1):
        ax.set_yticks([])
        ax.set_xticks([])

    f.subplots_adjust(wspace=0.05, hspace=0.02)
    
    if isinstance(scalebar, str) and scalebar.lower().startswith('v'):
        f1, ax1 = plt.subplots(1,1,figsize=(0.3,4))
        plt.colorbar(im, cax=ax1)
        ax1.set_xticks([])
        ax1.set_ylabel('Intensity', fontsize=fontsize)
    else:
        f1, ax1 = plt.subplots(1,1,figsize=(4,0.3))
        plt.colorbar(im, cax=ax1, orientation='horizontal')
        ax1.set_yticks([])
        ax1.set_xlabel('Intensity', fontsize=fontsize)
    return f, f1

def clustering_summary(tsne_vector, labels, 
                       figsize=(12,5.5), gridspec_kw={'width_ratios':[1,1,0.05]},
                       cmap=mpl.cm.viridis, fontsize=15):
    cmap.set_under([0.7, 0.7, 0.7])
    num_clusters = np.max(labels) + 1
    bounds = np.linspace(0,num_clusters,num_clusters+1)
    f, axs = plt.subplots(1,3,figsize=figsize, gridspec_kw=gridspec_kw)
    im = axs[0].scatter(*tsne_vector.T, c=labels.reshape(-1)[labels.reshape(-1) != -1], cmap=cmap)
    im = axs[1].imshow(labels, vmin=0, vmax=num_clusters-1, cmap=cmap)
    cb = plt.colorbar(im, cax=axs[2], ticks=bounds, boundaries=bounds-0.5, extend='min')
    cb.set_label(label='Cluster', size=fontsize-1.5)
    cb.ax.set_yticklabels((bounds+1).astype(int))
    axs[0].set_title('$t$-SNE', fontsize=fontsize)
    axs[1].set_title('Sample', fontsize=fontsize)
    f.subplots_adjust(wspace=0.05)
    for ax in axs[:2]: 
        ax.set_xticks([])
        ax.set_yticks([])
    return f

def clustering_details(x, ys, tsne_vector, labels, num_example,
                       figsize=(12,4), gridspec_kw={'width_ratios':[1.5,1,1]},
                       cmap=mpl.cm.viridis, fontsize=15, random_state=100):
    num_clusters = np.max(labels) + 1
    cmap.set_under([0.7, 0.7, 0.7])
    np.random.seed(random_state)
    n1, n2 = labels.shape
    l = labels.reshape(-1)
    ys_ = ys.reshape(x.shape[0], -1).T
    f, axs = plt.subplots(num_clusters, 3, figsize=(figsize[0], figsize[1]*num_clusters), gridspec_kw=gridspec_kw)
    l_img = np.ones((n1*n2, 3)) * 0.7
    l_img[l != -1] = [0.6, 0.6, 0.6]
    vmin, vmax, _ = norm_spectrum(ys, num_bit=0)
    for i, ax in enumerate(axs):
        ax[0].set_title(f'Cluster: {i+1} / Spectrum', fontsize=fontsize)
        ax[1].set_title(f'Cluster: {i+1} / $t$-SNE', fontsize=fontsize)
        ax[2].set_title(f'Cluster: {i+1} / Sample', fontsize=fontsize)

        lidxs = np.where(l == i)[0]
        img = l_img.copy()
        img[lidxs] = cmap(i/(num_clusters-1))[:3]
        
        idxs = np.arange(len(lidxs))
        np.random.shuffle(idxs)
        idxs = sorted(idxs[:num_example], key=lambda x: ys[:, lidxs[x]//n1, lidxs[x]%n1].std())
        ax[1].scatter(*tsne_vector[l[l != -1] != i].T, color=[0.6, 0.6, 0.6])
        ax[1].scatter(*tsne_vector[l[l != -1] == i].T, color=cmap(i/(num_clusters-1)))
        ax[2].imshow(img.reshape(n1,n2,-1))
        for ax_ in ax[1:]:
            ax_.set_xticks([])
            ax_.set_yticks([])
        for j, idx in enumerate(idxs):
            ax[0].plot(x, ys_[lidxs[idx]] + j * vmax * 0.3)
            ax[0].set_ylabel('Intensity (a.u.)', fontsize=fontsize)
            ax[0].set_xlabel('Wavelength (nm)', fontsize=fontsize)
            ax[1].scatter(*tsne_vector[l[l != -1] == i][idx], s=100, edgecolor=[0,0,0], marker='D')
            ax[2].scatter(lidxs[idx]%n1, lidxs[idx]//n1, color=mpl.cm.tab10(j), s=100, edgecolor=[0,0,0], marker='D')
        ax[0].set_ylim([0, (0.7 + 0.3*num_example) * vmax])
        for j in range(num_example+2):
            ax[0].axhline(vmin + j * vmax * 0.3, ls='--', color=[0,0,0], lw=0.5)
    f.subplots_adjust(wspace=0.03)
    f.subplots_adjust(hspace=0.4)
    return f 