import numpy as np
from .spectrum import read_spectrum, norm_spectrum, denorm_spectrum
from pyflow import pyflow
import torch
from torch.nn.functional import interpolate

def image_to_spectrum(imgs, infos):
    outputs = {}
    for info, img in zip(infos, imgs):
        inp, tgt, vmin, vmax, x = info
        tag = (inp, tgt)
        if tag not in outputs.keys():
            outputs[tag] = {'x':[], 'y':[]}
        outputs[tag]['x'].append(x)
        outputs[tag]['y'].append(denorm_spectrum(img.reshape(-1), vmin, vmax))
        
    output_mats = {}
    for tag, v in outputs.items():
        x = np.array(v['x'])
        y = np.array(v['y'])
        i = np.argsort(x)
        output_mats[tag] = (x[i], y[i])

    return output_mats

def augment_image(img_inp, img_tgt, img_bic, img_nn=None, img_flow=None, flip_h=True, flip_v=True):
    if flip_h and np.random.rand() < 0.5:
        img_inp = torch.flip(img_inp, dims=[-1])
        img_tgt = torch.flip(img_tgt, dims=[-1])
        img_bic = torch.flip(img_bic, dims=[-1])
        if img_nn is not None:
            img_nn = torch.flip(img_nn, dims=[-1])
        if img_flow is not None:
            img_flow = torch.flip(img_flow, dims=[-1])
    if flip_v and np.random.rand() < 0.5:
        img_inp = torch.flip(img_inp, dims=[-2])
        img_tgt = torch.flip(img_tgt, dims=[-2])
        img_bic = torch.flip(img_bic, dims=[-2])
        if img_nn is not None:
            img_nn = torch.flip(img_nn, dims=[-2])
        if img_flow is not None:
            img_flow = torch.flip(img_flow, dims=[-2])

    return img_inp, img_tgt, img_bic, img_nn, img_flow

def load_spectrum_pair(inp_fn, tgt_fn, root, mode='histogram'):
    ys_inp = None
    if inp_fn is not None:
        x_inp, ys_inp = read_spectrum(inp_fn, root, transpose_y=True) # (n_images, R, R)
        vmin, vmax, ys_inp = norm_spectrum(ys_inp, ref=ys_inp, mode=mode)
    
    ys_tgt = None
    if tgt_fn is not None:
        x_tgt, ys_tgt = read_spectrum(tgt_fn, root, transpose_y=True)
        if inp_fn is None:        
            vmin, vmax, ys_tgt = norm_spectrum(ys_tgt, ref=ys_tgt, mode=mode)
        else:
            _, _, ys_tgt = norm_spectrum(ys_tgt, vmin=vmin, vmax=vmax, mode='minmax')
    
    if ys_inp is not None and ys_tgt is not None:
        if tuple(x_inp.shape) != tuple(x_tgt.shape):
            print('Warning: dimension mismatch btw.', (inp_fn, x_inp.shape, tgt_fn, x_tgt.shape))
            return None
        if np.sum(np.abs(x_inp - x_tgt) > 1e-4) != 0:
            print('Warning: wavelength mismatch btw.', (inp_fn, tgt_fn, np.std(x_inp - x_tgt)))
            return None

    xs = x_inp.copy() if ys_tgt is None else x_tgt.copy()
    return xs, vmin, vmax, ys_inp, ys_tgt

def convert_to_image(inp, tgt, upscale_factor, channels=3):
#    def stack_spectrum(ys, channels, channel_stride):
#        ys_0 = ys[:, np.newaxis]
#        ys_out = ys_0.copy()
#        h = (channels - 1) // 2
#        if h < 1:
#            return ys_out # (batch, 1, R, R)
#        if channel_stride is None: 
#            channel_stride = 0
#        channel_stride = np.max([0, channel_stride]).astype(int)
#        if channel_stride == 0:
#            return np.repeat(ys_out, repeats=channels, axis=1)
#        for i in range(1, h+1):
#            ys_out = np.concatenate([
#                ys_out,
#                np.vstack([ys_0[channel_stride*i:], [ys_0[-1]]*channel_stride*i]), # copy edge
#                np.vstack([[ys_0[0]]*channel_stride*i, ys_0[:-channel_stride*i]]),
#            ], axis=1) # (batch, channels, R, R)
#        return ys_out[:, :channels]
    
    # generate channel axis (batch, channels, R, R)
    if inp is not None:
        R1 = inp.shape[-1]
#        inp = stack_spectrum(inp, channels=channels, channel_stride=channel_stride)
    if tgt is not None:
        R2 = tgt.shape[-1]
#        tgt = stack_spectrum(tgt, channels=channels, channel_stride=channel_stride)
        
    # make bicubic and convert to tensor [0,1]
    if inp is None:
        R1 = R2 // upscale_factor
        tgt_imgs = torch.from_numpy(tgt).float().unsqueeze(1)
        inp_imgs = interpolate(tgt_imgs, scale_factor=1/upscale_factor, mode='bicubic')
        bic_imgs = interpolate(inp_imgs, scale_factor=upscale_factor, mode='bicubic')
    elif tgt is None:
        R2 = R1 * upscale_factor
        inp_imgs = torch.from_numpy(inp).float().unsqueeze(1)
        bic_imgs = interpolate(inp_imgs, scale_factor=upscale_factor, mode='bicubic')
        tgt_imgs = bic_imgs
    else:
        inp_imgs = torch.from_numpy(inp).float().unsqueeze(1)
        bic_imgs = interpolate(inp_imgs, scale_factor=upscale_factor, mode='bicubic')
        tgt_imgs = torch.from_numpy(tgt).float().unsqueeze(1)
    
    # check
    if R1 * upscale_factor != R2:
        print('Warning: upscale factor mismatch.', upscale_factor, R1, R2)
        return None
    if channels == 3:
        inp_imgs = torch.concat([
            inp_imgs,
            interpolate(interpolate(inp_imgs, scale_factor=upscale_factor, mode='bilinear'), scale_factor=1/upscale_factor, mode='bicubic'),
            interpolate(interpolate(inp_imgs, scale_factor=upscale_factor, mode='nearest'), scale_factor=1/upscale_factor, mode='bicubic'),
        ], dim=1)
    return inp_imgs, tgt_imgs, bic_imgs

def get_neighbors(imgs, num_frames, frame_stride):
    frame_stride = np.max([frame_stride, 1]).astype(int)
    n = len(imgs)
    h = np.max([num_frames // 2, 1])

    nbrs = []
    for i in range(n):
        nbr = []
        for j in range(-h, -h+num_frames):
            if j == 0: continue
            k = i + j * frame_stride
            k = np.max([k, 0])
            k = np.min([k, n-1])
            nbr.append(imgs[k])
        nbrs.append(torch.stack(nbr, dim=0))
    return torch.stack(nbrs, dim=0)

def get_flow(inp, nbr, 
             alpha=0.0012, # reduced for spectrum image set
             ratio=0.6, # reduced for spectrum image set
             minWidth=4, # reduced for spectrum image set
             nOuterFPIterations=4, # reduced for spectrum image set
             nInnerFPIterations=1,
             nSORIterations=30
             ):
    colType = 0 if inp.shape[-1] == 3 else 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, im2W = pyflow.coarse2fine_flow(inp, nbr, alpha, ratio, minWidth, 
                                         nOuterFPIterations, nInnerFPIterations,
                                         nSORIterations, colType)
    return np.stack([u, v], axis=0)