from torch.utils.data import Dataset
from .image import augment_image, load_spectrum_pair, convert_to_image
from .spectrum import get_mask, transpose_spectrum, norm_spectrum
import torch, tqdm
import numpy as np

class SpectrumImageDataset(Dataset):
    def __init__(self, upscale_factor, data_augmentation=False, 
                 channels=3, mode='histogram'):
        
        super(SpectrumImageDataset, self).__init__()
        self.augmentation = data_augmentation
        self.upscale_factor = upscale_factor
        self.channels = channels
        self.mode = mode

        self.xs = {}
        self.infos = []
        self.inputs = []
        self.targets = []
        self.bicubics = []
        self.subset_index = []

    def from_pairs(self, paired_idxs, root='/mnt/DATA/2D/pl_postech/spectrum'):
        self.xs = {}
        self.infos = []
        self.inputs = []
        self.targets = []
        self.bicubics = []
        self.subset_index = []

        _list_res = []
        for inp_tag, tgt_tag in tqdm.tqdm(paired_idxs, total=len(paired_idxs), desc='Data generation...'):
            if inp_tag is None and tgt_tag is None:
                continue
            
            _x = load_spectrum_pair(inp_fn=inp_tag, tgt_fn=tgt_tag, root=root, mode=self.mode)
            if _x is None:
                continue
            xs, vmin, vmax, inp, tgt = _x
            
            _x = convert_to_image(inp=inp, tgt=tgt, upscale_factor=self.upscale_factor, channels=self.channels)
            if _x is None: 
                continue
            imgs_inp, imgs_tgt, imgs_bic = _x
            
            R1 = imgs_inp.shape[-1]
            
            # masking
            vars = imgs_inp[:, 0].reshape(xs.shape[0], -1).var(dim=1).numpy()
            hist, bins = np.histogram(vars, bins=2000)
            bins = (bins[1:] + bins[:-1]) * 0.5
            mask = vars > bins[np.argmax(hist)]
            n = np.sum(mask)
            mask[np.argsort(vars)[-n-128+n%128:-n]] = True

            for i in np.where(mask)[0]:
                self.infos.append([inp_tag, tgt_tag, vmin, vmax, xs[i]])
                self.inputs.append(imgs_inp[i])
                self.targets.append(imgs_tgt[i])
                self.bicubics.append(imgs_bic[i])
                _list_res.append(R1)
            self.xs[f'{inp_tag}'] = xs
            if f'{tgt_tag}' not in self.xs.keys():
                self.xs[f'{tgt_tag}'] = xs
        list_res = np.array(_list_res)
        for r in np.sort(np.unique(list_res)):
            self.subset_index.append(np.where(r == list_res)[0])

    def from_data(self, x, ys_inp, xs_ref=None, variance_masking=False):
        self.xs = {}
        self.infos = []
        self.inputs = []
        self.targets = []
        self.bicubics = []

        inp = ys_inp.squeeze()
        n = x.shape[0]
        shape = inp.shape
        inp = transpose_spectrum(inp, shape.index(n))
        if len(shape) == 4:
            inp = inp[..., 0]

        vmin, vmax, inp = norm_spectrum(inp, ref=inp, mode=self.mode)

        xs = x.copy()
        mask = np.ones(n, dtype=bool)
        if xs_ref is not None:
            mask = get_mask(x, xs_ref)
            n = mask.shape[0]
            xs = xs_ref.copy()
            inp_ = np.zeros((n, *shape[1:]), dtype=np.float32)
            inp_[mask] = inp
            inp = inp_.copy()

        if variance_masking:
            vars = np.var(inp.reshape(n, -1), axis=1)
            hist, bins = np.histogram(vars, bins=2000)
            bins = (bins[1:] + bins[:-1]) * 0.5
            mask = vars > bins[np.argmax(hist)]
            n = np.sum(mask)
            mask[np.argsort(vars)[-n-128+n%128:-n]] = True

        _x = convert_to_image(inp, None, upscale_factor=self.upscale_factor, channels=self.channels)

        imgs_inp, imgs_tgt, imgs_bic = _x
        for i in np.where(mask)[0]:
            self.infos.append([None, None, vmin, vmax, xs[i]])
            self.inputs.append(imgs_inp[i])
            self.targets.append(imgs_tgt[i])
            self.bicubics.append(imgs_bic[i])

    def __len__(self):
        return len(self.infos)
    
    def __getitem__(self, index):
        input = self.inputs[index]
        target = self.targets[index]
        bicubic = self.bicubics[index]
        info = self.infos[index]
        
        if self.augmentation:
            input, target, bicubic = augment_image(input, target, bicubic)[:3]
        return input, target, bicubic, info

def collate_fn(data):
    inps = []
    tgts = []
    bics = []
    infos = []
    for (inp, tgt, bic, info) in data:
        inps.append(inp)
        tgts.append(tgt)
        bics.append(bic)
        infos.append(info)
    return torch.stack(inps, dim=0), torch.stack(tgts, dim=0), torch.stack(bics, dim=0), infos
