from torch.utils.data import Dataset
from .image import load_spectrum_pair, convert_to_image
from .spectrum import (transpose_spectrum, norm_spectrum, read_spectrum, 
                       get_mask, wave_unit_converters)
import torch, tqdm
import numpy as np

class SpectrumSRDataset(Dataset):
    def __init__(self, upscale_factor, include_x=True, maxlen=512, 
                 mode='histogram', x_unit=('nm','eV')):
        super(SpectrumSRDataset, self).__init__()
        self.upscale_factor = upscale_factor
        self.maxlen = maxlen
        self.mode = mode
        
        self.include_x = include_x
        self.unit_converter = wave_unit_converters[x_unit[0]][x_unit[1]]
        self.xs = {}
        self.infos = []
        self.targets = []
        self.bicubics = []

    def from_pairs(self, paired_idxs, root='/mnt/DATA/2D/pl_postech/spectrum'):
        self.xs = {}
        self.infos = []
        self.targets = []
        self.bicubics = []
        
        for inp_tag, tgt_tag in tqdm.tqdm(paired_idxs, total=len(paired_idxs), desc='Data generation...'):
            if inp_tag is None and tgt_tag is None:
                continue
            
            _x = load_spectrum_pair(inp_fn=inp_tag, tgt_fn=tgt_tag, root=root, mode=self.mode)
            if _x is None:
                continue
            xs, vmin, vmax, inp, tgt = _x
            
            _x = convert_to_image(inp=inp, tgt=tgt, upscale_factor=self.upscale_factor,
                                  channels=1, channel_stride=0)
            if _x is None: 
                continue
            _, tgt, bic = _x

            n = xs.shape[0]
            bic = bic.reshape(n,-1).T
            tgt = tgt.reshape(n,-1).T
            m = bic.shape[0]
            
            # masking and padding
            if n < self.maxlen:
                p0 = (self.maxlen - n)//2
                p1 = (self.maxlen - n)//2 + (self.maxlen - n)%2
                tgt = torch.hstack([torch.zeros((m, p0)), tgt, torch.zeros((m, p1))]).float()
                bic = torch.hstack([torch.zeros((m, p0)), bic, torch.zeros((m, p1))]).float()
                xs  = np.hstack([np.zeros(p0), xs, np.zeros(p1)])
            elif n > self.maxlen:
                vars = bic.var(dim=0).numpy()
                mask = np.zeros(n, dtype=bool)
                mask[np.argsort(vars)[-self.maxlen:]] = True
                xs = xs[mask]
                tgt = tgt[..., mask]
                bic = bic[..., mask]
            
            bic = bic.unsqueeze(1).float()
            tgt = tgt.unsqueeze(1).float()
            
            if self.include_x:
                x = torch.from_numpy(self.unit_converter(xs)).float()
                bic = torch.concat([bic, x.expand(bic.shape)], dim=1)
            
            self.targets.append(tgt)
            self.bicubics.append(bic)
            self.infos.append(np.hstack([
                    np.repeat([[str(inp_tag), str(tgt_tag), vmin, vmax]], m, axis=0),
                    np.arange(m).reshape(-1,1),
                    np.repeat(xs.reshape(1,-1), m, axis=0)
                ])
            )
            self.xs[f'{inp_tag}'] = xs
            if f'{tgt_tag}' not in self.xs.keys():
                self.xs[f'{tgt_tag}'] = xs
        self.targets = torch.vstack(self.targets)
        self.bicubics = torch.vstack(self.bicubics)
        self.infos = np.vstack(self.infos)
        
    def from_data(self, xs, ys_inp, xs_ref=None):
        self.xs = {}
        self.infos = []
        self.targets = []
        self.bicubics = []
        
        inp = ys_inp.squeeze()
        n = xs.shape[0]
        shape = inp.shape
        inp = transpose_spectrum(inp, shape.index(n))
        if xs_ref is not None:
            mask = get_mask(xs, xs_ref)
            xs = xs[mask]
            inp = inp[mask]
            n = xs.shape[0]

        vmin, vmax, inp = norm_spectrum(inp, ref=inp, mode=self.mode)

        _, tgt, bic = convert_to_image(inp, None, upscale_factor=self.upscale_factor, 
                                       channels=1, channel_stride=0)

        tgt = tgt.reshape(n, -1).T.float()
        bic = bic.reshape(n, -1).T.float()
        m = bic.shape[0]
        # masking and padding
        if n < self.maxlen:
            p0 = (self.maxlen - n)//2
            p1 = (self.maxlen - n)//2 + (self.maxlen - n)%2
            tgt = torch.hstack([torch.zeros((m, p0)), tgt, torch.zeros((m, p1))]).float()
            bic = torch.hstack([torch.zeros((m, p0)), bic, torch.zeros((m, p1))]).float()
            xs  = np.hstack([np.zeros(p0), xs, np.zeros(p1)])
        elif n > self.maxlen:
            vars = bic.var(dim=0).numpy()
            mask = np.zeros(n, dtype=bool)
            mask[np.argsort(vars)[-self.maxlen:]] = True
            xs = xs[mask]
            tgt = tgt[:, mask]
            bic = bic[:, mask]
        
        bic = bic.unsqueeze(1)
        if self.include_x:
            x = torch.from_numpy(self.unit_converter(xs)).float()
            bic = torch.concat([bic, x.expand(bic.shape)], dim=1)            
        
        self.bicubics = bic
        self.targets = tgt.unsqueeze(1)
        self.infos = np.hstack([
            np.repeat([['None', 'None', vmin, vmax]], m, axis=0),
            np.arange(m).reshape(-1,1),
            np.repeat(xs.reshape(1,-1), m, axis=0)
        ])

    def __len__(self):
        return len(self.infos)
    
    def __getitem__(self, index):
        bicubic = self.bicubics[index]
        target = self.targets[index]
        info = self.infos[index]
        
        return bicubic, target, info

class SpectrumAEDataset(Dataset):
    def __init__(self, maxlen=512, mode='histogram', include_x=False, x_unit=('nm','eV')):
        super(SpectrumAEDataset, self).__init__()
        self.maxlen = maxlen
        self.mode = mode
        
        self.include_x = include_x
        self.unit_converter = wave_unit_converters[x_unit[0]][x_unit[1]]

        self.xs = {}
        self.infos = []
        self.inputs = []
        self.targets = []

    def from_fns(self, fns, xs_ref=None, root='/mnt/DATA/2D/pl_postech/spectrum'):
        self.infos = []
        self.inputs = []
        self.targets = []
        
        for fn in tqdm.tqdm(fns, total=len(fns), desc='Data generation...'):
            xs, ys = read_spectrum(fn, root=root, transpose_y=False)
            vmin, vmax, inp = norm_spectrum(ys=ys, ref=ys, mode=self.mode)
            
            n = xs.shape[0]
            inp = torch.from_numpy(inp.reshape(-1, n)).float()
            m = inp.shape[0]
            
            mask = np.ones(n, dtype=bool)
            if xs_ref is not None and n != xs_ref.shape[0]:
                mask = get_mask(xs, xs_ref)
                xs = xs[mask]
                inp = inp[:, mask]
                m, n = inp.shape

            if n < self.maxlen:
                p0 = (self.maxlen - n)//2
                p1 = (self.maxlen - n)//2 + (self.maxlen - n)%2
                inp = torch.hstack([torch.zeros((m, p0)), inp, torch.zeros((m, p1))]).float()
                xs  = np.hstack([np.zeros(p0), xs, np.zeros(p1)])
            elif n > self.maxlen:
                vars = inp.var(dim=0).numpy()
                mask = np.zeros(n, dtype=bool)
                mask[np.argsort(vars)[-self.maxlen:]] = True
                xs = xs[mask]
                inp = inp[:, mask]

            inp = inp.unsqueeze(1)

            self.targets.append(inp)           
            if self.include_x:
                x = torch.from_numpy(self.unit_converter(xs)).float()
                inp = torch.concat([inp, x.expand(inp.shape)], dim=1)

            self.inputs.append(inp)
            self.infos.append(np.hstack([
                    np.repeat([[fn, 'None', vmin, vmax]], m, axis=0),
                    np.arange(m).reshape(m,1),
                    np.repeat(xs.reshape(1,-1), m, axis=0)
                ])
            )
        self.inputs = torch.vstack(self.inputs)
        self.targets = torch.vstack(self.targets)
        self.infos = np.vstack(self.infos)
    
    def from_data(self, xs, ys, xs_ref=None):
        self.infos = []
        self.inputs = []
        self.targets = []
        
        vmin, vmax, inp = norm_spectrum(ys, ref=ys, mode=self.mode)
        n = xs.shape[0]
        shape = ys.shape
        inp = transpose_spectrum(inp, shape.index(n))
        inp = inp.reshape(n, -1).T
        m, n = inp.shape
        
        mask = np.ones(xs.shape[0], dtype=bool)
        if xs_ref is not None:
            mask = get_mask(xs, xs_ref)
            xs = xs[mask]
            inp = inp[..., mask]
            m, n = inp.shape
    
        if n < self.maxlen:
            p0 = (self.maxlen - n)//2
            p1 = (self.maxlen - n)//2 + (self.maxlen - n)%2
            inp = torch.hstack([torch.zeros((m, p0)), inp, torch.zeros((m, p1))]).float()
            xs  = np.hstack([np.zeros(p0), xs, np.zeros(p1)])
        elif n > self.maxlen:
            vars = inp.var(dim=0).numpy()
            mask = np.zeros(n, dtype=bool)
            mask[np.argsort(vars)[-self.maxlen:]] = True
            xs = xs[mask]
            inp = inp[..., mask]

        inp = torch.from_numpy(inp).float().unsqueeze(1)
        self.targets = inp
        if self.include_x:
            x = torch.from_numpy(self.unit_converter(xs)).float()
            inp = torch.concat([inp, x.expand(inp.shape)], dim=1)
        self.inputs = inp
        self.infos = np.hstack([
                np.repeat([['None', 'None', vmin, vmax]], m, axis=0),
                np.arange(m).reshape(m,1),
                np.repeat(xs.reshape(1,-1), m, axis=0)
            ]
        )

    def __len__(self):
        return len(self.infos)
    
    def __getitem__(self, index):
        input = self.inputs[index]
        target = self.targets[index]
        info = self.infos[index]
        
        return input, target, info

def collate_fn(data):
    inps = []
    tgts = []
    infos = []
    for (inp, tgt, info) in data:
        inps.append(inp)
        tgts.append(tgt)
        infos.append(info)
    return torch.stack(inps, dim=0), torch.stack(tgts, dim=0), np.array(infos)
