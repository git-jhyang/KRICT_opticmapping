from torch.utils.data import Dataset
from .image import augment_image, load_spectrum_pair, convert_to_image, get_neighbors, get_flow
from .spectrum import norm_spectrum, transpose_spectrum, get_mask
import torch, tqdm
import numpy as np

class SpectrumVedioDataset(Dataset):
    def __init__(self,upscale_factor, 
                 channels=3, num_frames=8, frame_stride=5, 
                 data_augmentation=False, mode='histogram'):
        super(SpectrumVedioDataset, self).__init__()
        self.augmentation = data_augmentation
        self.upscale_factor = upscale_factor
        self.channels = channels
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.mode = mode

        self.xs = {}
        self.infos = []
        self.inputs = []
        self.targets = []
        self.bicubics = []
        self.neighbors = []
        self.flows = []
        self.subset_index = []

    def from_pairs(self, paired_idxs, root='/mnt/DATA/2D/pl_postech/spectrum'):
        self.xs = {}
        self.infos = []
        self.inputs = []
        self.targets = []
        self.bicubics = []
        self.neighbors = []
        self.flows = []
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
            if R1 < 10:
                continue
            # get neighbors
            imgs_nbrs = get_neighbors(imgs=imgs_inp, num_frames=self.num_frames, frame_stride=self.frame_stride)

            # masking
            vars = imgs_inp[:, 0].reshape(xs.shape[0], -1).var(dim=1).numpy()
            hist, bins = np.histogram(vars, bins=500)
            bins = (bins[1:] + bins[:-1]) * 0.5
            mask = vars > bins[np.argmax(hist)]
            n = np.sum(mask)
            mask[np.argsort(vars)[-n-128+n%128:-n]] = True
            
            inps_np = imgs_inp.double().numpy().transpose(0,2,3,1).copy(order='C')
            nbrs_np = imgs_nbrs.double().numpy().transpose(0,1,3,4,2).copy(order='C')
            for i in np.where(mask)[0]:
                self.infos.append([inp_tag, tgt_tag, vmin, vmax, xs[i]])
                self.inputs.append(imgs_inp[i])
                self.targets.append(imgs_tgt[i])
                self.bicubics.append(imgs_bic[i])
                self.neighbors.append(imgs_nbrs[i])
                nbrs = nbrs_np[i]
                inp = inps_np[i]
                flow = np.array([get_flow(inp, nbr) for nbr in nbrs])
                self.flows.append(torch.from_numpy(flow).float())
                _list_res.append(R1)
            self.xs[f'{inp_tag}'] = xs
            if f'{tgt_tag}' not in self.xs.keys():
                self.xs[f'{tgt_tag}'] = xs

        list_res = np.array(_list_res)
        for r in np.sort(np.unique(list_res)):
            self.subset_index.append(np.where(r == list_res)[0])

    def from_data(self, x, ys_inp, xs_ref, variance_masking=False):
        self.xs = {}
        self.infos = []
        self.inputs = []
        self.targets = []
        self.bicubics = []
        self.neighbors = []
        self.flows = []
        
        _inp = ys_inp.squeeze()
        n = x.shape[0]
        shape = _inp.shape
        _inp = transpose_spectrum(_inp, shape.index(n))
        if len(shape) == 4:
            _inp = _inp[..., 0]

        vmin, vmax, _inp = norm_spectrum(_inp, ref=_inp, mode=self.mode)
        
        mask = get_mask(x, xs_ref)
        xs = xs_ref.copy()
        n = mask.shape[0]
        inp = np.zeros((n, *shape[1:]), dtype=np.float32)
        inp[mask] = _inp

        if variance_masking:
            vars = np.var(inp.reshape(n, -1), axis=1)
            hist, bins = np.histogram(vars, bins=2000)
            bins = (bins[1:] + bins[:-1]) * 0.5
            mask = vars > bins[np.argmax(hist)]
            n = np.sum(mask)
            mask[np.argsort(vars)[-n-128+n%128:-n]] = True

        _x = convert_to_image(inp, None, upscale_factor=self.upscale_factor, channels=self.channels)

        imgs_inp, imgs_tgt, imgs_bic = _x

        imgs_nbrs = get_neighbors(imgs=imgs_inp, num_frames=self.num_frames, 
                                  frame_stride=self.frame_stride)

        inps_np = imgs_inp.double().numpy().transpose(0,2,3,1).copy(order='C')
        nbrs_np = imgs_nbrs.double().numpy().transpose(0,1,3,4,2).copy(order='C')

        for i in np.where(mask)[0]:
            self.infos.append([None, None, vmin, vmax, xs[i]])
            self.inputs.append(imgs_inp[i])
            self.targets.append(imgs_tgt[i])
            self.bicubics.append(imgs_bic[i])
            self.neighbors.append(imgs_nbrs[i])
            nbrs = nbrs_np[i]
            inp = inps_np[i]
            flow = np.array([get_flow(inp, nbr) for nbr in nbrs])
            self.flows.append(torch.from_numpy(flow).float())

    def __len__(self):
        return len(self.infos)
    
    def __getitem__(self, index):
        input = self.inputs[index]
        target = self.targets[index]
        bicubic = self.bicubics[index]
        neighbor = self.neighbors[index]
        flow = self.flows[index]
        info = self.infos[index]
        
        if self.augmentation:
            input, target, bicubic, neighbor, flow = augment_image(input, target, bicubic, img_nn=neighbor, img_flow=flow)
        
        return input, target, neighbor, flow, bicubic, info

def collate_fn(data):
    inps = []
    tgts = []
    bics = []
    flows = []
    nbrs = []
    infos = []
    for (inp, tgt, nbr, flow, bic, info) in data:
        inps.append(inp)
        tgts.append(tgt)
        nbrs.append(nbr)
        flows.append(flow)
        bics.append(bic)
        infos.append(info)
    inps = torch.stack(inps, 0).float()
    tgts = torch.stack(tgts, 0).float()
    nbrs = torch.stack(nbrs, 1).float()
    flows = torch.stack(flows, 1).float()
    bics = torch.stack(bics, 0).float()
    return inps, tgts, nbrs, flows, bics, infos
