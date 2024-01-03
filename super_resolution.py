import torch, json, os, gc
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from utils.imagedata import SpectrumImageDataset, collate_fn
from utils.spectrum import read_spectrum, save_spectrum, denorm_spectrum
from utils.trainer import DBPNTrainer
from dbpn.dbpn_iterative import Net as DBPNITERNet
from dbpn.dbpn_iterative_pool import Net as DBPNITERNetPool
from argparse import ArgumentParser

parser = ArgumentParser(
    """
    =================================================================
            Super-resolution of spectrum mapping data toolkit
             2DMaterials platform (https://2dmat.chemdx.org)
    =================================================================
    This script implements a Deep Back-Projection Networks (DBPN) 
    algorithm to enhance the resolution of mapping data. The algorithm 
    is based on the paper "Deep Back-Projection Networks for Super-
    Resolution" (DOI: 10.1109/CVPR.2018.00179).
    """
)

parser.add_argument('input_file', type=str, help='Input mapping data file in .txt format.')
# Add optional arguments
parser.add_argument('--batch_size', type=int, default=8,
                    help='Batch size for processing the mapping data. Reduce it if out-of-memory error occurred.')
parser.add_argument('--target_resolution', type=int, default=200,
                    help='Target resolution of the processed data. Default is 200.')
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default=None,
                    help='Device to use for processing, \'cpu\' or \'cuda\'. If not given, it will automatically detect \'cuda\' if available.')
parser.add_argument('--upscale_factor', type=int, default=4,
                    help='Upscaling factor. Default is 4')


args = parser.parse_args()

if args.device is None:
    setattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')

model = DBPNITERNet(num_channels=3, base_filter=64, feat=256, num_stages=3, scale_factor=args.upscale_factor)
pt_dict = torch.load('./src/models/DBPN-RES-MR64-3_4x.pth', map_location='cpu')
pt_dict = {k.replace('module.',''):v for k,v in pt_dict.items()}
md_dict = model.state_dict()
md_dict.update(pt_dict)
model.load_state_dict(md_dict)
model.to(args.device)
trainer = DBPNTrainer(model=model, opt=None, residual=True, device=args.device)

dataset = SpectrumImageDataset(upscale_factor=args.upscale_factor, data_augmentation=False)

x, ys = read_spectrum(args.input_file)
for i in range(1,20):
    dataset.from_data(x, ys)
    data = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)
    out = trainer.pred(data)['None_None']
    x = out['x']
    ys = denorm_spectrum(out['pred'], out['vmin'], out['vmax'])
    m = 2 ** i
    if ys.shape[1] >= args.target_resolution:
        break

save_spectrum(x, ys, fn=args.input_file.replace('.txt', f'_{m}x.txt'))