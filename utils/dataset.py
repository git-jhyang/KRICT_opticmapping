from torch.utils.data import Dataset
from .image import augment_image, convert_to_image
from .spectrum import get_mask, norm_spectrum, transpose_spectrum
import torch
import numpy as np

class BaseDataset(nn.Module):
    pass
    
