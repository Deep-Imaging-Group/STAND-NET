import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import scipy.io as sio
from tqdm import tqdm


class Dummyataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        
        self.mask = target = np.ones((1, 64, 64), dtype=np.cfloat)

    
    def __len__(self):
        return 64

    def __getitem__(self, index):
        target = np.ones((1, 64, 64), dtype=np.complex64)
        zf_kspace = np.ones((1, 64, 64), dtype=np.complex64)
        zf_img = np.ones((1, 64, 64), dtype=np.complex64)

        return zf_img, zf_kspace, target, self.mask

