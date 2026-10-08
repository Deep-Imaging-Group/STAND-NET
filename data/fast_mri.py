import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import scipy.io as sio
from tqdm import tqdm


class FastMriDataset(Dataset):
    def __init__(self, data_root, mask_path, normalized, data_range, which_one=None) -> None:
        super().__init__()
        if normalized:
            self.norm = "ortho"
        else:
            self.norm = "backward"
        
        mask = sio.loadmat(mask_path)["mask"]
        self.mask = mask[None, :, :]

        files = os.listdir(data_root)
        start_indice = 12
        end_indice = -15

        dataset = []

        if which_one == None:
            # for file in tqdm(files[:1], desc="Reading Data"):
            for file in tqdm(files, desc="Reading Data"):
                path = os.path.join(data_root, file)
                data = sio.loadmat(path)
                image = data["image"]
                b, h, w = image.shape
                image = image[start_indice:end_indice, h//2-128:h//2+128, w//2-128:w//2+128]
                image *= data_range
                dataset.append(image)
                self.slice_num_of_sample = image.shape[0]
            
            dataset = np.concatenate(dataset, axis=0)
            self.dataset = dataset[:, None, :, :]
        else:
            sample_num, slice_num = which_one.split("_")
            sample_num = int(sample_num)
            slice_num = int(slice_num)
            for index, file in  enumerate(files):
                if index != sample_num:
                    continue
                path = os.path.join(data_root, file)
                data = sio.loadmat(path)
                image = data["image"]
                b, h, w = image.shape
                image = image[start_indice:end_indice, h//2-128:h//2+128, w//2-128:w//2+128]
                self.slice_num_of_sample = image.shape[0]
                break
            if int(slice_num) >= self.slice_num_of_sample:
                raise Exception(f"The slice number of per sample is {self.slice_num_of_sample}, but the given slice index is {slice_num}")
            image = image[slice_num:slice_num+1, None, :, :]
            self.dataset = image

    
    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        target = self.dataset[index]
        kspace = fftshift(fft2(target, norm=self.norm), axes=[-2, -1])
        zf_kspace = self.mask * kspace
        zf_img = ifft2(ifftshift(zf_kspace, axes=[-2, -1]), norm=self.norm)

        return zf_img, zf_kspace, target, self.mask



if __name__ == "__main__":
    cc_dataset = CCDataset("./data/cc/train/", "./data/mask/radial_256_256_20.dat", True)
    print(len(cc_dataset))

    input = cc_dataset[0]
    print(input[0].shape, input[1].shape, input[2].shape, input[3].shape)

    dataloader = DataLoader(cc_dataset, batch_size=4, shuffle=True)

    for input in dataloader:
        print(input[0].shape, input[1].shape, input[2].shape, input[3].shape)
        break
