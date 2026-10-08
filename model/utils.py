"""
    This module stores the common network operations.
"""
import torch
import torch.nn.functional as F
from loguru import logger


def unfold(input, patch_size):
    """
        split the input image to non-overlapped patches
        Args:
            input(Tensor): [B, C, H, W], image tensor to be split
            patch(tuple): [ph, pw], the size of the patch
    """
    if input.ndim == 3:
        input = input[:, None, :, :]
    b, c, h, w = input.shape
    ph, pw = patch_size
    padding_w = pw - w % pw if w % pw != 0 else 0
    padding_h = ph - h % ph if h % ph != 0 else 0

    padded_size = [h+padding_h, w+padding_w]

    padding = (0, padding_w, 0, padding_h)
    padding_input = F.pad(input, padding, mode="replicate")

    # logger.debug("padded size of input: {}".format(padding_input.shape))

    output = F.unfold(padding_input, patch_size, stride=patch_size)

    output = torch.permute(output, (0, 2, 1)).reshape(b, -1, c, ph, pw)
    return output, padded_size   # output: [b, P, c, ph, pw], P is the number of the pathes.


def fold(input, padded_size, target_size):
    """
        merge the small non-overlaped patches into big image tensor.
        Args:
            input(Tensor):[B, L, C, W, H], L is the number of patches with size w*h.
            padding_size(tuple): [ph, pw], the padded input before unfold
            target_size(tuple): [M, N], the size of the merged targeted image.
    """
    b, L, c, ph, pw = input.shape
    input = torch.reshape(input, (b, L, -1)).permute(0, 2, 1)
    padding_output = F.fold(input, padded_size, (ph, pw), stride=(ph, pw))

    h, w = target_size
    output = padding_output[:, :, :h, :w]

    return output # [B, C, M, N]

def euclidean_distance(x, y):
    """
        Compute the similarity between patches based on the Euclidean distance.
        Args:
            x(tensor): [B, P, M]
            y(Tensor): [B, R, M]
    """
    assert x.shape[0] == y.shape[0] and x.shape[-1] == y.shape[-1]

    dist_matrix = torch.cdist(x, y)

    return 0 - dist_matrix # [B, P, R], the smaller the Euclidean Distance, more high the similarity, so take the opposite number.


def cosine_similarity(x, y):
    """
        Compute the similarity between patches based on the sosine similarity.
        Args:
            x(tensor): [B, P, M]
            y(Tensor): [B, R, M]
    """
    assert x.shape[0] == y.shape[0]

    B, P, M = x.shape
    B, R, M = y.shape

    x1 = x[:, :, None, :].expand(B, P, R, M)
    y1 = y[:, None, :, :].expand(B, P, R, M)

    cosine_similarity = F.cosine_similarity(x1, y1, dim=-1)

    return cosine_similarity


def select_most_similar_patches(x, y, dist_func, topk):
    """
        select the topk most similar pathes
        Args:
            x(Tensor): [B, P, C, W, H]
            y(Tensor): [B, R, C, W, H]
    """
    x_shape = x.shape
    y_shape = y.shape
    
    x1 = torch.reshape(x, (x_shape[0], x_shape[1], -1))# [B, P, C*W*H]
    y1 = torch.reshape(y, (y_shape[0], y_shape[1], -1)) # [B, R, C*W*H]

    if torch.is_complex(x):
        dist_matrix = dist_func(x1.abs(), y1.abs()) # [B, P, R]
    else:
        dist_matrix = dist_func(x1, y1)

    most_similar_patch_indices = torch.topk(dist_matrix, k=topk).indices # [B, P, K]

    # #  repeat way
    # repeated_y1 = y1.unsqueeze(dim=1).repeat(1, x_shape[1], 1, 1)
    # repeated_indices = most_similar_patch_indices.unsqueeze(dim=3).repeat(1, 1, 1, y1.shape[-1])

    # expand way
    b, r, s = y1.shape
    repeated_y1 = y1.unsqueeze(dim=1).expand(b, x_shape[1], r, s)
    b, p, k = most_similar_patch_indices.shape
    repeated_indices = most_similar_patch_indices.unsqueeze(dim=3).expand(b, p, k, s)

    patches = torch.gather(repeated_y1, 2, repeated_indices) # [B, P, topk, C*W*H]

    patches = patches.reshape(x_shape[0], x_shape[1], topk, x_shape[2], x_shape[3], x_shape[4])

    return patches, most_similar_patch_indices # [B, P, topk, C, W, H]

def real_to_complex(x):
    """
        Args:
            x(tensor): [B, C, H, W], float
        Return:
            output(tensor): [B, C//2, H, W], complex64
    """
    # b, _, h, w = x.shape

    # x = torch.reshape(x, (b, -1, 2, h, w))
    # x = torch.permute(x, (0, 1, 3, 4, 2)).contiguous()
    
    shape = x.shape
    new_shape = list(shape[:-3])
    new_shape.append(-1)
    new_shape.append(2)
    new_shape.append(shape[-2])
    new_shape.append(shape[-1])
    x = torch.reshape(x, new_shape)
    x = x.transpose(-3, -2).transpose(-2, -1).contiguous()

    output = torch.view_as_complex(x)

    return output


def complex_to_real(x):
    """
        Args:
            x(tensor): [B, C, H, W] or [B, H, W], complex64
        Return:
            output(tensor): [B, C*2, H, W], float
    """
    if x.dim() == 3:
        x = torch.unsqueeze(x, dim=1)
    # b, _, h, w = x.shape
    # x = torch.view_as_real(x)
    # x = torch.permute(x, (0, 1, 4, 2, 3))
    # output = torch.reshape(x, (b, -1, h, w))
    # output = output.float()

    shape = x.shape

    x = torch.view_as_real(x)
    x = x.transpose(-1, -2).transpose(-2, -3)
    new_shape = list(shape[:-3])
    new_shape.append(shape[-3] * 2)
    new_shape.append(shape[-2])
    new_shape.append(shape[-1])
    output = torch.reshape(x, new_shape)
    output = output.float()

    return output


if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt

    image = Image.open("./test_img.png")
    img = torch.from_numpy(np.array(image)).float()
    logger.debug(img.shape)
    input = torch.permute(img, (2, 0, 1)).unsqueeze(0).contiguous()

    patches, padded_size = unfold(input, (64, 64))

    fold_image = fold(patches, padded_size, input.shape[-2:])

    topk = 5
    most_similar_patches = select_most_similar_patches(patches, patches, euclidean_distance, topk)
    # most_similar_patches = select_most_similar_patch(patches, patches, cosine_similarity, topk)

    logger.debug(most_similar_patches.shape)

    num_patch = 5
    start_patch_index = 50
    for i in range(num_patch):
        for j in range(topk):
            index = i * topk + j + 1
            plt.subplot(num_patch, topk, index)
            patch = most_similar_patches[0, start_patch_index + i, j]
            # plt.imshow()
            patch_np = patch.permute(1, 2, 0).int().numpy()
            plt.imshow(patch_np)
    
    plt.show()




