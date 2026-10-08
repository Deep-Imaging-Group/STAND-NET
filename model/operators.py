import torch
import torch.nn as nn
import torch.nn.functional as F

########################### operations ##########################################




########################### activations #########################

def activation_wrap(x: torch.Tensor, activation) -> torch.Tensor:
    is_complex = x.is_complex()
    if is_complex:
        x = torch.view_as_real(x)
    output = activation(x)
    if is_complex:
        output = torch.view_as_complex(output)
    return output

class ComplexSigmoid(nn.Sigmoid):
    def forward(self, x: torch.Tensor) -> torch.tensor:
        return activation_wrap(x, super)

class ComplexLeakyRelU(nn.LeakyReLU):
    def forward(self, x: torch.Tensor) -> torch.tensor:
        return activation_wrap(x, super().forward)


def conv2d_with_activation(in_channel, out_channel, kernel_size, stride, padding, activation=None):
    if activation is not None:
        conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            activation()
        )
    else:
        conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        )

    return conv

def conv1d_with_activation(in_channel, out_channel, kernel_size, stride, padding, activation=None):
    if activation is not None:
        conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding),
            activation()
        )
    else:
        conv = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding)
        )

    return conv

def linear_with_activation(in_channel, out_channel, activation=None):
    if activation is not None:
        conv = nn.Sequential(
            nn.Linear(in_channel, out_channel),
            activation()
        )
    else:
        conv = nn.Sequential(
            nn.Linear(in_channel, out_channel),
        )

    return conv


def kronecker_product(a: torch.tensor, b: torch.tensor, c: torch.tensor):
    """
        Args:
            a: [B, R, I]
            b: [B, R, J]
            c: [B, R, K]
    """

    a = torch.unsqueeze(a, dim=3)
    b = torch.unsqueeze(b, dim=2)

    ab = torch.matmul(a, b) # [B, R, I, J]
    ab = torch.unsqueeze(ab, dim=4) # [B, R, I, J, 1]

    c = c[:, :, None, None] # [B, R, 1, 1, K]

    output = ab * c # [B, R, I, J, K]

    return output

def attention(q: torch.tensor, k: torch.tensor, v: torch.tensor):
    """
        Args:
            Q: [B, L, I]
            K: [B, M, I]
            V: [B, M, J]
    """
    k = torch.permute(k, (0, 2, 1))
    scores = F.softmax(torch.matmul(q, k), dim=2)   #[B, L, M]
    output = torch.matmul(scores, v)    #[B, L, J]

    return output


def fft2(x, fft_norm):
    if fft_norm:
        return torch.fft.fft2(x, norm="ortho")
    else:
        return torch.fft.fft2(x)


def ifft2(x, fft_norm):
    if fft_norm:
        return torch.fft.ifft2(x, norm="ortho")
    else:
        return torch.fft.ifft2(x)





