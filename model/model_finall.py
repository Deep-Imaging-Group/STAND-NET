# same as the codes of model_v8_cross_v2_thre 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from einops import rearrange
import numbers

from model.basemodel import BaseModel
from model.operators import conv2d_with_activation, kronecker_product, attention, fft2, ifft2
from model.utils import real_to_complex, complex_to_real, unfold, fold, select_most_similar_patches, cosine_similarity, euclidean_distance



class HQSLearnableLRModel(BaseModel):
    def __init__(self, niter, args, training) -> None:
        super().__init__(args, training)
        self.channel = 32
        dis_funcs = [euclidean_distance, cosine_similarity]
        topk = args.topk
        ph, pw = args.patch_size
        cp_in_channel = 2 * topk * len(dis_funcs)
        cp_fea_channel = cp_in_channel * 2
        cp_i = topk * len(dis_funcs)
        cp_j = ph
        cp_k = pw
        cp_num_basis = args.num_basis
        patch_size = args.patch_size

        
        blocks = []
        for index in range(niter):
            blocks.append(IterationBlock(index, cp_in_channel, cp_fea_channel, cp_i, cp_j, cp_k, cp_num_basis, patch_size, topk, dis_funcs, args.fft_norm))
        self.iteration_block = nn.ModuleList(blocks)
        self.blocks = blocks
            
    
    def forward(self, rec_img: torch.tensor, zf_kspace: torch.tensor, mask: torch.tensor):
        self.all_stages_output = []
        pre_output1 = rec_img
        pre_output2 = rec_img
        for i, operator in enumerate(self.iteration_block):
            rec_img = operator(rec_img, zf_kspace, pre_output1, mask)
            pre_output1, pre_output2 = pre_output2, rec_img
            self.all_stages_output.append(operator.x)
            self.all_stages_output.append(operator.lr_z)
            self.all_stages_output.append(operator.sp_t)
            self.all_stages_output.append(operator.enhanced_x)

        return rec_img

    def get_input_for_model(self, input, device, is_prediction):
        # zf_img = torch.squeeze(input[0], dim=1)
        zf_img = Variable(input[0]).to(device)

        # zf_kspace = torch.squeeze(input[1], dim=1)
        zf_kspace = Variable(input[1]).to(device) 

        

        # mask = torch.squeeze(input[3], dim=1)
        mask = Variable(input[3]).to(device)

        inputs = (zf_img, zf_kspace, mask)

        if is_prediction:
            return inputs
        else:
            # target = torch.squeeze(input[2], dim=1)
            target = Variable(input[2]).to(device) 
            return inputs, target
            
    
    def compute_loss(self, pred, target):
        total_loss = 0 
        weight = self._args.sparsity_loss_weight
        for index, block in enumerate(self.iteration_block):
            loss = block.compute_loss() * weight
            total_loss = total_loss + loss
        
        total_loss += self.criterion(pred.cfloat().abs(), target.cfloat().abs())

        return total_loss
    
    def get_pred_for_metric(self, pred):
        return pred.abs()
    
    def get_target_for_metric(self, target):
        return target.abs()
    
    def vis_image(self, input, pred, target):
        index = 0
        row = 4
        zf_img = torch.abs(input[0][index])
        zf_kspace = torch.log(1 + torch.abs(input[1][index]))
        mask = input[2][index]
        target = torch.abs(target[index])
        vis_input = [zf_img, mask, zf_kspace, target]
        for block in self.iteration_block:
            x, lr_z, sp_t, DT_D_x = block.get_output()
            vis_input.append(torch.abs(x[0]))
            vis_input.append(torch.abs(lr_z[0]))
            vis_input.append(torch.abs(sp_t[0]))
            vis_input.append(torch.abs(DT_D_x[0]))

        images = torch.stack(vis_input, dim=0)
        nrow = images.shape[0] // row
        b, c, h, w = images.shape
        images = torch.reshape(images, (-1, row, c, h, w))
        images = torch.permute(images, (1, 0, 2, 3, 4))
        images = torch.reshape(images, (-1, c, h, w))
        vis_input = torchvision.utils.make_grid(images.abs(), nrow=nrow, pad_value=1, padding=5, normalize=True, scale_each=True)
        self._vis.images(vis_input, nrow=nrow, padding=10, win=self.TIMESTAMP, opts={"title": self.TIMESTAMP})

class IterationBlock(nn.Module):
    def __init__(
            self, 
            index, 
            cp_in_channel, 
            cp_fea_channel, 
            cp_i, 
            cp_j, 
            cp_k, 
            cp_num_basis, 
            patch_size, 
            topk,
            dis_funcs,
            fft_norm
            ):
        super().__init__()
        self.fft_norm = fft_norm
        self.index = index
        self.patch_size = patch_size
        self.topk = topk
        # self.dis_func_num = len(dis_funcs)
        self.dis_funcs = dis_funcs
        # self.cp_num_basis = cp_num_basis
        
        self.beta_1 = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)
        self.beta_2 = nn.Parameter(torch.tensor(0.1, dtype=torch.float), requires_grad=True)
        # self.lambda_1_beta_1 = nn.Parameter(torch.tensor(0.001, dtype=torch.float), requires_grad=True)
        self.lambda_2_beta_2 = nn.Parameter(torch.tensor(0.001, dtype=torch.float), requires_grad=True)
        self.eta = nn.Parameter(torch.tensor(1e-3, dtype=torch.float), requires_grad=True)


        # sparsity conv operations
        self.G = conv2d_with_activation(2, 32, 3, 1, 1)
        self.sp_enconv1 = conv2d_with_activation(32, 32, 3, 1, 1, nn.ReLU)
        self.sp_enconv2 = conv2d_with_activation(32, 32, 3, 1, 1)
        self.sp_deconv1 = conv2d_with_activation(32, 32, 3, 1, 1, nn.ReLU)
        self.sp_deconv2 = conv2d_with_activation(32, 32, 3, 1, 1)
        self.D = conv2d_with_activation(32, 2, 3, 1, 1)

        # low-rank conv operations
        self.lr_cp = CPDecomposition(cp_i, cp_j, cp_k, cp_num_basis)
        self.lr_conv = nn.Sequential(
            nn.Conv2d(topk*len(dis_funcs) , 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            # nn.ReLU()
        )

        # Gradient enhancement module
        self.csm = CrossStageModule(fft_norm)
        
    def forward(self, 
                x: torch.tensor, 
                y: torch.tensor, 
                pre_output: torch.tensor, 
                mask: torch.tensor
                ):
        lr_z = self.lowrank(x)
        sp_t = self.sprsity(x)
        new_x = self.gradient_decesent(x, y, lr_z, sp_t, mask)
        enhanced_x = self.csm(new_x, y, pre_output, mask)
        self.x = new_x
        self.lr_z = lr_z
        self.sp_t = sp_t
        self.enhanced_x = enhanced_x

        return enhanced_x
    
    def get_output(self):
        return self.x, self.lr_z, self.sp_t, self.enhanced_x

    def lowrank(self, x: torch.tensor):
        """
            Args:
                x: [B, H, W] and complex
        """
        # batch_x = x[:, None]
        magnitude = torch.abs(x).float()
        angle = torch.angle(x).float()
        h, w = x.shape[-2:]
        nonoverlap_patchs, padded_size = unfold(magnitude, self.patch_size) # [b, p, c, ph, pw]
        all_patches = []
        for distance_function in self.dis_funcs:
            patches, indicex = select_most_similar_patches(nonoverlap_patchs, nonoverlap_patchs, distance_function, self.topk) # [B, P, topk, C, H, W]
            all_patches.append(patches)
        nonlocal_patches = torch.concat(all_patches, dim=2) # [B, P, topk*n, C, H, W]

        B_, P_, TOPKN_, C_, H_, W_ = nonlocal_patches.shape

        cp_des_input = torch.reshape(nonlocal_patches, (B_*P_, TOPKN_, H_, W_))
        cp_des_output = self.lr_cp(cp_des_input)    # [B*P, topk*n, H, W]


        cp_des_output = torch.reshape(cp_des_output, (B_, P_, TOPKN_*C_, H_, W_))  
        merge_output = fold(cp_des_output, padded_size, (h, w)) # [B, topk*n*C, H, W]

        magnitude_output = self.lr_conv(merge_output)

        output = magnitude_output * torch.exp(1j * angle)

        return output


    
    def sprsity(self, x: torch.tensor):
        """
            Args:
                x: [B, H, W] and complex
        """
        # real_x = torch.view_as_real(x).type(torch.float)
        # real_x = torch.permute(real_x, (0, 3, 1, 2))
        real_x = complex_to_real(x)

        x_D = self.G(real_x)

        D_x = self._sparsity_encoder(x_D)

        shrinked_t = torch.sign(D_x) * F.relu(torch.abs(D_x) - self.lambda_2_beta_2)
        t = self._sparsity_decoder(shrinked_t)
        t = self.D(t)
        t = t + real_x
        # t = torch.permute(t, (0, 2, 3, 1)).contiguous()
        # t = torch.view_as_complex(t)
        # output = t.unsqueeze(dim=1)
        output = real_to_complex(t)


        if self.training:
            DT_D_x = self._sparsity_decoder(D_x)
            self.sp_encoder_input = x_D
            self.sp_decoder_output = DT_D_x
            self.DT_D_x = output # there is no DT_D_x, we assign t to it just for not modifying the vis_img
        
        return output
    
    def _sparsity_encoder(self, x):
        x_1 = self.sp_enconv1(x)
        x_2 = self.sp_enconv2(x_1)
        return x_2
    
    def _sparsity_decoder(self, x):
        x_1 = self.sp_deconv1(x)
        x_2 = self.sp_deconv2(x_1)

        return x_2
    
    def gradient_decesent(self, x: torch.tensor, y: torch.tensor, lr_z: torch.tensor, sp_t: torch.tensor, mask: torch.tensor):
        a = ifft2(torch.fft.ifftshift(torch.fft.fftshift(fft2(x, self.fft_norm), dim=[-2, -1]) * mask - y, dim=[-2, -1]), self.fft_norm)
        b = self.beta_1 * (x - lr_z)
        c = self.beta_2 * (x - sp_t)
        output = x - self.eta * (a + b + c)

        return output
    
    def compute_loss(self):
        return F.mse_loss(self.sp_encoder_input.float(), self.sp_decoder_output.float())


class CPDecomposition(nn.Module):
    def __init__(self, I, J, K, num_basis) -> None:
        super().__init__()

        rtgus = []
        for index in range(num_basis):
            rtgus.append(RTGU(index, I, J, K))
        
        self.blocks = nn.ModuleList(rtgus)

        self.channel_fusion = nn.Conv2d(I*num_basis, I, 3, 1, 1)
        # self.height_fusion = nn.Conv2d(J*num_basis, J, 3, 1, 1)
        # self.width_fusion = nn.Conv2d(K*num_basis, K, 3, 1, 1)

        self.fea_conv = nn.Sequential(
            nn.Conv2d(I, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, I, 3, 1, 1)
        )

        self.weight = nn.Parameter(torch.tensor([0.001] * num_basis, requires_grad=True), requires_grad=True)
        
    
    def forward(self, x):
        """
            Args:
                x: [B, C, H, W]
            Output:
                ouput: [B, C, H, W]
        """
        
        fea = self.fea_conv(x)
        next_input = fea
        outputs = []
        pre_output = 0
        
        
        for _, block in enumerate(self.blocks):
            cur_output = block(next_input)
            next_input = next_input - cur_output
            outputs.append(cur_output)
        
        all_tensors = torch.stack(outputs, dim=1)
        weight = torch.reshape(self.weight, (1, -1, 1, 1, 1))
        weight = F.softmax(weight, dim=1)
        weight_sum = torch.sum(all_tensors * weight, dim=1)

        output = weight_sum * fea

        return output


class RTGU(nn.Module):
    def __init__(self, index, I, J, K) -> None:
        super().__init__()
        self.index = index

        self.a_conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(I, I, 1, 1)
        )

        self.a_conv2 = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(I, I, 1, 1)
        )

        self.a_conv3 = nn.Sequential(
            nn.Conv2d(2*I, I, 1, 1),
            nn.Sigmoid()
        )

        self.b_conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(J, J, 1, 1)
        )

        self.b_conv2 = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(J, J, 1, 1)
        )

        self.b_conv3 = nn.Sequential(
            nn.Conv2d(2*J, J, 1, 1),
            nn.Sigmoid()
        )

        self.c_conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(K, K, 1, 1)
        )

        self.c_conv2 = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(K, K, 1, 1)
        )

        self.c_conv3 = nn.Sequential(
            nn.Conv2d(2*K, K, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.tensor):
        """
            Args:
                x: [B, I, J, K]
            Output:
                ouput: [B, I, J, K]
        """
        channel_x = x
        height_x = x.transpose(-3, -2)
        width_x = x.transpose(-3, -1)
        
        a1 = self.a_conv1(channel_x)  # [B, I, 1, 1]
        a2 = self.a_conv2(channel_x)  # [B, I, 1, 1]
        a_cat = torch.concat((a1, a2), dim=1)
        a = self.a_conv3(a_cat)

        b1 = self.b_conv1(height_x)   # [B, J, 1, 1]
        b2 = self.b_conv2(height_x)  # [B, I, 1, 1]
        b_cat = torch.concat((b1, b2), dim=1)
        b = self.b_conv3(b_cat)

        c1 = self.c_conv1(width_x)    # [B, K, 1, 1]
        c2 = self.c_conv2(width_x)  # [B, I, 1, 1]
        c_cat = torch.concat((c1, c2), dim=1)
        c = self.c_conv3(c_cat)


        # kronecker product
        a = a.squeeze(dim=-1)
        b = b.squeeze(dim=-1).transpose(-2, -1)

        ab = torch.matmul(a, b).unsqueeze(dim=-1) # [B, I, J, 1]
        c = c.transpose(1, -1)  # [b, 1, 1, k]

        output = ab * c # [B, I, J, K]

        return output


class CrossStageModule(nn.Module):
    def __init__(self, fft_norm) -> None:
        super().__init__()
        self.fft_norm = fft_norm
        in_channel = 2
        out_channel = 16
        num_heads = 8

        self.cur_dm = DualModel(in_channel, out_channel)
        self.pre_dm = DualModel(in_channel, out_channel)

        self.attention = Attention(out_channel*2, num_heads, False)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channel, 2, 3, 1, 1),
        )
        self.eta = nn.Parameter(torch.tensor(1e-3, dtype=torch.float), requires_grad=True)

    def forward(
            self, 
            x: torch.tensor, 
            y: torch.tensor, 
            pre_output: torch.tensor, 
            mask: torch.tensor):
        
        cur_dm_output = self.cur_dm(x)
        pre_dm_output = self.pre_dm(pre_output)

        cat_output = torch.concat((cur_dm_output, pre_dm_output), dim=1)
        attention_output = self.attention(cat_output)
        out = self.conv(attention_output)



        rec_kspace = torch.fft.fftshift(fft2(real_to_complex(out), self.fft_norm), dim=[-2, -1])
        dc_output = (1- mask) * rec_kspace + (self.eta * mask * rec_kspace + y) / (1 + self.eta)
        output = ifft2(torch.fft.ifftshift(dc_output, dim=[-2, -1]), self.fft_norm)

        return output


class DualModel(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.freq_conv1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1,groups=out_channel),
            nn.Conv2d(out_channel, out_channel, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1,groups=out_channel),
            nn.Conv2d(out_channel, out_channel, 1, 1),
        )

        self.freq_conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1,groups=out_channel),
            nn.Conv2d(out_channel, out_channel, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1,groups=out_channel),
            nn.Conv2d(out_channel, out_channel, 1, 1),
        )


        self.spatial_conv1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1,groups=out_channel),
            nn.Conv2d(out_channel, out_channel, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1,groups=out_channel),
            nn.Conv2d(out_channel, out_channel, 1, 1),
        )

        self.spatial_conv2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, 1, 1,groups=out_channel),
            nn.Conv2d(out_channel, out_channel, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1,groups=out_channel),
            nn.Conv2d(out_channel, out_channel, 1, 1),
        )
        self.eta1 = nn.Parameter(torch.tensor(1e-3, dtype=torch.float), requires_grad=True)
        self.eta2 = nn.Parameter(torch.tensor(1e-3, dtype=torch.float), requires_grad=True)

    
    def forward(self, x: torch.tensor):
        real_x = complex_to_real(x)
        fea_x = self.conv(real_x)

        freq_in1 = complex_to_real(torch.fft.fft2(real_to_complex(fea_x), norm="ortho"))
        spatial_in1 = fea_x

        freq_output1 = self.freq_conv1(freq_in1) + freq_in1
        ift_freq_output1 = complex_to_real(torch.fft.ifft2(real_to_complex(freq_output1), norm="ortho"))

        spatial_output1 = self.spatial_conv1(spatial_in1) + spatial_in1

        add_output = spatial_output1 + (1 - self.eta1) * ift_freq_output1

        freq_in2 = complex_to_real(torch.fft.fft2(real_to_complex(add_output), norm="ortho"))
        spatial_in2 = add_output

        freq_output2 = self.freq_conv2(freq_in2) + freq_in2
        ift_freq_output2 = complex_to_real(torch.fft.ifft2(real_to_complex(freq_output2), norm="ortho"))

        spatial_output2 = self.spatial_conv2(spatial_in2) + spatial_in2

        output = spatial_output2 + (1 - self.eta2) * ift_freq_output2

        return output


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, LayerNorm_type = 'WithBias'):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm(dim, LayerNorm_type)
        


    def forward(self, x):
        b,c,h,w = x.shape

        layer_orm_output = self.norm(x)
        qkv = self.qkv_dwconv(self.qkv(layer_orm_output))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        output = out + x
        return output

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)
