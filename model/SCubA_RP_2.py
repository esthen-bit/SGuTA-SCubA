import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
import time 

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    ## type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Norm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def cube_partition(x, cube_size):
    """
    Args:
        x: (B, H, W, C)
        cube_size (int): cube size

    Returns:
        cubes: (num_cubes*B, cube_size, cube_size, C)
    """
    B, D, H, W, C = x.shape
    b_d, b_h, b_w =cube_size[0], cube_size[1], cube_size[2]
    x = x.view(B, D // b_d, b_d, H // b_h, b_h, W // b_w, b_w, C)
    cubes = x.permute(0, 1, 3, 5, 2, 4, 6 ,7).contiguous().view(-1, b_d,b_h, b_w, C)
    return cubes


def compute_attn_mask(D, H, W, device,cube_size=(2,8,8)):
    b_d, b_h, b_w = cube_size[0], cube_size[1], cube_size[2]  # cube depth/height/width    
    m_d, m_h, m_w = b_d//2, b_h//2, b_w//2 ## move depth/height/width
    mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 C H W 1
    d_slices = (slice(0, -b_d),
                slice(-b_d, -m_d),
                slice(-m_d, None))
    h_slices = (slice(0, -b_h),
                slice(-b_h, -m_h),
                slice(-m_h, None))
    w_slices = (slice(0, -b_w),
                slice(-b_w, -m_w),
                slice(-m_w, None))
    cnt = 0
    for d in d_slices:
        for h in h_slices:
            for w in w_slices:
                mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_cubes = cube_partition(mask, cube_size)  # nW, cube_size, cube_size, 1
    mask_cubes = mask_cubes.view(-1, b_d*b_h*b_w)# nB, cube_depth
    attn_mask = mask_cubes.unsqueeze(1) - mask_cubes.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask   

class MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            cube_size = None
            
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.cube_size = cube_size
        self.dim = dim
        self.to_q = nn.Linear(dim, dim_head * heads, bias=True)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=True)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=True)
        self.rescale = dim_head ** -0.5
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * cube_size[0] - 1) * (2 * cube_size[1] - 1) * (2 * cube_size[2] - 1), heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.cube_size[0])
        coords_h = torch.arange(self.cube_size[1])
        coords_w = torch.arange(self.cube_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.cube_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.cube_size[1] - 1
        relative_coords[:, :, 2] += self.cube_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.cube_size[1] - 1) * (2 * self.cube_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.cube_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        
    def forward(self, x, move_cube):
        b, d, h, w, c= x.shape
        cube_size = self.cube_size 
        c_d, c_h, c_w = cube_size[0], cube_size[1], cube_size[2]
        patches_in_cube = c_d * c_h * c_w       
        if move_cube:
            x = torch.roll(x, shifts=(-c_d//2, -c_h//2, -c_w//2), dims=(1, 2, 3))
        x = cube_partition(x, cube_size).view(-1, patches_in_cube, c) 
            
        q_inp = self.to_q(x) 
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))

        # q: b,heads,d*h*w,c
        q = F.normalize(q, dim=-2, p=2)
        k = F.normalize(k, dim=-2, p=2)        
        attn = (q @ k.transpose(-2, -1)) * self.rescale

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:patches_in_cube, :patches_in_cube].reshape(-1)].reshape(
            patches_in_cube, patches_in_cube, -1) 
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, patches_in_cube, patches_in_cube
        
        if move_cube:
            mask = compute_attn_mask(d, h, w, x.device,cube_size)
            n_c = mask.shape[0]
            attn = attn.view(b, n_c, self.num_heads, patches_in_cube, patches_in_cube) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, patches_in_cube, patches_in_cube)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(-1, patches_in_cube, c)
        out = self.proj(x).view(b, d, h, w, c)
        if move_cube:
            out = torch.roll(out, shifts=(c_d//2, c_h//2, c_w//2), dims=(1, 2, 3))
        return out


        

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv3d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv3d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,d,h,w,c]
        return out: [b,d,h,w,c]
        """
        out = self.net(x.permute(0, 4, 1, 2, 3))
        return out.permute(0, 2, 3, 4, 1)

class MAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_cubes,
            cube_size
    ):
        super().__init__()
        self.cubes = nn.ModuleList([])
        for _ in range(num_cubes):
            self.cubes.append(nn.ModuleList([
                Norm(dim, MSA(dim=dim, dim_head=dim_head, heads=heads, cube_size=cube_size)),
                Norm(dim, MSA(dim=dim, dim_head=dim_head, heads=heads, cube_size=cube_size)),
                Norm(dim, FeedForward(dim=dim)),
                Norm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,d,h,w]
        return out: [b,c,d,h,w]
        """
        x = x.permute(0, 2, 3, 4, 1) 
        for (attn, sc_attn, ff, sc_ff) in self.cubes:
            x = attn(x,move_cube=False) + x
            x = ff(x) + x
            
            x = sc_attn(x, move_cube=True) + x
            x = sc_ff(x) + x  
        out = x.permute(0, 4, 1, 2, 3)
        return out

class MultiStageScale(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, emb_dim=32, patch_size=(3,4,4), cube_size=(2,8,8), num_scale=3, num_cubes=[1,1,1]):
        super(MultiStageScale, self).__init__()
        
        self.num_scale = num_scale

        # Input projection
        self.embedding = nn.Conv3d(in_channels, emb_dim, patch_size, (1,2,2), (1,1,1))
        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = emb_dim
        for i in range(num_scale):
            self.encoder_layers.append(nn.ModuleList([
                MAB(dim=dim_stage, num_cubes=num_cubes[i], dim_head=emb_dim, heads=dim_stage // emb_dim, cube_size=cube_size), 
                nn.Conv3d(dim_stage, dim_stage * 2, kernel_size=patch_size, stride=(1,2,2), padding=(1,1,1)), 
            ])) 
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = MAB(dim=dim_stage, dim_head=emb_dim, heads=dim_stage // emb_dim, num_cubes=num_cubes[-1], cube_size=cube_size)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(num_scale):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose3d(dim_stage, dim_stage // 2, kernel_size=patch_size, stride=(1,2,2), padding=(1,1,1)),# Upsample
                nn.Conv3d(dim_stage, dim_stage // 2, 1, 1, bias=False), 
                MAB(dim=dim_stage // 2, num_cubes=num_cubes[num_scale - 1 - i], dim_head=emb_dim,heads=(dim_stage // 2) // emb_dim, cube_size=cube_size),
            ]))
            dim_stage //= 2

        # Output projection
        self.de_emb = nn.ConvTranspose3d(emb_dim, out_channels, patch_size, (1,2,2), (1,1,1))

        # #### activation function
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)  

        # Encoder
        fea_encoder = []
        for (Ec_MSAB, DownSample) in self.encoder_layers:
            fea = Ec_MSAB(fea)
            fea_encoder.append(fea)
            fea = DownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (UpSample, Fusion, Dc_MSAB) in enumerate(self.decoder_layers):
            fea = UpSample(fea)
            fea = Fusion(torch.cat([fea, fea_encoder[self.num_scale-1-i]], dim=1)) # Fution：1*1卷积
            fea = Dc_MSAB(fea)

        # de_emb
        out = self.de_emb(fea) + x

        return out

class SCubA_RP_2(nn.Module): 
    def __init__(self, in_channels=3, n_inputs=4, n_outputs=1, n_feat=36, patch_size=(3,3,3), cube_size=(2,4,4), stage=1, num_scale=3, **kwargs):
        super(SCubA_RP_2, self).__init__()
        out_channels = n_outputs * in_channels
        modules_body = [MultiStageScale(in_channels, out_channels, emb_dim=n_feat, patch_size=patch_size, cube_size=cube_size, num_scale=num_scale, num_cubes=[1,1,1]) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body) 
        self.conv_out = nn.Conv3d(out_channels, out_channels, kernel_size=(n_inputs,3,3), padding=(0,1,1))
        self.out_channels = out_channels
        self.n_inputs = n_inputs
        self.cube_size = cube_size
        self.num_scale = num_scale
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        x = torch.stack(x, dim=2)
        ## Batch mean normalization works slightly better than global mean normalization, thanks to https://github.com/myungsub/CAIN
        mean_ = x.mean(2, keepdim=True).mean(3, keepdim=True).mean(4, keepdim=True)
        x = x-mean_ 

        b, c, d_inp, h_inp, w_inp = x.shape
        
        db, hb, wb = self.n_inputs, 2**(self.num_scale+1)*self.cube_size[1], 2**(self.num_scale+1)*self.cube_size[2]
        pad_d = (db - d_inp % db) % db 
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb

        x = F.pad(x, [pad_w//2, pad_w//2, pad_h//2, pad_h//2, pad_d//2,pad_d//2], mode='constant')#(left_pad, right_pad, top_pad, bottom_pad, front_pad, back_pad)
       
        h = self.body(x)
        out = self.conv_out(h + x).squeeze(dim=2)
        out = torch.split(out, dim=1, split_size_or_sections=self.out_channels)
        mean_ = mean_.squeeze(2)
        out = [o+mean_ for o in out]
        out = [o[:, :,  pad_h//2:h_inp+pad_h//2, pad_w//2:w_inp+pad_w//2] for o in out]
        return out
    
    
if __name__ == "__main__":
    # device = "cuda:0"
    device = "cpu"

    model = SCubA_RP_2(in_channels=3, n_outputs=1, n_feat=32, patch_size=(3,4,4), cube_size=(2,8,8), stage=2, num_scale=3).to(device)
    b,c,d,h,w = 1, 3, 4, 256, 256
    input = [torch.randn(b,c,h,w).to(device) for _ in range(d)]
    import time
    t = time.time()
    model(input)
    print('{:.2f} seconeds consummed'.format(time.time() - t))

    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    flops_counter = FlopCountAnalysis(model, (input,))
    flops = flops_counter.total()
    params = parameter_count_table(model)
    params = sum(p.numel() for p in model.parameters())
    print("Number of parameters:")
    print(params)
    print(f"FLOPs: {flops / 1e9:.2f}G")
    print(f'# of parameters: {params / 1e6:.2f}M' )
