import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
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


class PreNorm(nn.Module):
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
        block_size (int): block size

    Returns:
        blocks: (num_blocks*B, block_size, block_size, C)
    """
    B, D, H, W, C = x.shape
    b_d, b_h, b_w =cube_size[0], cube_size[1], cube_size[2]
    # if b_d == 0:
    #     b_d = D
    x = x.view(B, D // b_d, b_d, H // b_h, b_h, W // b_w, b_w, C)
    blocks = x.permute(0, 1, 3, 5, 2, 4, 6 ,7).contiguous().view(-1, b_d,b_h, b_w, C)
    # return blocks.view(-1, block_size[0]*block_size[1]*block_size[2], C)
    return blocks


def compute_attn_mask(D, H, W, cube_size=(2,8,8)):
    b_d, b_h, b_w = cube_size[0], cube_size[1], cube_size[2]  # block depth/height/width    
    m_d, m_h, m_w = b_d//2, b_h//2, b_w//2 ## move depth/height/width
    mask = torch.zeros((1, D, H, W, 1))  # 1 C H W 1
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
    mask_blocks = cube_partition(mask, cube_size)  # nW, window_size, window_size, 1
    mask_blocks = mask_blocks.view(-1, b_d*b_h*b_w)# nB, block_depth
    attn_mask = mask_blocks.unsqueeze(1) - mask_blocks.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask   

class TransCube(nn.Module):
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
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv3d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim
        self.cube_size = cube_size
        

    def forward(self, x, move_block):
        b, d, h, w, c= x.shape
        cube_size = self.cube_size 
        c_d, c_h, c_w = cube_size[0], cube_size[1], cube_size[2]       
        if move_block:
            x = torch.roll(x, shifts=(-c_d//2, -c_h//2, -c_w//2), dims=(1, 2, 3))
        x = cube_partition(x, cube_size).view(-1, c_d*c_h*c_w, c) 
            
        q_inp = self.to_q(x) 
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))

        # q: b,heads,d*h*w,c
        q = F.normalize(q, dim=-2, p=2)
        k = F.normalize(k, dim=-2, p=2)
        # attn = (k @ q.transpose(-2, -1)) 
        
   
        attn = (q @ k.transpose(-2, -1) ) 
        attn = attn * self.rescale
        
        if move_block:
            mask = compute_attn_mask(d, h, w, cube_size)
            n_B = mask.shape[0]
            N = c_d* c_h* c_w
            attn = attn.view(b, n_B, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0).cuda()
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1,2)
        x = x.reshape(b, d * h * w , self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, d, h, w, c)
        if move_block:
            x = torch.roll(out_c, shifts=(c_d//2, c_h//2, c_w//2), dims=(1, 2, 3))
        out_p = self.pos_emb(v_inp.reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1) # P = f_p(v)
        out = out_c + out_p
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
            num_blocks,
            cube_size
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, TransCube(dim=dim, dim_head=dim_head, heads=heads, cube_size=cube_size)),
                PreNorm(dim, TransCube(dim=dim, dim_head=dim_head, heads=heads, cube_size=cube_size)),
                PreNorm(dim, FeedForward(dim=dim)),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,d,h,w]
        return out: [b,c,d,h,w]
        """
        x = x.permute(0, 2, 3, 4, 1) 
        for (attn, sc_attn, ff, sc_ff) in self.blocks:
            x = attn(x,move_block=False) + x
            x = ff(x) + x
            
            x = sc_attn(x, move_block=True) + x
            x = sc_ff(x) + x  
        out = x.permute(0, 4, 1, 2, 3)
        return out

class TransUnet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, emb_dim=32, patch_size=(1,4,4), cube_size=(2,4,4), stage=2, num_blocks=[2,4,4]):
        super(TransUnet, self).__init__()
        
        self.stage = stage

        # Input projection
        # self.embedding = nn.Conv3d(in_channels, emb_dim, patch_size, patch_size, bias=False)
        self.embedding = nn.Conv3d(in_channels, emb_dim, patch_size, (1,2,2), (0,1,1), bias=False)
        # self.embedding = nn.Conv3d(in_channels, emb_dim, (1,3,3), 1, (0,1,1), bias=False)


        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = emb_dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                MAB(dim=dim_stage, num_blocks=num_blocks[i], dim_head=emb_dim, heads=dim_stage // emb_dim, cube_size=cube_size), # SAB 模块
                nn.Conv3d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False), # 4*4的卷积核
            ])) 
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = MAB(dim=dim_stage, dim_head=emb_dim, heads=dim_stage // emb_dim, num_blocks=num_blocks[-1], cube_size=cube_size)

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose3d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),# Upsample
                nn.Conv3d(dim_stage, dim_stage // 2, 1, 1, bias=False), # 1*1的卷积核
                MAB(dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], dim_head=emb_dim,heads=(dim_stage // 2) // emb_dim, cube_size=cube_size),
            ]))
            dim_stage //= 2

        # Output projection
        # self.mapping = nn.ConvTranspose3d(emb_dim, out_channels, patch_size, patch_size, bias=False)
        self.mapping = nn.ConvTranspose3d(emb_dim, out_channels, patch_size, (1,2,2),(0,1,1), bias=False)

        # self.mapping = nn.Conv3d(emb_dim, out_channels, (1,3,3), 1, (0,1,1), bias=False)

        #### activation function
        # self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # self.apply(self._init_weights)

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
            fea = Fusion(torch.cat([fea, fea_encoder[self.stage-1-i]], dim=1)) # Fution：1*1卷积
            fea = Dc_MSAB(fea)

        # Mapping
        out = self.mapping(fea) + x

        return out

class SCubA(nn.Module): 
    def __init__(self, in_channels=3, n_outputs=1, n_feat=64, patch_size=(1,4,4), cube_size=(2,4,4), stage=3, **kwargs):
        super(SCubA, self).__init__()
        out_channels = n_outputs * in_channels
        modules_body = [TransUnet(in_channels, out_channels, emb_dim=n_feat, patch_size=patch_size, cube_size=cube_size, stage=2, num_blocks=[1,1,1]) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body) 
        self.conv_out = nn.Conv3d(out_channels, out_channels, kernel_size=(8,3,3), padding=(0,1,1),bias=False)
        self.out_channels = out_channels
    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        x = torch.stack(x, dim=2)
        ## Batch mean normalization works slightly better than global mean normalization, thanks to https://github.com/myungsub/CAIN
        mean_ = x.mean(2, keepdim=True).mean(3, keepdim=True).mean(4,keepdim=True)
        x = x-mean_ 

        b, c, d_inp, h_inp, w_inp = x.shape
        db, hb, wb = 8, 32, 32 
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        pad_d = (db - d_inp % db) % db
        x = F.pad(x, [pad_w//2, pad_w//2, pad_h//2, pad_h//2, pad_d//2,pad_d//2], mode='constant')#(left_pad, right_pad, top_pad, bottom_pad, front_pad, back_pad)
       
        h = self.body(x)
        out = self.conv_out(h + x).squeeze(dim=2)
        out = torch.split(out, dim=1, split_size_or_sections=self.out_channels)
        mean_ = mean_.squeeze(2)
        out = [o+mean_ for o in out]
        out = [o[:, :,  pad_h//2:h_inp+pad_h//2, pad_w//2:w_inp+pad_w//2] for o in out]
        return out
    
    
if __name__ == "__main__":
    model = SCubA(in_channels=3, n_outputs=1, n_feat=64, patch_size=(1,4,4), cube_size=(2,4,4), stage=2).cuda()
    b,c,d,h,w = 2, 3, 8, 128, 128 
    input = [torch.randn(b,c,h,w).cuda() for _ in range(d)]
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
