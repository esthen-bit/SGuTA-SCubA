import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out

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





class TransFrame(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_frm
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.num_frm = num_frm
        self.to_q = nn.Linear(dim* num_frm, dim_head * heads * num_frm, bias=False)
        self.to_k = nn.Linear(dim* num_frm, dim_head * heads * num_frm, bias=False)
        self.to_v = nn.Linear(dim* num_frm, dim_head * heads * num_frm, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads * num_frm, dim * num_frm, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv3d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        # self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b, d, h, w, c]
        return out: [b,h,w,c]
        """
        b, h, w, d, c= x_in.shape  ####b,h,w,d,c
        x = x_in.reshape(b,h*w,d*c)
        q_inp = self.to_q(x) #b,h*w,d*c  ####
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                (q_inp, k_inp, v_inp))# b, num_head, h*w, c/num_head 为各个头分配自己的d
        v = v 
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1)) 
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2) 
        x = x.reshape(b, h * w , self.num_heads * self.dim_head * d) #self.num_heads * self.dim_head=d
        out_c = self.proj(x).view(b, h, w, d, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, d, c).permute(0, 4, 3, 1, 2)).permute(0, 3, 4, 2, 1) # P = f_p(v)
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

class FAB(nn.Module): # 论文SAB模块
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
            num_frm
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, ResBlock3d(in_channels=dim,out_channels=dim)),
                PreNorm(dim, FeedForward(dim=dim))
            ]))
            # self.blocks.append(PreNorm(dim, FeedForward(dim=dim)))


    def forward(self, x):
        """
        x: [b,c,d,h,w]
        return out: [b,c,d,h,w]
        """
        # x = x.permute(0, 3, 4, 2, 1) # b,h,w,d,c
        x = x.permute(0, 2, 3, 4, 1)
        # for (attn, ff) in self.blocks:
        for ff in self.blocks:
            # x = attn(x) + x
            x = ff(x) + x
        # out = x.permute(0, 4, 3, 1, 2) 
        out = x.permute(0, 4, 1, 2, 3)
        return out
    


class ResBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock3d, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out



class TransUnet(nn.Module):# 论文Figure 2.(b) SST
    def __init__(self, in_channels=3, out_channels=3, emb_dim=32, stage=2, patch_size=(1,4,4), num_blocks=[1,1,1], num_frm=6):
        super(TransUnet, self).__init__()
        self.dim = emb_dim
        self.stage = stage

        # Input projection
        # self.embedding = nn.Conv3d(in_channels, emb_dim, (1,3,3), 1, (0,1,1), bias=False)
        self.embedding = nn.Conv3d(in_channels, emb_dim, patch_size, (1,2,2), (0,1,1), bias=False)


        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_stage = emb_dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                # FAB(dim=dim_stage, num_blocks=num_blocks[i], dim_head=emb_dim, heads=dim_stage // emb_dim, num_frm=num_frm), # SAB 模块
                ResBlock3d(in_channels=dim_stage,out_channels=dim_stage),
                nn.Conv3d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False), # DownSample 每次Channel*2,H/2,W/2
            ])) 
            dim_stage *= 2
            num_frm //= 2
        # Bottleneck
        # self.bottleneck = FAB(dim=dim_stage, dim_head=emb_dim, heads=dim_stage // emb_dim, num_blocks=num_blocks[-1], num_frm=num_frm)
        self.bottleneck = ResBlock3d(in_channels=dim_stage, out_channels=dim_stage)
        num_frm *= 2
        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose3d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0),# Upsample
                nn.Conv3d(dim_stage, dim_stage // 2, 1, 1, bias=False), # 1*1的卷积核
                ResBlock3d(in_channels=dim_stage //2 ,out_channels=dim_stage // 2),
            ]))
            dim_stage //= 2
            num_frm *= 2
        # Output projection
        # self.mapping = nn.Conv3d(emb_dim, out_channels, (1,3,3), 1, (0,1,1), bias=False)
        self.mapping = nn.ConvTranspose3d(emb_dim, out_channels, patch_size, (1,2,2), (0,1,1), bias=False)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
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
        fea = self.embedding(x)  ###

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

class ResUnet(nn.Module): # Figure2 a)
    def __init__(self, in_channels=3, out_channels=3, n_feat=64, patch_size=(1,4,4),stage=2, num_frm=8):
        super(ResUnet, self).__init__()
        modules_body = [TransUnet(in_channels, out_channels, emb_dim=n_feat, stage=stage, patch_size=patch_size, num_blocks=[1,1,1], num_frm=num_frm) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body) # starred expression 用于unpacking可迭代的变量
        self.conv_out = nn.Conv3d(out_channels, out_channels, kernel_size=(8,3,3), padding=(0,1,1),bias=False)
  

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

        db, hb, wb = 8, 8, 8 
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        pad_d = (db - d_inp % db) % db
        x = F.pad(x, [pad_w//2, pad_w//2, pad_h//2, pad_h//2, pad_d//2, pad_d//2], mode='constant')
        h = self.body(x)
        out = self.conv_out(h + x).squeeze(dim=2)
        out = torch.split(out, dim=1, split_size_or_sections=3)
        mean_ = mean_.squeeze(2)
        out = [o+mean_ for o in out]
        return out

if __name__ == "__main__":
    model = ResUnet(in_channels=3, out_channels=3, n_feat=64, patch_size=(1,4,4), stage=2, num_frm=8).cuda()
    b,c,d,h,w = 1, 3, 6, 128, 128
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











