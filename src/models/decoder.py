import torch.nn.functional as F
import torch.nn as nn
import math

class VideoDecoder(nn.Module):
    def __init__(self,
                 embed_dim,
                 patch_size,
                 tubelet_size,
                 num_frames, height, width,
                 num_layers=4,
                 stem_dim=256,
                 use_tanh=True):
        
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.use_tanh = use_tanh

        self.t_latent = num_frames // tubelet_size
        self.Nh = height // patch_size
        self.Nw = width // patch_size
        self.N = self.Nh * self.Nw

        # decoder input shape: [B, t*N, C] → [B, C, t, Nh, Nw]
        self.restructure = lambda x, B: x.view(B, self.t_latent, self.N, embed_dim).transpose(2, 3).reshape(B, embed_dim, self.t_latent, self.Nh, self.Nw)

        # project to stem_dim * num_layers
        self.proj = nn.Conv3d(embed_dim, stem_dim * num_layers, kernel_size=1)

        # calculate required upsampling steps
        self.t_scale = tubelet_size
        self.h_scale = patch_size
        self.w_scale = patch_size

        t_upsample = int(math.log2(self.t_scale))
        h_upsample = int(math.log2(self.h_scale))
        w_upsample = int(math.log2(self.w_scale))
        self.upsample_steps = max(t_upsample, h_upsample, w_upsample)
        step = num_layers
        # build decoder blocks
        layers = []
        in_dim = stem_dim * step
        for i in range(self.upsample_steps):
            step = step - 1 if (step - 1) >= 1 else 1
            out_dim = stem_dim * step if i < self.upsample_steps - 1 else stem_dim
            scale_t = 2 if i < t_upsample else 1
            scale_h = 2 if i < h_upsample else 1
            scale_w = 2 if i < w_upsample else 1
            layers.append(nn.Sequential(
                nn.Upsample(scale_factor=(scale_t, scale_h, scale_w), mode='trilinear', align_corners=False),
                nn.Conv3d(in_dim, out_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ))
            in_dim = out_dim

        self.decoder_blocks = nn.Sequential(*layers)
        self.out_proj = nn.Conv3d(stem_dim, 3, kernel_size=1)

        if self.use_tanh: self.activation = nn.Tanh()

    def forward(self, x):
        B = x.size(0)
        x = self.restructure(x, B)  # (B, C, t, Nh, Nw)
        x = self.proj(x)            # (B, stem_dim*num_layers, t, Nh, Nw)
        x = self.decoder_blocks(x) # (B, stem_dim, T, H, W)
        x = self.out_proj(x)        # (B, 3, T, H, W)
        if self.use_tanh: x = self.activation(x)      # clamp to [-1, 1]
        return x