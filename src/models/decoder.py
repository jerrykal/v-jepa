import torch.nn.functional as F
import torch.nn as nn
import math

def compute_scale_schedule(start, target, num_layers, prefer=2):
    if num_layers == 1:
        return [target / start]

    scales = []

    cur = start
    for i in range(num_layers - 1):

        remaining_layers = num_layers - 1 - i

        if cur * (prefer ** remaining_layers) <= target:
            s = prefer
        else:

            max_s = math.floor(target / (cur * (prefer ** (remaining_layers - 1))))  
            s = max(1, max_s)
        cur *= s
        scales.append(s)

    final_scale = target / cur
    scales.append(final_scale)
    return scales

class VideoDecoder(nn.Module):
    def __init__(self,
                 embed_dim=768,
                 out_dim=3,
                 num_layers=4,
                 patch_size=16,
                 tubelet_size=2,
                 img_size=224,
                 num_frames=16,
                 use_tanh=True,
                 prefer_scale=2):
        
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.img_size = img_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        self.t0 = num_frames // tubelet_size
        self.h0 = img_size // patch_size
        self.w0 = img_size // patch_size
        self.n_patch = self.h0 * self.w0
        self.total_tokens = self.t0 * self.n_patch

        self.temp_scales = compute_scale_schedule(self.t0, num_frames, num_layers, prefer=prefer_scale)
        self.spat_scales = compute_scale_schedule(self.h0, img_size, num_layers, prefer=prefer_scale)

        assert len(self.temp_scales) == num_layers
        assert len(self.spat_scales) == num_layers

        in_ch = embed_dim
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            out_ch = in_ch // 2 if i < num_layers - 1 else in_ch  
            conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
            bn = nn.BatchNorm3d(out_ch)
            act = nn.ReLU(inplace=True)
            self.layers.append(nn.Sequential(conv, bn, act))
            in_ch = out_ch

        self.head = nn.Conv3d(in_ch, out_dim, kernel_size=1)
        self.act_out = nn.Tanh() if use_tanh else nn.Identity()

    def forward(self, x):
        """
        x: [B, T*N, C] where T = num_frames//tubelet_size, N = (H*W)/patch_size^2
        """
        B, TN, C = x.shape
        assert TN == self.total_tokens, f"Expected {self.total_tokens}, got {TN}."

        # [B, T'*H'*W', C] → [B, T', H', W', C]
        x = x.view(B, self.t0, self.h0, self.w0, C)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, T', H', W']

        t, h, w = self.t0, self.h0, self.w0

        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                tgt_t, tgt_h, tgt_w = self.num_frames, self.img_size, self.img_size
                x = F.interpolate(x, size=(tgt_t, tgt_h, tgt_w), mode='trilinear', align_corners=False)
            else:
                sT = self.temp_scales[i]
                sS = self.spat_scales[i]
                tgt_t = int(round(t * sT))
                tgt_h = int(round(h * sS))
                tgt_w = int(round(w * sS))
                x = F.interpolate(x, size=(tgt_t, tgt_h, tgt_w), mode='trilinear', align_corners=False)
                t, h, w = tgt_t, tgt_h, tgt_w

            x = layer(x)

        x = self.head(x)
        x = self.act_out(x)
        return x