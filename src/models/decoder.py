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
                 stem_dim=64,
                 num_layers=4,
                 out_dim=3,
                 patch_size=16,
                 tubelet_size=2,
                 img_size=224,
                 num_frames=16,
                 use_tanh=True):
        
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.img_size = img_size
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.stem_dim = stem_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        # Input token info
        self.h0 = self.w0 = img_size // patch_size
        self.t0 = num_frames // tubelet_size
        self.n_patch = self.h0 * self.w0
        self.total_tokens = self.t0 * self.n_patch


        # Step 1: channel schedule
        start_ch = stem_dim * num_layers
        ch_schedule = [embed_dim, start_ch]
        step = (start_ch - stem_dim) // (num_layers - 1) if num_layers > 1 else 0
        for i in range(1, num_layers):
            ch_schedule.append(start_ch - i * step)
        ch_schedule.append(stem_dim)  # ensure last
        ch_schedule = ch_schedule[:num_layers + 1]  # total num_layers steps

        # Step 2: build decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for in_ch, out_ch in zip(ch_schedule[:-1], ch_schedule[1:]):
            block = nn.Sequential(
                nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False),
                nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.decoder_blocks.append(block)

        # Step 3: final to RGB + optional tanh
        self.head = nn.Conv3d(stem_dim, out_dim, kernel_size=1)
        self.act = nn.Tanh() if use_tanh else nn.Identity()

    def forward(self, x):  # x: [B, t*N, C]
        B, TN, C = x.shape
        assert TN == self.total_tokens, f"Expected {self.total_tokens}, got {TN}"

        # → [B, t, H, W, C] → [B, C, t, H, W]
        x = x.view(B, self.t0, self.h0, self.w0, C)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, t, h, w]

        for block in self.decoder_blocks:
            x = block(x)
        B, C, T, H, W = x.shape
        if self.num_frames != T and \
            self.img_size != H and self.img_size != W:
            x = F.interpolate(
                x, 
                size=(self.num_frames, self.img_size, self.img_size), 
                mode='trilinear',
                align_corners=False
            )

        x = self.head(x)
        return self.act(x)