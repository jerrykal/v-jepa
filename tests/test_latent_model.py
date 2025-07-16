import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.models.latent_action import LatentActionEncoder

if __name__ == "__main__":
    assert (torch.cuda.is_available(), "CUDA is not available. Please run on a GPU machine.")

    device = torch.device("cuda")

    B = 64        # batch size
    T = 4         # time steps
    P = 196       # patches per frame
    D = 192       # feature dim
    d_codebook = 10

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = LatentActionEncoder(
        num_heads=8,
        d_codebook=d_codebook,
        inp_dims=D,
        n_codebook=1,
        lfq_bias=True,
        lfq_commit_weight=0.25,
        lfq_entropy_weight=0.1,
        lfq_diversity_weight=1.0,
        quant_loss_weight=1.0
    ).to(device)

    model.train()

    x = torch.randn(B, T, P, D, device=device)

    z_q, loss = model(x)

    current_mem = torch.cuda.memory_allocated(device) / 1024**2  # MB
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2  # MB

    print("Quantized latent action shape:", z_q.shape)
    print("Quantization loss:", loss)
    print(f"Current VRAM used : {current_mem:.2f} MB")
    print(f"Peak VRAM used    : {peak_mem:.2f} MB")
