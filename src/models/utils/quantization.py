import torch
import torch.nn as nn
from torch import log

from torch import Tensor
from einops import reduce
from einops import einsum
from einops import rearrange
from einops import pack, unpack

import torch.nn.functional as F

from typing import Tuple
from abc import ABC, abstractmethod


def entropy(p: Tensor, eps: float = 1e-6) -> Tensor:
    """Calculates the entropy of a probability distribution.

    Args:
        p (Tensor): The probability distribution.
        eps (float, optional): A small value to avoid taking the logarithm of zero.
            Defaults to 1e-6.

    Returns:
        Tensor: The entropy of the probability distribution.
    """
    return -(p * log(p.clamp(min=eps))).sum(dim=-1)


class IQuantization(nn.Module, ABC):
    """
    Abstract base class for all quantization modules.

    This interface defines the structure of a quantizer that:
    - projects input into a codebook space,
    - performs quantization (typically via nearest-neighbor or sign-based logic),
    - returns discrete indices and quantized vectors,
    - optionally computes training losses (e.g., commitment, entropy).

    Subclasses should implement encode, quantize, decode, and forward.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def quantize(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Quantizes the encoded input tensor.

        Args:
            x (Tensor): Encoded tensor in the quantization (codebook) space.

        Returns:
            Tuple[Tensor, Tensor]:
                - quantized vector (same shape as x),
                - codebook indices for each input vector (discrete representation).
        """
        pass

    @abstractmethod
    def encode(self, x: Tensor) -> Tensor:
        """
        Projects the raw input into the quantization space (before quantization).

        Args:
            x (Tensor): Raw input tensor of shape [B, D_in].

        Returns:
            Tensor: Encoded tensor of shape [B, D_q], where D_q = codebook_dim * num_codebooks.
        """
        pass

    @abstractmethod
    def decode(self, idx: Tensor) -> Tensor:
        """
        Decodes the codebook indices back into reconstructed vectors (in input space).

        Args:
            idx (Tensor): Codebook indices of shape [B] or [B, C] depending on codebook setup.

        Returns:
            Tensor: Reconstructed vectors of shape [B, D_in], same as original input.
        """
        pass


    @abstractmethod
    def forward(self, x: Tensor) -> Tuple[Tuple[Tensor, Tensor], Tensor | None]:
        """
        Full quantization forward pass including optional training loss computation.

        Typical flow:
            1. Encode the input into the codebook space.
            2. Quantize the encoded tensor to get discrete representation and quantized vector.
            3. Decode the quantized vector back to input space.
            4. If training: compute and return quantization loss (e.g., commitment, entropy).
               Else: return only reconstruction and indices.

        Args:
            x (Tensor): Raw input tensor [B, D_in].

        Returns:
            Tuple:
                - (reconstructed_tensor, indices): Reconstructed version of input and code indices.
                - loss (or None if not training)
        """
        pass

        

# Simplified version of the myscience implementation at: https://github.com/myscience/open-genie
class LookupFreeQuantization(IQuantization):
    """
    Lookup-Free Quantization module as originally introduced
    in the paper "Language Model Beats Diffusion: Tokenizer
    is key to visual generation" Yu et al. (2024).
    """

    def __init__(
        self,
        codebook_dim: int,
        input_dim: int,
        num_codebook: int = 1,
        use_bias: bool = True,
        commit_weight: float = 0.25,
        entropy_weight: float = 0.1,
        diversity_weight: float = 1.0,
    ) -> None:
        super().__init__()
        raise "not convert to IQuantization based"
        codebook_size = (2**codebook_dim) * num_codebook
        project = input_dim != codebook_dim * num_codebook

        self.proj_inp = (
            nn.Linear(input_dim, codebook_dim * num_codebook, bias=use_bias)
            if project
            else nn.Identity()
        )
        self.proj_out = (
            nn.Linear(codebook_dim * num_codebook, input_dim, bias=use_bias)
            if project
            else nn.Identity()
        )

        self.codebook_dim = codebook_dim
        self.num_codebooks = num_codebook
        self.codebook_size = codebook_size
        self.commit_weight = commit_weight
        self.entropy_weight = entropy_weight
        self.diversity_weight = diversity_weight

        # * Initialize the codebook
        # Use the bit_mask to generate the bit-codes for all the codebook entries
        # and then convert them to the actual codebook values {-1, 1}. Resulting
        # codebook will have shape (codebook_size, d_codebook).
        self.register_buffer(
            "bit_mask", 2 ** torch.arange(codebook_dim - 1, -1, -1)
        )  # if codebook_dim =8 than get tensor([128,  64,  32,  16,   8,   4,   2,   1])

        codes = torch.arange(codebook_size, dtype=int)[:, None] & self.bit_mask
        self.register_buffer("codebook", 2 * (codes != 0).float() - 1, persistent=False)

    def forward(
        self, inp: Tensor, beta: float = 100.0, transpose: bool = False
    ) -> Tuple[Tuple[Tensor, Tensor], Tensor | None]:
        # Standardize the input tensor to have shape (batch_size, seq_len, inp_dim)
        inp = rearrange(inp, "b d ... -> b ... d") if transpose else inp
        inp, ps = pack([inp], "b * d")  # pack to b ... d

        inp = self.proj_inp(inp)

        # Split into n_codebook parts
        inp = rearrange(inp, "b n (c d) -> b n c d", c=self.num_codebooks)

        # Quantize by simply assigning {-1, 1} to the input tensor depending on the sign
        # of the input tensor values. This is the lookup-free quantization step.
        # See Eq. (3) in the original paper. To obtain the quantized-code indices
        # we simply sum the bit-codes representation of the quantized values.
        quant = inp.sign()
        idxs = reduce((inp > 0).int() * self.bit_mask.int(), "b n c d -> b n c", "sum")

        # Use straight-through estimator to back-propagate through the quantization step
        code = (inp + (quant - inp).detach()) if self.training else quant
        code = rearrange(code, "b n c d -> b n (c d)")

        # Reconstruct the input tensor from the quantized values
        out = self.proj_out(code)
        out = unpack(out, ps, "b * d")[0]
        out = rearrange(out, "b ... d -> b d ...") if transpose else out

        # NOTE: Squeeze to remove the n_codebook dimension
        idxs = unpack(idxs, ps, "b * d")[0].squeeze()

        # No need to compute the loss if we are not training
        if not self.training:
            return (out, idxs), None

        # Compute the entropy loss
        inp_prob = 2 * einsum(inp, self.codebook, "... i d, j d -> ... i j")
        inp_prob = (inp_prob * beta).softmax(dim=-1)
        inp_prob = rearrange(inp_prob, "b n ... -> (b n) ...")

        avg_prob = reduce(inp_prob, "... c d -> c d", "mean")

        inp_ent = entropy(inp_prob).mean()
        avg_ent = entropy(avg_prob).mean()

        entropy_loss = inp_ent + self.diversity_weight * avg_ent

        # Compute commitment loss
        commit_loss = F.mse_loss(inp, quant.detach(), reduction="mean")

        # Compute the complete final loss
        loss = entropy_loss * self.entropy_weight + commit_loss * self.commit_weight

        return (out, idxs), loss


# Pytorch implementation of the official implementation at: https://github.com/google-deepmind/sonnet/blob/v1/sonnet/python/modules/nets/vqvae.py
class VectorQuantization(IQuantization):
    """
    Vector Quantization module as originally introduced in the paper "Neural Discrete Representation Learning"
    by van den Oord et al. (2017).
    """

    def __init__(
        self,
        embedding_dim: int,
        input_dim: int,
        num_embeddings: int,
        commitment_cost: float = 0.25,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost

        self.in_proj = (
            nn.Linear(input_dim, embedding_dim)
            if input_dim != embedding_dim
            else nn.Identity()
        )
        self.out_proj = (
            nn.Linear(embedding_dim, input_dim)
            if input_dim != embedding_dim
            else nn.Identity()
        )

        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1.0 / embedding_dim, 1.0 / embedding_dim)
        
    def encode(self, x: Tensor) -> Tensor:
        return self.in_proj(x)
    
    def quantize(self, encoded: Tensor) -> Tuple[Tensor, Tensor]:
        flat = encoded.view(-1, self._embedding_dim)
        # compute distances
        d2 = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2 * flat @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(dim=1)
        )
        idx_flat = torch.argmin(d2, dim=1)
        idxs = idx_flat.view(encoded.shape[:-1])
        quantized = self.codebook(idx_flat).view_as(encoded)
        return quantized, idxs
    
    def decode(self, idx: Tensor) -> Tensor:
        quant = self.codebook(idx)
        return self.out_proj(quant)
    
    def forward(
        self,
        inputs: Tensor,
    ) -> tuple[tuple[Tensor, Tensor], Tensor | None]:
        # Project the inputs to the codebook space
        encoded = self.in_proj(inputs)
        quantized, idxs = self.quantize(encoded)
        outputs = self.out_proj(quantized)

        # No need to compute the loss if we are not training
        if not self.training:
            return (outputs, idxs), None

        # Vector quantization loss
        e_latent_loss = F.mse_loss(quantized.detach(), encoded)
        q_latent_loss = F.mse_loss(quantized, encoded.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        return (outputs, idxs), loss
