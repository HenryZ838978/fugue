"""Graft: c25 (2048-d @ 25 Hz) -> MiniMax-Music3 DiT (frozen) -> Flow-VAE vocoder (frozen) -> 44.1 kHz stereo.

Replicates the official MiniMax-Music3 modular pipeline's chunked denoising (200-frame windows, 100-frame hop, 172-latent
overlap blending, 86/258-latent crops) so any latent length works. The only thing changed vs. the official pipeline is
where the condition comes from: instead of the 8-layer LM/depth hidden bundle passed through ConditionEncoder, we feed our
c25 and apply ConditionEncoder's own nearest-neighbour upsample (x441/128 = 3.445) ourselves.
"""
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import MiniMaxMusic3ConditionEncoder, MiniMaxMusic3Transformer1DModel, MiniMaxMusic3Vocoder
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

CHUNK_FRAMES, CHUNK_HOP = 200, 100
OVERLAP_LATENT = 172
CROP_LEFT_LATENT, CROP_RIGHT_LATENT = 86, 344 - 86
GUIDANCE = 1.7


class Rootstock:
    """Frozen MM3 acoustic side: condition_encoder (config only) + transformer + vocoder + scheduler."""

    def __init__(self, model_dir, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.cond_enc = MiniMaxMusic3ConditionEncoder.from_pretrained(model_dir, subfolder="condition_encoder", torch_dtype=dtype).to(device).eval()
        self.dit = MiniMaxMusic3Transformer1DModel.from_pretrained(model_dir, subfolder="transformer", torch_dtype=dtype).to(device).eval()
        self.vocoder = MiniMaxMusic3Vocoder.from_pretrained(model_dir, subfolder="vocoder", torch_dtype=dtype).to(device).eval()
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler")
        self.sr = int(self.vocoder.config.sampling_rate)

    def latent_len(self, frames):
        c = self.cond_enc.config
        return max(1, int(frames * c.output_sampling_rate / c.input_sampling_rate * c.input_hop_length / c.output_hop_length))

    def upsample(self, c25_chunk):
        x = c25_chunk.transpose(0, 1)[None]
        return F.interpolate(x, size=self.latent_len(c25_chunk.shape[0]), mode="nearest").transpose(1, 2)

    @torch.no_grad()
    def __call__(self, c25, steps=30, seed=7, guidance=GUIDANCE):
        """c25: (T, 2048) tensor -> (2, S) float32 waveform at self.sr."""
        c25 = c25.to(self.device)
        num_frames = c25.shape[0]
        generator = torch.Generator(self.device).manual_seed(seed)
        chunk_starts = [0] if num_frames <= CHUNK_FRAMES else list(range(0, num_frames - CHUNK_HOP, CHUNK_HOP))
        previous_latent = previous_condition = None
        latent_chunks = []
        for s in chunk_starts:
            e = min(s + CHUNK_FRAMES, num_frames)
            condition = self.upsample(c25[s:e].to(self.dit.dtype))
            overlap = 0
            if previous_latent is not None:
                overlap = min(previous_latent.shape[-1], condition.shape[1])
                condition[:, :overlap] = previous_condition[:, :overlap]
            latents = randn_tensor((1, self.dit.config.in_channels, condition.shape[1]), generator=generator,
                                   device=torch.device(self.device), dtype=condition.dtype)
            noise_prompt = latents[..., :overlap].clone() if overlap > 0 else None
            self.scheduler.set_timesteps(sigmas=np.linspace(1.0, 1.0 / steps, steps), device=self.device)
            pair = torch.cat([condition, torch.zeros_like(condition)], dim=0)
            for t in self.scheduler.timesteps:
                if overlap > 0:
                    tv = t.to(latents.dtype)
                    latents[..., :overlap] = (1.0 - (1.0 - 1e-6) * tv) * noise_prompt + tv * previous_latent[..., :overlap]
                pred = self.dit(hidden_states=latents.repeat(2, 1, 1), timestep=t.expand(2).to(latents.dtype),
                                encoder_hidden_states=pair, return_dict=False)[0]
                velocity = pred[1:2] + guidance * (pred[0:1] - pred[1:2])
                latents = self.scheduler.step(velocity, t, latents, return_dict=False)[0]
            if overlap > 0:
                latents[..., :overlap] = previous_latent[..., :overlap]
            os_ = max(0, latents.shape[-1] - 2 * OVERLAP_LATENT)
            oe_ = max(os_, latents.shape[-1] - OVERLAP_LATENT)
            previous_latent, previous_condition = latents[..., os_:oe_], condition[:, os_:oe_]
            latent_chunks.append(latents)
        hop = int(self.cond_enc.config.output_hop_length)
        wavs = []
        for i, latents in enumerate(latent_chunks):
            w = self.vocoder(latents.to(self.vocoder.dtype))
            left = 0 if i == 0 else CROP_LEFT_LATENT * hop
            right = 0 if i == len(latent_chunks) - 1 else CROP_RIGHT_LATENT * hop
            wavs.append(w[..., left: w.shape[-1] - right])
        return torch.cat(wavs, dim=-1).float().clamp(-1, 1)[0]
