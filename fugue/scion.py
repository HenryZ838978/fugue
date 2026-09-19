"""Scion: the symbolic side. SheetSage2 transcription + YuE2-3B (ABC + tags + lyrics -> semantic tokens -> NAR flow-matching
-> 64-d latent @ 25 Hz). YuE2's own VAE decoder is bypassed; the latent goes to fugue.adapter instead.
Requires the `yue2-infer` wheel shipped with m-a-p/YuE2-3B and (for transcription) m-a-p/SheetSage2 + m-a-p/MERT-v2-FullSong.
"""
import os
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _local(path_or_repo):
    """Local dir as-is; otherwise resolve an HF repo id to its snapshot dir."""
    if Path(path_or_repo).exists():
        return str(path_or_repo)
    from huggingface_hub import snapshot_download
    return snapshot_download(str(path_or_repo))


def with_tempo(abc, bpm):
    if bpm is None or any(l.startswith("Q:") for l in abc.split("\n")):
        return abc
    lines = abc.split("\n")
    i = next((k for k, l in enumerate(lines) if l.startswith("L:")), 3)
    lines.insert(i + 1, f"Q:1/4={bpm}")
    return "\n".join(lines)


def strip_chords(abc):
    return re.sub(r'"[^"]*"', "", abc)


class Transcriber:
    def __init__(self, sheetsage2_dir, mert_dir, device="cuda"):
        import sys
        sheetsage2_dir, mert_dir = _local(sheetsage2_dir), _local(mert_dir)
        sys.path.insert(0, sheetsage2_dir)
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(sheetsage2_dir, base_model_path=mert_dir, trust_remote_code=True, local_files_only=True).eval().to(device)

    def __call__(self, audio_path, work_dir=None):
        tmp = tempfile.mkdtemp(prefix="ss2_", dir=work_dir)
        try:
            res = self.model.transcribe(str(audio_path), output_dir=tmp)
            abc = (Path(tmp) / "score.abc").read_text() if (Path(tmp) / "score.abc").exists() else None
            kl = Path(tmp) / "key.lab"
            return dict(abc=abc, key_lab=kl.read_text().strip() if kl.exists() else None,
                        **{k: res.get(k) for k in ("duration_seconds", "vocal_notes", "instrumental_notes", "abc_measures", "abc_error")})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            torch.cuda.empty_cache()


class Scion:
    def __init__(self, yue2_dir, vae_dir, device="cuda"):
        from yue2 import YuE2Pipeline
        from yue2.modeling_vae import YuE2VAE
        yue2_dir, vae_dir = _local(yue2_dir), _local(vae_dir)
        self.pipe = YuE2Pipeline.from_pretrained(yue2_dir, vae=vae_dir, local_files_only=True, progress=False)
        self.pipe._load_model()
        self.vae = YuE2VAE.from_pretrained(vae_dir, device=device, local_files_only=True).eval()
        self.device = device

    def generate(self, style, lyrics="[Instrumental]", abc=None, cot="full", seed=0, max_sem=9000):
        """Returns (latent (T,64) float32 numpy, info dict). cot='off' ignores abc."""
        from yue2.protocol import Sampling, SongRequest
        req = SongRequest(style=style, lyrics=lyrics, cot=cot, abc=abc if cot != "off" else None, seed=seed)
        plan = self.pipe.plan(request=req)
        sem = self.pipe.generate_semantic(plan, sampling=Sampling(max_tokens=max_sem, min_tokens=min(200, max_sem)))
        lat = self.pipe.synthesize(sem).astype(np.float32)
        torch.cuda.empty_cache()
        return lat, dict(abc_tokens=len(plan.abc_ids), semantic_tokens=len(sem.tokens), truncated=bool(sem.truncated), latent_frames=int(lat.shape[0]))

    @torch.inference_mode()
    def decode_native(self, latent):
        """YuE2's own VAE decode (48 kHz stereo) — the baseline we compare against."""
        z = torch.from_numpy(np.asarray(latent, dtype=np.float32)).T[None].to(self.device)
        return self.vae.decode_tiled(z, output_device="cpu")[0].clamp(-1, 1).T.numpy(), 48000
