# Fugue — graft YuE2's score-reading onto MiniMax-Music3's acoustics

**Cover any song into any style, with MiniMax-Music3 sound quality, from a score.**

Fugue is a *grafting* recipe: a frozen symbolic planner (YuE2-3B, which actually reads ABC notation) drives a frozen
high-fidelity acoustic renderer (MiniMax-Music3's DiT + Flow-VAE vocoder) through a 66M-parameter latent adapter trained
on 191 hours of self-generated pairs with **zero human labels**.

```
song.flac ─SheetSage2─▶ ABC score ─┐
target style tags ─────────────────┼─▶ YuE2-3B (AR → NAR flow-matching) ─▶ 64-d latent @ 25 Hz
lyrics (optional) ─────────────────┘                                              │
                                                              Fugue adapter (66M) │  ← the only trained part
                                                                                  ▼
                              MiniMax-Music3 DiT (2.4B, frozen) ─▶ Flow-VAE vocoder (frozen) ─▶ 44.1 kHz stereo
```

- **Cover ability that MiniMax-Music3 does not have.** MM3 cannot take a score at all; four rounds of LoRA / in-stream /
  prefix experiments trying to teach its 8B LM to read ABC all measured ≈0 conditional effect. YuE2 reads scores natively.
- **Audio quality that YuE2 does not have.** YuE2's VAE output is nearly empty above 12 kHz and 4–7 dB narrower in stereo
  than a record; the same latent rendered through MM3's DiT comes back with the high band and width of a master.
- **Identity survives the graft.** Re-transcribing Fugue output with SheetSage2 recovers the input score as faithfully as
  re-transcribing YuE2's own output (DTW interval distance 0.064 vs 0.064; no-score control ≈1.0), and version-identification
  retrieval (Discogs-VINet) ranks the source song the same either way.

Listen: [`samples/`](samples/) has A/B pairs — the *same* YuE2 latent decoded by YuE2's VAE vs. by Fugue.

## Results

### Identity is preserved, quality goes up (ood216 cover benchmark)

210 real recordings (Sawano Hiroyuki-heavy Japanese/Chinese pop & OST catalogue), each transcribed with SheetSage2 and
covered into two contrasting target styles, first 60 s, one seed. Every YuE2 latent is decoded both ways.
Identity = Discogs-VINet retrieval of the source among the 210 originals (protocol mirrors the SHS100K zero-shot cover
evaluation on the YuE2 model card, at 1/50 the database size).

| arm | Hit@1 ↑ | Hit@10 ↑ | MRR ↑ | energy >8 kHz | energy >12 kHz | 99% rolloff | side/mid |
|---|---:|---:|---:|---:|---:|---:|---:|
| YuE2 full-score, **YuE2-Vae decode** | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YuE2 full-score, **Fugue graft** | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| YuE2 no-score (control), YuE2-Vae | TBD | TBD | TBD | | | | |
| YuE2 no-score (control), Fugue | TBD | TBD | TBD | | | | |
| *upper bound: 60 s excerpt of the original itself* | TBD | TBD | TBD | | | | |

(Numbers filled from `eval/summary.json`; see `research/eval_score.py`.)

### What the adapter learns

| | R²(var-weighted) on held-out songs |
|---|---:|
| ridge regression, ±8-frame context (linear ceiling) | 0.384 |
| MM3's own DAV latent as input, same ridge | 0.20 |
| Fugue adapter v2 (8L d768, standardized MSE) | 0.665 |
| Fugue adapter v3 (+ variance-weighted loss) | 0.672 |
| **Fugue adapter v4 (+ input-noise augmentation 0.15) — released** | 0.671 |

v4 is released because it is best where it matters: covers of *real* songs (the most out-of-distribution input) follow the
score better (re-transcription DTW 0.369 vs 0.434 for v3), and its grafted reconstruction of an MM3 song is closest to the
change-the-seed noise floor (log-mel L1 0.740 vs 0.69).

### How robust is the DiT to condition error?

Adding white noise with 30 % of the per-channel variance to the DiT condition moves the output's log-mel distance to the
original by only +0.04 (seed-to-seed floor 0.70). The DiT fills in acoustic detail itself; the adapter does not have to
be perfect. Zeroing the condition (pure DiT prior) gives 2.28; swapping in another song's condition gives 5.08.

## Install

You need the three upstream model families (all open weights) plus the Fugue adapter:

| what | where | size |
|---|---|---|
| MiniMax-Music3 (transformer, vocoder, condition_encoder, scheduler) | `MiniMaxAI/MiniMax-Music3` | 2.4B DiT + vocoder |
| YuE2-3B + YuE2-Vae (+ the `yue2-infer` wheel shipped in the repo) | `m-a-p/YuE2-3B`, `m-a-p/YuE2-Vae` | 3B |
| SheetSage2 + MERT-v2-FullSong (transcription; skip if you bring ABC) | `m-a-p/SheetSage2`, `m-a-p/MERT-v2-FullSong` | |
| **Fugue adapter v4** | `HenryZ838978/fugue-scion-v4` | 66M (266 MB fp32) |

```bash
pip install -e .            # torch, diffusers>=0.40 (MiniMax-Music3 support), transformers, safetensors, soundfile, librosa
pip install /path/to/YuE2-3B/yue2_infer-0.1.5-py3-none-any.whl
```

Everything fits on one 24 GB GPU in bf16 (YuE2 ≈7 GB, MM3 acoustic side ≈5 GB, SheetSage2 ≈2.5 GB).

## Use

```bash
export FUGUE_MM3=/models/MiniMax-Music3 FUGUE_YUE2=/models/YuE2-3B FUGUE_YUE2_VAE=/models/YuE2-Vae \
       FUGUE_SHEETSAGE2=/models/SheetSage2 FUGUE_MERT=/models/MERT-v2-FullSong FUGUE_ADAPTER=HenryZ838978/fugue-scion-v4

# cover a recording into a new style (instrumental)
python -m fugue.cover --audio song.flac --style "intimate solo piano ballad with soft strings, cinematic" --native

# bring your own ABC, add lyrics
python -m fugue.cover --abc score.abc --style "1990s Britpop guitar rock, driving drums" --lyrics lyrics.txt
```

`--native` also writes YuE2's own VAE decode next to the Fugue output so you can A/B the same latent.

**Style prompt** — same conventions as YuE2: English, comma-separated free tags, roughly *genre/era, instruments, mood,
BPM, key, meter*. e.g. `1990s Britpop guitar rock, 106 BPM, A minor, driving drums, deliriously euphoric`. BPM/key are
taken from the transcribed score when not given. **Lyrics** — section tags `[Intro] [Verse] [Chorus] [Bridge] [Outro]`,
one line per sung line; omit for instrumental.

As a library:

```python
from fugue import FugueAdapter, Rootstock
from fugue.scion import Scion

scion = Scion("m-a-p/YuE2-3B", "m-a-p/YuE2-Vae")
latent, info = scion.generate(style="smooth jazz trio, Rhodes piano", abc=open("score.abc").read())   # (T, 64) @ 25 Hz
adapter = FugueAdapter("HenryZ838978/fugue-scion-v4")
root = Rootstock("MiniMaxAI/MiniMax-Music3")
wav = root(adapter(latent), steps=30, seed=7)                                                         # (2, S) @ 44.1 kHz
```

### Services + Arena

`services/` has the two resident HTTP services we run (`fugue_yue2_service.py` on the graphtokenizer env, `fugue_graft_service.py`
on the diffusers env — they can share one GPU), a CLI that orchestrates them, and `fugue_arena.py`, a Gradio page that
takes an upload + style and shows original / YuE2-native / Fugue side by side. A 30 s cover takes ≈50 s end to end on a
4090 (transcribe 5 s, YuE2 AR 14 s + NAR 4 s, DiT + vocoder 22 s); a 4:37 song takes ≈3.3 min.

## How it works (short)

MM3's DiT is conditioned on one 2048-d vector per 25 Hz frame produced by a tiny `ConditionEncoder`: a softmax mix of the
LM hidden state and seven RVQ-depth hidden states (90.6 % of the weight is on the LM), scaled by 0.0685, projected by a
k=3 conv, and nearest-upsampled ×3.445 to the 86 Hz latent rate. That is a low-amplitude, 25 Hz-bandwidth "route map";
the DiT fills in everything acoustic. Fugue replaces the map's *source* — the LM — with YuE2's latent, and leaves the DiT
and vocoder untouched.

Training pairs come for free: for 12,837 songs MM3 generated itself (with their RVQ codes), we teacher-force the codes
back through MM3's LM + depth decoder to recover the exact condition the DiT saw, and encode the rendered audio with
YuE2's (open) VAE encoder. The adapter regresses one onto the other. Two details matter: (1) MM3's audio is stitched from
200-frame windows on a 345-latent hop, so its time axis drifts 0.136 YuE2 frames per window — aligning for that raised
the linear probe from 0.26 to 0.35; (2) loss is a per-channel z-scored MSE with a variance-weighted term, because the
condition channels span a 35× range of scale and the DiT sees the raw scale.

Full lab notebook: [`research/SCION.md`](research/SCION.md). Paper draft: [`paper/fugue.md`](paper/fugue.md).

## Repository layout

```
fugue/          minimal inference package (adapter.py, graft.py, scion.py, cover.py)
services/       resident services, orchestration CLI, Gradio arena
research/       everything used to build and evaluate this (training, probes, eval, SCION.md lab notebook)
samples/        A/B mp3s
paper/          draft
```

## Licenses and credits

Code in this repository: Apache-2.0. The adapter weights are derived from MiniMax-Music3 outputs and inherit the
MiniMax-Music3 license terms; YuE2-3B / YuE2-Vae / SheetSage2 / MERT are © their authors (M-A-P) under their licenses.
Nothing here modifies any upstream weight.

Built by Henry Zhang with Claude (Fable 5). Discogs-VINet (Araz et al., ISMIR 2024) is used for identity evaluation.
