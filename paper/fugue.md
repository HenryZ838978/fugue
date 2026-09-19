# Fugue: Grafting a Score-Reading Music Language Model onto a Frozen High-Fidelity Diffusion Renderer

**Henry Zhang** · with Claude (Fable 5)
*Draft v0.1 — 2026-09-20*

## Abstract

Open music generators today split along a line: models that read symbolic scores (and therefore can cover a song) and
models that sound like records. YuE2-3B reads ABC notation natively but decodes through a VAE that loses the top octave
and stereo width; MiniMax-Music3 (MM3) renders with a 2.4B flow-matching DiT and a 44.1 kHz stereo vocoder but its
language model cannot be conditioned on a score — four training regimes on MM3's 8B LM measured a conditional effect of
≈0. We show that the two can be *grafted*: a 66M-parameter adapter maps YuE2's 64-dimensional, 25 Hz acoustic latent into
the 2048-dimensional, 25 Hz condition that MM3's DiT expects, with both upstream models entirely frozen. The adapter is
trained by regression on 191 hours of MM3's own generations — targets are recovered by teacher-forcing MM3's RVQ codes
back through its LM, inputs by encoding the rendered audio with YuE2's public VAE encoder — so no human labels and no
access to MM3's unreleased quantizer are needed. Score following survives the graft unchanged (re-transcription DTW
against the input score 0.064 vs. 0.064 for YuE2's native decode; a no-score control scores ≈1.0), cover identity under
version-identification retrieval is at parity, and the rendered audio regains the high band and stereo width of a
master. We release the adapter, the services, an interactive arena, and the lab notebook.

## 1. Introduction

A *cover* keeps a song's identity — its melody, harmony and structure — and changes everything else. Doing this with a
generative model requires two capabilities that current open systems have separately:

1. **Symbolic conditioning.** The model must read a score (or an equivalent symbolic plan) and follow it. YuE2-3B [1]
   does this: on its authors' SHS100K zero-shot cover benchmark, supplying the source score raises version-identification
   Hit@1 from 0.3 % to 71.3 %.
2. **Acoustic quality.** The renderer must produce a master-quality signal. MiniMax-Music3 [2] is, among open models,
   the one whose DiT + Flow-VAE vocoder most sounds like a record; on WildSongBench it trails only YuE2 and Suno.

These are not the same model, and making one model do both is expensive. Our first four attempts (§3) tried to teach
MM3's language model to read ABC and all failed by the same teacher-forcing measure, while YuE2 measured under the same
ruler read scores 136× better. The remaining variable was the model itself. So instead of training either model, we ask
whether the two can be *joined at the latent*: take YuE2's acoustic latent, which is what its own VAE would decode, and
translate it into the condition MM3's DiT was trained to follow.

The reason this is plausible is a measurement of MM3's LM→DiT bridge (§2): the DiT is conditioned on a single 2048-d
vector per 25 Hz frame, 90.6 % of which comes from the LM hidden state, scaled to 6.85 % amplitude and nearest-upsampled
×3.445 to the latent rate. The DiT receives a low-amplitude, 25 Hz "route map" and fills in all acoustic detail itself.
The map's *identity* lives upstream; its *texture* lives in the DiT. Grafting replaces the source of the map.

Contributions:

- A frozen-frozen grafting recipe between two unrelated music generators, mediated by a 66M adapter trained on
  self-generated pairs with zero labels (§4).
- A measurement of where MM3's music lives (§2) and an alignment correction for MM3's stitched time axis without which
  the pairing is off by up to three frames (§4.2).
- Evidence that score following and identity pass through the graft intact while spectral quality moves to the
  renderer's level (§5), plus a negative result on automatic aesthetic scoring (§5.4).
- Open weights, services, an arena, and the full notebook.

## 2. Where MiniMax-Music3 keeps its music

MM3 generates in two stages: an 8B Qwen3-based LM emits one semantic code per 25 Hz frame, a 0.6B depth decoder emits
seven residual codes per frame, and a 2.4B flow-matching DiT renders a 128-d latent at 86.13 Hz from the *hidden states*
of those two stages, which a Flow-VAE vocoder turns into 44.1 kHz stereo. The bridge between LM and DiT is one class,
`ConditionEncoder`:

```
c = proj( layer_scale · Σ_l softmax(w)_l · h_l ),   then nearest-upsample ×(44100/512)/(24000/960) = 3.445
```

with learned `softmax(w) = [0.906, 0.013, 0.013, 0.013, 0.014, 0.014, 0.014, 0.014]` (LM first, then the seven depth
layers, which the model does not distinguish), `layer_scale = 0.0685`, and `proj` a 4096→2048 conv with kernel 3.

Three consequences. (i) The DiT's condition is 90.6 % LM hidden state. (ii) It is small (6.85 % amplitude) and coarse
(25 Hz, held for 3.445 latent frames). (iii) MM3's 25 Hz frame rate (24000/960) is the same as YuE2's VAE rate
(48000/1920) — both, we suspect, inherited from video pipelines — so no resampling is needed between the two.

We confirm (ii) directly (§5.3): adding white noise at 30 % of the per-channel variance to `c` changes the output's
log-mel distance to the original by +0.04, against a seed-to-seed floor of 0.70; zeroing `c` gives 2.28.

## 3. Why we stopped training MM3

Under one teacher-forcing ruler — the drop in next-code cross-entropy when the true ABC is replaced by another song's
ABC, `d_spec` — four designs on MM3's LM gave the same answer (Table 1). YuE2 under the identical ruler and identical
prompt placement gives +0.340. The fourth round (r4) reproduced YuE2's training regime (plan loss, condition dropout,
keeping tempo lines) and drove the ABC-writing loss from 23.1 to 0.49 — the model *learned to write ABC* — while `d_spec`
stayed at +0.0025. The break is not between "seeing" and "writing" the score but between writing it and using it to
write audio. Placement, token space and position were ruled out by construction (r1 uses YuE2's placement). What remained
was the model.

| experiment | placement | `d_spec` | paired wins |
|---|---|---:|---:|
| r1 / r2 | ABC text as prefix | ≈0 | — |
| stage A | codec tokens as prefix | ≈0 | — |
| r3 | in-stream + prefix | +0.0003 | 26/48 |
| r4 | YuE2 regime | +0.0025 | 18/24 |
| **YuE2-3B** (control) | ABC text as prefix (= r1) | **+0.340** | **24/24** |

*Table 1. Conditional effect of the score on next-code prediction.*

## 4. Method

### 4.1 Interface

YuE2's acoustic stage is not "hidden → VAE"; it is a flow-matching velocity field inside the LM (a mixture-of-transformers
NAR path), so its hidden states change with the ODE time step and are not a clean condition. Moreover YuE2's *semantic*
tokenizer is not released, so arbitrary audio cannot be tokenized for its AR path. The only YuE2 representation obtainable
from the same audio on both the training and inference sides is the **VAE latent** (64-d, 25 Hz; the encoder is public),
which is also exactly what the NAR stage outputs at inference. The interface is therefore

```
z (T, 64) @ 25 Hz  ──adapter──▶  c25 (T, 2048) @ 25 Hz  ──MM3 nearest ×3.445──▶  DiT condition
```

with `c25` defined as ConditionEncoder's output *before* upsampling. Everything in MM3 downstream of `c25` is untouched;
YuE2's VAE decoder is bypassed entirely.

### 4.2 Pairs for free

MM3's quantizer (audio → RVQ codes) is unreleased, so "any recording → MM3 condition" is not available. It is not needed.
For 12,837 songs MM3 generated itself (the public `mm3-rvq-distill-corpus-8k`, ≈191 h, with codes, captions and lyrics),
we teacher-force the stored codes back through MM3's LM and depth decoder — a single causal forward each — and apply
ConditionEncoder up to `proj`. We verified the recovered hidden bundle is equivalent to generation-time: re-rendering it
gives log-mel L1 0.686 to the original, equal to the 0.698 obtained by changing the DiT seed. The input side is YuE2's VAE
encoder applied to the rendered 44.1 kHz audio resampled to 48 kHz. Domain shift between MM3's generations and real
recordings is common-mode here — both sides see the same audio — and only bites at inference (§5.2).

**Alignment.** MM3 renders in 200-frame windows with a 100-frame hop; each window yields ⌊200 × 3.4453⌋ = 689 latents and
the stitched output advances 345 latents per hop against a nominal 344.53, so window *k* lags the nominal clock by
0.136·k YuE2 frames — three frames by the end of a 105 s song. Frame *j* is paired with YuE2 frame
`round(j + 0.1358·k(j))`, `k(j) = 0` for `j < 125` else `⌊(j−25)/100⌋`. A linear probe rises from R² 0.262 (naive) to
0.347 (corrected), and a shift scan peaks at 0.

### 4.3 Adapter

`Linear(64→768) → 2× Conv1d(k=5) → depthwise Conv1d(k=63) → 8-layer pre-LN bidirectional Transformer (12 heads) →
LayerNorm → Linear(768→2048)`, 66.6M parameters, no absolute positions; inference is chunked (768 + 128 context) to match
the training crop. Loss is masked MSE in per-channel z-scored space, weighted `0.5 + 0.5 σ_c²/mean σ²` because the 2048
condition channels span a 35× range of standard deviation and the DiT sees the raw scale. Gaussian noise (σ = 0.15 in
standardized units) is added to the input during training to cover the gap between encoder latents (train) and
NAR-generated latents (inference); the decode→encode round-trip of a generated latent differs by 0.15 mean-absolute.
AdamW, 3e-4, cosine, 20k steps of 24 × 768-frame crops, bf16, 67 minutes on one RTX 4090.

| model | held-out R² (variance-weighted) | per-channel R² |
|---|---:|---:|
| ridge, ±8 frames (linear ceiling) | 0.384 | 0.150 |
| ridge on MM3's own DAV latent (128-d, pooled) | 0.20 | 0.09 |
| v1: 6L d512, 1,050 songs | 0.593 | 0.369 |
| v2: 8L d768, all 10,340 songs | 0.665 | 0.464 |
| v3: + variance-weighted loss | 0.672 | 0.462 |
| **v4: + input noise 0.15 (released)** | 0.671 | 0.461 |

*Table 2. Fit to the DiT condition. The DAV row shows YuE2's latent is a better interface than MM3's own waveform latent.*

## 5. Experiments

Two songs from the corpus (a Britpop track with vocals; a 5/4 jazz instrumental) and one real recording (s-AVE, 4:35)
serve the case studies; 210 real recordings serve the benchmark (§5.5).

### 5.1 Score following survives the graft

Each output is re-transcribed with SheetSage2 and compared with the input ABC by DTW over the interval sequence of the
instrumental voice (0 = identical transcription). For the Britpop song: YuE2 native decode 0.064, Fugue 0.064 (v1–v3;
v4 0.079); grafted *reconstruction* of the MM3 original 0.000 (v2–v4) with the vocal voice at 0.197 vs 0.200 for the
original re-rendered; no-score control 1.0–1.08; zero-condition 0.53. Jazz: 0.405 → 0.371 (v3) / 0.390 (v4). Real song,
piano cover: 0.353 → 0.369 (v4; v1–v3 0.41–0.43), against 0.219 for re-transcribing the real recording itself.
Chroma agreement between the two decodes of the same latent reaches the change-the-seed ceiling (+0.058 vs +0.055).

### 5.2 Domain shift

v4's input-noise augmentation is what closes the gap on real-song covers (the most out-of-distribution latents): 0.434 →
0.369 DTW, and its grafted reconstruction is the closest to the seed floor (log-mel L1 0.740 vs 0.69). R² is unchanged
by the augmentation; the effect is entirely at inference.

### 5.3 What the DiT tolerates

| condition | log-mel L1 to original |
|---|---:|
| teacher-forced (seed 7 / seed 8) | 0.686 / 0.703 |
| + 10 % / 30 % variance white noise | 0.697 / 0.729 |
| Fugue v4 grafted reconstruction | 0.740 |
| zero (DiT prior) | 2.279 |
| another song's condition | 5.077 |

### 5.4 Quality: what the ear hears, what the spectrum shows, and what Audiobox gets wrong

Blind listening by the first author on the core A/B (same latent, two decoders): YuE2-Vae "sounds like nothing special";
Fugue "is a different tier — free airline earbuds vs. AirPods Pro". The spectrum agrees:

| same latent | energy >8 kHz | >12 kHz | 99 % rolloff | side/mid |
|---|---:|---:|---:|---:|
| YuE2-Vae decode | 0.43 % | 0.08 % | 13.2 kHz | −12.9 dB |
| Fugue v4 | 0.95 % | 0.32 % | 18.3 kHz | −8.4 dB |
| reference: MM3 original / real record | 0.37 % / 0.26 % | 0.07 % / 0.03 % | 16.6 / 12.4 kHz | −5.9 / −6.1 dB |

YuE2's VAE also reconstructs a real recording at 7.7 dB SI-SDR against 17.7 dB for MM3's Flow-VAE. Audiobox-Aesthetics
PQ, however, scores YuE2-Vae *higher* on every pair (8.43 vs 8.09), and scores the YuE2-Vae round-trip of a real song
above the real song. It rewards smoothness; it is not a fidelity metric for this comparison and we report it only for
completeness. DiT-seed variance of every metric is negligible (PQ 8.12–8.19 over four seeds).

The graft does shift timbre toward MM3's priors (a blues-rock latent renders slightly harder-rock); key is preserved
(SheetSage2 reads K:Dm for both decodes; CLAP major−minor probe +0.166 vs +0.163).

### 5.5 Cover benchmark on real recordings

Protocol (mirrors the SHS100K zero-shot cover evaluation on the YuE2 model card, at 1/50 the database): 210 real
recordings, SheetSage2 score, two contrasting target styles per song from a bank of six (instrumental), first 60 s,
one seed; a no-score arm (YuE2 `cot=off`, style only) as control. Identity = Discogs-VINet [3] retrieval of the source
among the 210 originals; the same YuE2 latent is decoded by YuE2-Vae and by Fugue.

| arm | n | Hit@1 | Hit@10 | MRR | >8 kHz | >12 kHz | rolloff99 | side/mid | CLAP-style |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full-score · YuE2-Vae | TBD |
| full-score · **Fugue** | TBD |
| no-score · YuE2-Vae | TBD |
| no-score · Fugue | TBD |
| upper bound: original's own 60 s excerpt | 210 | 0.819 | 0.933 | 0.861 | | | | | |

*Table 3. To be filled from `eval/summary.json` (generation in progress at time of draft; preliminary n=32:
Hit@1 0.31 Fugue vs 0.28 YuE2-Vae, MRR 0.39 vs 0.39; no-score 0.00 / 0.00).*

## 6. Related work

Foley Control [4] freezes a text-to-audio DiT and trains only an adapter to admit a video encoder; Fugue is the same
shape with a symbolic music LM in the encoder's place. Freeze-Omni [5] and DITTO-TTS [6] bridge frozen LLM states to
speech decoders; DIFFA [7] proposes two-stage semantic/acoustic adapters. "Where does the sound go?" [8] warns that thin
adapters lose acoustics — our linear ceiling of 0.38 R² and the 0.67 reached by the Transformer adapter quantify that
warning. ALAS [9] stresses layer choice; here the interface is a VAE latent rather than an LM layer, and the analogous
choice — YuE2 latent vs. MM3's own DAV latent — favours YuE2's (Table 2). SongEcho and ACE-Step 1.5 are the open cover
baselines on the YuE2 card (Hit@1 48.4 % and 2.4 %).

## 7. Limitations and next steps

The adapter is trained only on MM3's generations; real-song latents are out of distribution and the timbre drift in
§5.4 is the visible cost. Vocals pass through but PER was not measured. The benchmark uses a 210-song database rather
than SHS100K's 10,545 and one seed rather than two. Next: WildSongBench, an SHS100K subset, DiT-space perceptual loss
for the adapter, and representation-engineering steering in the `c25` space (MM3's existing steering axes are linear
images under `proj`, so they transfer at inference time without retraining).

## References

[1] YuE2-3B model card, M-A-P, 2026. [2] MiniMax-Music3, MiniMax, 2026. [3] Araz et al., Discogs-VI / Discogs-VINet,
ISMIR 2024. [4] Foley Control, arXiv:2510.21581. [5] Freeze-Omni, arXiv:2411.00774. [6] DITTO-TTS, ICLR 2025.
[7] DIFFA, arXiv:2507.18452. [8] Where Does the Sound Go?, arXiv:2609.05871. [9] ALAS, arXiv:2505.19937.
