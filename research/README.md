# research/ — how this was built

Everything here ran on one 8×4090 box with two conda envs (`mm3`: diffusers 0.40-dev with MiniMax-Music3; `graphtokenizer`:
the `yue2-infer` wheel + SheetSage2). Paths are hard-coded to that box (`/cache/zhangjing/...`); read them as a lab notebook,
not a package. The reproducible inference path is `../fugue/`.

| stage | script | notes |
|---|---|---|
| lab notebook | `SCION.md` | the whole story, §1–§15, decisions and dead ends included |
| MM3 target side | `mm3_tf.py` | teacher-force RVQ codes through MM3 LM + depth decoder → `c25` (ConditionEncoder before upsample) |
| YuE2 input side | `yue2_encode.py` | YuE2-Vae encoder on the rendered audio (44.1k → 48k) |
| pairing / alignment | `data.py` | `align_index()` — MM3's 200/100-window stitched time axis, 0.136 frames/window drift |
| probes | `probe_linear.py`, `probe_dav.py` | ridge ceiling (0.38), shift scan, DAV-latent control (0.20) |
| training | `train_adapter.py` | v1–v4; `--var_weight`, `--x_noise` are the v3/v4 differences |
| decode | `graft_decode.py` | adapter → DiT → vocoder, chunked `predict_c25` |
| judgement experiments | `expA.py`, `expB_*.py` | condition ablation; YuE2-Vae vs MM3 Flow-VAE round-trips |
| scion inference | `yue2_gen.py` | YuE2 from ABC + tags + lyrics → latent (+ native decode) |
| content metrics | `followscore.py`, `chroma_agree.py` | SheetSage2 re-transcription DTW; chroma agreement |
| quality metrics | `quality_score.py` | Audiobox-Aesthetics + CLAP probes (Audiobox disagrees with ears — §13.11) |
| benchmark | `eval_gen.py`, `eval_score.py`, `vinet_embed.py` | ood216 protocol; Discogs-VINet loader without essentia |
| results | `eval_summary.json`, `eval_per_item.json` | the numbers in README §Results / paper §5.5 |
| launchers | `scripts/run_*.sh` | exact commands used (services, eval, training) |
