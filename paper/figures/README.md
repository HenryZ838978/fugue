# Figures

These figures are shared by the project README and the paper. PNG files provide predictable Markdown rendering; SVG files retain editable text and vector geometry.

| Figure | Type | Source |
|---|---|---|
| `cover-workflow` | Inference schematic | Released inference implementation |
| `training-pairs` | Training/inference schematic | `research/mm3_tf.py`, `research/yue2_encode.py`, and the adapter |
| `identity-and-style` | Aggregate metrics | `research/eval_summary.json`, checked against per-output records |
| `acoustic-changes` | Paired scatter plots | All 418 full-score pairs in `research/eval_per_item.json` |

Run from the repository root with Python 3, NumPy, Matplotlib, and Pillow:

```bash
python3 paper/figures/render_figures.py
python3 paper/figures/render_figures.py --check
```

The first command renders all four SVG/PNG pairs and writes `figure_stats.json`. The second checks the source summaries, data hashes, paired statistics, image dimensions, and nonblank image content without writing files. The script does not run generation or change the evaluation data.

The acoustic plot uses logarithmic axes for high-band energy and linear axes for rolloff and stereo power. Its diagonal means equal measured values, not equal perceptual quality. The highlighted point contains the marginal medians. The waveform frequency range is not the 25 Hz latent frame rate.

The diagrams describe implemented module interfaces, not a measured decomposition into independently controllable stems. Only the Fugue adapter is optimized during adaptation; all components are frozen at cover inference.
