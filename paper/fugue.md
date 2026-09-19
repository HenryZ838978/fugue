# Fugue: High-Fidelity Music Covers by Grafting Frozen Generators

**Henry Zhang**

*Draft v0.2. Implementation and experiment assistance: Claude (Fable 5).*

## Abstract

A useful music cover must remain recognizable as the source song while adopting a different arrangement and production style. Fugue combines YuE2-3B's score-conditioned generation with MiniMax-Music3's diffusion-based acoustic renderer, without updating either pretrained model. A 66.6M-parameter adapter maps YuE2's acoustic latents into the renderer's existing conditioning interface. The adapter requires no additional original/cover pairs or human annotations: its inputs and targets are constructed from the audio and stored codes of MM3's own generations. On 209 real songs covered into two target styles each, Fugue achieves source-song retrieval MRR 0.609 versus 0.646 for YuE2's native decoder; a style-only control yields 0.022. Across the same 418 generated latents, median spectral rolloff rises from 13.8 to 17.6 kHz and stereo side/mid power from -10.9 to -4.8 dB. These acoustic changes accompany the author's listening preference for the grafted renderer. We release the inference package, adapter, evaluation records, and figure-generation code. Fugue demonstrates that score-conditioned behavior can be added to a frozen renderer by learning an external interface from self-generated data.

## 1. Introduction

Cover generation starts from musical material that a user already wants to keep. The task is to change its setting: for example, turn a vocal recording into a piano arrangement, or realize the same song as guitar-driven rock. This makes identity preservation, style control, and acoustic rendering jointly important. A convincing output in the requested genre is not a successful cover if it has lost the intended song.

YuE2-3B [1] provides score-conditioned music generation, including ABC input and a non-autoregressive acoustic stage. MiniMax-Music3 (MM3) [2] provides a diffusion transformer and stereo vocoder whose rendering quality motivated this work. We use the former to generate score-conditioned musical content and the latter to render it. The models are connected before waveform decoding through a learned latent mapping.

![Fugue inference pipeline and its native-decoder reference branch.](figures/cover-workflow.png)

*Figure 1. A recording is transcribed into ABC, or a score is supplied directly. YuE2 generates an acoustic latent under the score and style prompt. Fugue maps that latent into MM3's native DiT condition. The native YuE2 decoder provides a paired reference for isolating the rendering change.*

The name **Fugue** draws on the return of a recognizable subject through interweaving musical voices. It expresses a design goal: keep continuity in a vocal or instrumental line while allowing its surroundings to change and respond. The current system realizes this through score-conditioned generation of mixed audio.

The central contribution is **a way to compose existing capabilities without additional cover-pair supervision**. Training examples are constructed from MM3 generations with stored RVQ codes; no recording has to be paired with a human or model-produced cover. We also identify a usable cross-model interface, correct the time-axis mismatch introduced by chunked rendering, and evaluate the resulting system on real-song covers. The main experiment appears in Section 3; the unsuccessful attempts to train score conditioning directly into MM3 are retained in Appendix A.

## 2. Method

### 2.1 A shared interface for training and inference

YuE2's AR path produces semantic tokens, and its NAR flow-matching stage produces an acoustic latent `z` with 64 dimensions at 25 Hz. The public YuE2 VAE encoder maps audio into the same latent format. We choose this representation because it is available both from training audio and from score-conditioned generation. YuE2's NAR hidden states vary with the denoising step; they are not the fixed interface used here.

MM3 normally constructs its DiT condition from hidden states of its language model and RVQ depth decoder. We target the 2048-dimensional output of `ConditionEncoder` **before temporal upsampling**, denoted `c25`. The adapter predicts `c25` from `z`; MM3's upsampling, DiT, and vocoder then run unchanged. YuE2's native VAE decoder and MM3's LM/depth decoder are not used on the main cover-generation path.

```text
z: (T, 64), 25 Hz
    -> Fugue adapter
    -> c25: (T, 2048), 25 Hz
    -> MM3 native nearest-neighbor upsampling
    -> frozen DiT and Flow-VAE vocoder
    -> 44.1 kHz stereo
```

The two representations have the same nominal frame rate. MM3's chunked rendering nevertheless introduces a time-axis drift that must be corrected when pairing the training data.

### 2.2 Supervision from self-generated recordings

The paired corpus contains 12,837 MM3 generations, approximately 191 hours, with audio, RVQ codes, captions, and lyrics. For each generation we construct:

1. **Input:** resample the rendered audio from 44.1 to 48 kHz and encode it with the frozen YuE2 VAE encoder.
2. **Target:** teacher-force the stored codes and original prompt through MM3's frozen LM and depth decoder, then apply `ConditionEncoder` through its projection layer to recover `c25`.

![Training and inference paths, highlighting the shared adapter and the construction of supervision from one MM3 generation.](figures/training-pairs.png)

*Figure 2. Both sides of a training pair describe the same generated recording. At inference, YuE2's score-conditioned AR/NAR generator replaces the VAE encoder as the source of `z`. Only the adapter is optimized. No original/cover pairs are used in this adaptation stage.*

Stored generation codes avoid the need for MM3's unreleased audio-to-RVQ quantizer. In a reconstruction check, rendering the teacher-forced condition gives log-mel L1 0.686 to the stored original, compared with 0.698 between two renders using different DiT seeds. This provides an empirical consistency check on condition recovery.

Training uses encoder latents while cover inference uses NAR-generated latents. These inputs share a format but need not share a distribution. We therefore normalize channels and add Gaussian noise to the training inputs. Evaluation on real-song covers tests the complete path through transcription, YuE2 generation, and the adapter.

**Temporal alignment.** MM3 renders 200-frame windows with a 100-frame hop. Each full window yields 689 acoustic latents; stitching advances 345 latents per hop instead of the nominal 344.53125. Consequently, frame `j` of the MM3 condition corresponds approximately to YuE2 frame

```text
m(j) = round(j + 0.136054 * k(j))
k(j) = 0                              if j < 125
       floor((j - 25) / 100)           otherwise
```

The implementation clamps `k(j)` to the final rendering window and clips the resulting input index. The coefficient is `345 / (441 / 128) - 100`. Accounting for this drift raises the three-frame-context ridge probe from R² 0.262 to 0.347 in the alignment experiment.

### 2.3 Adapter and optimization

The adapter consists of a 64-to-768 linear projection, two residual local convolutions of kernel size 5, a residual grouped positional convolution of kernel size 63 with 16 groups, eight pre-LN bidirectional Transformer layers with 12 heads, and a final normalization and 768-to-2048 projection. It has 66.6M parameters and no absolute position embeddings.

Inputs and targets are standardized per channel. Let `y_tc` be a standardized target, `a_tc` the adapter prediction, and `sigma_c` the raw target-channel standard deviation. The masked loss is

```text
w_c = 0.5 + 0.5 * sigma_c^2 / mean_c(sigma_c^2)
L   = mean_valid_frames [ mean_c (w_c * (a_tc - y_tc)^2) ]
```

Input noise has standard deviation 0.15 in standardized units. The adapter is trained on 10,340 songs using AdamW, learning rate 3e-4 with warmup and cosine decay, 20,000 steps, and batches of 24 crops of 768 frames. The v4 run takes 67 minutes on one RTX 4090 after feature extraction. Corpus generation and extraction are additional costs; the measured corpus-wide teacher-forcing and VAE-encoding passes took about 53 and 74 single-GPU minutes respectively.

At inference, adapter predictions use 768-frame windows with 128 frames of context on each side, avoiding full-song quadratic attention. Output channels are unstandardized before MM3 rendering. YuE2 generation and acoustic rendering have separate random seeds.

## 3. Real-song cover evaluation

### 3.1 Protocol

The local benchmark, retained under the historical name `ood216`, contains 210 recordings with SheetSage2 transcriptions from a Japanese/Chinese pop and soundtrack catalogue. One transcription exceeds YuE2's context limit, leaving **209 source songs** for the full-score condition. Each is generated into two target styles, producing **418 latents**, each decoded by both YuE2's native VAE and Fugue v4. The retrieval gallery still contains **all 210 original recordings**.

The six target styles are piano ballad, Britpop guitar rock, jazz trio, synthwave, acoustic folk, and orchestral film score. Each song receives two styles three positions apart in this list, as specified in `eval_gen.py`. Generations are instrumental, limited to the first 60 seconds, with YuE2 seed 0; Fugue uses DiT seed 7 and 30 steps. The no-score control uses `cot=off`, the first assigned style for each song, and both decoders, yielding 210 outputs per decoder.

Identity is measured by cosine-similarity retrieval with Discogs-VINet [3], with the matching original as the single relevant gallery item. Style is measured by CLAP audio/text cosine similarity using `laion/larger_clap_music_and_speech`. Score following is measured by re-transcribing the output with SheetSage2 and computing DTW on instrumental-voice pitch-interval sequences. Outputs with insufficient transcribed notes are excluded from DTW aggregates.

This is a local paired-renderer evaluation, not a reproduction of the full SHS100K benchmark. The first 60 seconds of each original queried against the full-recording gallery provide a reference retrieval result: Hit@1 0.819, Hit@10 0.933, and MRR 0.861.

### 3.2 Identity and style together

![Identity and style metrics for the two decoders, with and without the source score.](figures/identity-and-style.png)

*Figure 3. Source identity and target style measure different requirements. Removing the score raises the aggregate CLAP median but reduces source retrieval to near-chance performance. The controls use one style per song; score-conditioned evaluation uses two.*

| Input and decoder | n | Hit@1 | Hit@10 | MRR | CLAP median |
|---|---:|---:|---:|---:|---:|
| Source score, YuE2 native | 418 | 0.557 | 0.809 | 0.646 | 0.319 |
| **Source score, Fugue** | **418** | **0.524** | **0.775** | **0.609** | **0.304** |
| Style only, YuE2 native | 210 | 0.000 | 0.033 | 0.022 | 0.436 |
| Style only, Fugue | 210 | 0.000 | 0.033 | 0.022 | 0.418 |

*Table 1. Identity and style on the local cover benchmark. Both conditions use YuE2 generation and differ in whether the source score is supplied.*

Fugue retains 94% of the native decoder's MRR. Its Hit@1 is lower by 3.35 percentage points. Across shared latents, source rank is unchanged in 239 pairs, higher with Fugue in 63, and higher with YuE2 native in 116. The median paired CLAP difference is -0.019. The graft therefore preserves much of the source-identification capability while changing the acoustic rendering.

Re-transcription DTW medians are 0.470 for YuE2 native and 0.523 for Fugue, computed over 372 and 375 valid outputs. Across the **372 pairs valid for both decoders**, the median paired difference is 0.000. The corresponding style-only medians are 1.686 and 1.622 over 197 valid outputs each.

### 3.3 Acoustic rendering and listening

The author preferred Fugue's rendering in informal development comparisons, particularly its high-frequency detail and stereo presentation. The release supplies same-latent A/B previews, a source-to-cover comparison in two styles, a vocal example, and one full-length example.

![Scatter plots of four acoustic measurements, one point per shared latent, with the identity diagonal and marginal medians.](figures/acoustic-changes.png)

*Figure 4. Acoustic changes across 418 paired renders. High-band energy axes are logarithmic; rolloff and stereo power use linear axes. The diamond marks the two marginal medians. Values above the diagonal indicate an increase in the plotted measurement, not a perceptual preference vote.*

| Measurement | YuE2 native median | Fugue median | Pairs with an increase |
|---|---:|---:|---:|
| Energy above 8 kHz | 0.466% | 0.966% | 415/418 (99.3%) |
| Energy above 12 kHz | 0.070% | 0.209% | 415/418 (99.3%) |
| 99% spectral rolloff | 13.8 kHz | 17.6 kHz | 418/418 (100%) |
| Stereo side/mid power | -10.9 dB | -4.8 dB | 405/418 (96.9%) |

*Table 2. Spectral and stereo measurements. All four increase simultaneously in 400/418 pairs (95.7%).*

The median paired differences are +3.8 kHz for rolloff and +6.1 dB for side/mid power. High-band energy is measured from the mono power spectrum. Rolloff is the median framewise frequency containing 99% of the cumulative magnitude spectrum. Stereo width is reported as `10 log10(var(L-R) / var(L+R))`, with the numerical stabilizers used in `eval_score.py`.

Audiobox-Aesthetics PQ decreases by a median paired 0.278, opposite to the author's listening preference. We report both results and leave the mechanism of the disagreement open. In a separate codec round-trip case study on a real recording, YuE2 VAE achieves SI-SDR 7.7 dB and MM3 Flow-VAE 17.7 dB. This experiment measures codec reconstruction, a distinct question from producing a new arrangement.

## 4. Interface analysis

### 4.1 What is learned

| Adapter or probe | Variance-weighted R² | Mean channel R² |
|---|---:|---:|
| Ridge, YuE2 latent with 17-frame context | 0.384 | 0.150 |
| Ridge, pooled MM3 DAV latent with 17-frame context | 0.201 | 0.086 |
| v1: 6 layers, width 512, 1,050 training songs | 0.593 | 0.369 |
| v2: 8 layers, width 768, 10,340 training songs | 0.665 | 0.464 |
| v3: v2 with variance-weighted loss | 0.672 | 0.462 |
| **v4: v3 with input noise, released** | **0.671** | **0.461** |

*Table 3. Condition-prediction results from the development runs. Ridge and neural-adapter experiments use different fitting/evaluation subsets; this table is not a controlled scaling comparison. The v4 validation run uses 150 held-out songs. R² uses the training-channel mean as its reference; it measures condition regression, not cover quality.*

The YuE2 latent is more predictive than the pooled MM3 DAV latent under the tested linear readouts. The Transformer adapters fit the condition better than the ridge baseline.

Input-noise augmentation leaves validation R² almost unchanged. In the real-song piano case study, v4 reduces re-transcription DTW from v3's 0.434 to 0.369, compared with 0.353 for YuE2 native. On one MM3 reconstruction case, v4 reduces log-mel L1 from v3's 0.787 to 0.740. These cases motivated the release choice; the 209-song benchmark evaluates v4 rather than an all-version ablation.

### 4.2 Renderer sensitivity

| Condition in the 30-second reconstruction case | Log-mel L1 to stored original |
|---|---:|
| Teacher-forced, DiT seed 7 / seed 8 | 0.686 / 0.703 |
| Added noise, 10% / 30% of per-channel variance | 0.697 / 0.729 |
| Fugue v4 condition predicted from encoded audio | 0.740 |
| Zero condition | 2.279 |
| Another song's condition | 5.077 |

*Table 4. Condition perturbations for one MM3-generated Britpop example. Each row compares its output with the stored original. The separate seed-to-seed distance is 0.698.*

This case shows tolerance to moderate unstructured condition error while remaining sensitive to removing or replacing the condition. Learned adapter residuals are structured and may behave differently from Gaussian noise.

The native condition projection and frame rates are documented in Appendix B.

### 4.3 Runtime and long-form example

The resident transcription/YuE2 and graft/rendering services occupy approximately 9.8 GB and 4.9 GB on the reference 24 GB GPU. A 30-second cover takes about 50 seconds end to end. A 4:37 piano output was generated with AR, NAR, and acoustic-rendering times of 54, 23, and 118 seconds. These individual runs use resident models; the long example exercises the chunked inference path.

## 5. Discussion: control after pretraining

A pretrained generator's default inference pipeline need not be its final interface. Fugue is an instance of **external adaptation after pretraining**: the learned bridge changes which upstream musical representations can drive a renderer, while the original weights remain fixed. The additional cover capability belongs to the composed system, not to a newly score-trained MM3 backbone.

Several intervention surfaces are relevant to this perspective. Representation engineering [7] can manipulate intermediate activations. LoRA [8] learns low-rank changes to effective weight transformations. Inference orchestration changes prompts, plans, execution, or selection around a model. Fugue instead learns the representation supplied at a module boundary. These are complementary sites for intervention with different access and training requirements; here we evaluate the learned external interface.

The engineering question is how to make a requested change while keeping the receiver in a regime where it behaves reliably. Here that question leads to a native conditioning target, temporal alignment, channel normalization, and input augmentation. Extending this to direct `c25` steering is a future experiment: small interventions may have little effect, while large ones may disrupt rendering. A useful next test would sweep intervention strength while tracking source identity, the requested style change, and audible artifacts.

## 6. Related work

**Score-conditioned music generation.** YuE2 [1] supplies the score-reading and generation capability used by Fugue. MM3 [2] supplies the acoustic renderer. The contribution is their composition and its training recipe, isolated by the local paired-decoder study.

**Frozen-model composition.** Foley Control [4] connects frozen video embeddings to a frozen text-to-audio DiT through added cross-attention, using video to condition sound. Fugue instead predicts the renderer's existing conditioning representation from another music generator's acoustic latent, with supervision recovered from the renderer's own generations. Freeze-Omni [5] likewise shows that speech interfaces can be learned around a frozen language model, using speech/text supervision. Fugue's bridge does not require additional examples of the target original-to-cover transformation.

**Representation and readout.** *Where Does the Sound Go?* [6] finds that acoustic information can remain recoverable in audio-conditioned LLM representations even when downstream answers fail to use it, implicating readout alignment. This motivates distinguishing information available in a representation from behavior expressed by a receiver. RepE [7] and LoRA [8] provide related but distinct intervention mechanisms discussed in Section 5.

## 7. Limitations and next steps

The benchmark uses a 210-recording gallery, one YuE2 seed, instrumental 60-second generations, and a limited source catalogue. Vocal intelligibility, listener preference, long-form consistency, and generalization across broader genres need dedicated evaluation. The adapter is fitted to MM3-generated audio encodings, while inference uses YuE2-generated latents. Some development examples shift timbre toward MM3's rendering preferences.

The present interface controls whole mixed-audio generation through the supplied score and style prompt. It does not guarantee an unchanged selected stem, exact note copying, or independent edits of vocal and instrumental lines. Larger-gallery cover evaluation, multi-listener comparisons, and controlled conditioning-space interventions are the next steps.

## Appendix A. Direct score-conditioning attempts on MM3

Before building the graft, we attempted text-prefix, token-prefix, and in-stream conditioning routes on MM3. The teacher-forcing diagnostic is `d_spec = CE(other-score) - CE(true-score)`: larger positive values indicate that the true score helps predict the audio codes.

| Experiment | Conditioning route | d_spec | Positive paired cases |
|---|---|---:|---:|
| r1 / r2 | ABC text prefix | approximately 0 | not tabulated |
| Stage A | Codec-token prefix | approximately 0 | not tabulated |
| r3 | In-stream and prefix | +0.0003 | 26/48 |
| r4 | Plan loss, condition dropout, tempo retained | +0.0025 | 18/24 |
| YuE2 control | ABC text prefix | +0.340 | 24/24 |

In r4, ABC-writing loss fell from 23.1 to 0.49 without a comparable increase in score-conditioned audio prediction. Within the tested settings, learning to produce ABC did not translate into useful score conditioning for audio prediction. These runs motivated reusing YuE2's existing capability.

## Appendix B. Native condition and reproducibility

MM3's `ConditionEncoder` forms a softmax-weighted sum of the LM hidden state and seven RVQ-depth hidden states, applies a learned scalar, and projects with a kernel-3 convolution from 4096 to 2048 channels. The measured mixture coefficients are approximately `[0.906, 0.013, 0.013, 0.013, 0.014, 0.014, 0.014, 0.014]`, and the scalar is 0.0685. Projection and normalization elsewhere in the model prevent interpreting that scalar alone as the strength of conditioning.

The condition rate is `24000/960 = 25` Hz; the acoustic latent rate is `44100/512 = 86.1328125` Hz. The nominal upsampling ratio is `441/128 = 3.4453125`, with integer output lengths per chunk. These are representation frame rates, not bounds on waveform frequency content.

The released [summary](../research/eval_summary.json) and [per-output records](../research/eval_per_item.json) support Tables 1-2 and Figures 3-4. [The figure script](figures/render_figures.py) validates the summary against those records, derives the paired counts, and exports SVG and PNG. The development analyses in Tables 3-4 are documented in the [historical lab notebook](../research/SCION.md). The [sample index](../samples/README.md) identifies the preview audio and generation settings.

## References

[1] M-A-P. *YuE2-3B*. Model card and inference implementation, 2026. `m-a-p/YuE2-3B` on Hugging Face.

[2] MiniMax. *MiniMax-Music3*. Model release, 2026. `MiniMaxAI/MiniMax-Music3` on Hugging Face.

[3] R. Oguz Araz, Xavier Serra, and Dmitry Bogdanov. *Discogs-VI: A Musical Version Identification Dataset Based on Public Editorial Metadata*. ISMIR, 2024. arXiv:2410.17400.

[4] Ciara Rowles et al. *Foley Control: Aligning a Frozen Latent Text-to-Audio Model to Video*. 2025. arXiv:2510.21581.

[5] Xiong Wang et al. *Freeze-Omni: A Smart and Low Latency Speech-to-speech Dialogue Model with Frozen LLM*. 2024. arXiv:2411.00774.

[6] Song-ha Jo et al. *Where Does the Sound Go? Tracing Acoustic Information Loss in Audio-Conditioned LLMs*. 2026. arXiv:2609.05871.

[7] Andy Zou et al. *Representation Engineering: A Top-Down Approach to AI Transparency*. 2023. arXiv:2310.01405.

[8] Edward J. Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*. 2021. arXiv:2106.09685.
