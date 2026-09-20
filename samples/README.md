# Listening examples

## Full-length covers of real songs

Two recordings, two target styles each, generated end to end at full length. SheetSage2 transcribed each source into a two-voice ABC score with chord annotations (`cot=full`), and the lyrics were supplied as text rather than recovered by ASR. Both covers of a song reuse that one score and that one lyric sheet, so the style prompt and the YuE2 seed are the only variables. No BPM appears in the prompts: the ABC carries its own `Q:` line, so tempo follows the original.

| File | Source | Style | YuE2 seed | Length |
|---|---|---|---:|---:|
| [`showcase/cover_ymca_rnb_fugue.mp3`](showcase/cover_ymca_rnb_fugue.mp3) | Y.M.C.A., Village People | Contemporary R&B | 2002 | 283.3 s |
| [`showcase/cover_ymca_britpop_fugue.mp3`](showcase/cover_ymca_britpop_fugue.mp3) | Same score | Britpop | 1147 | 284.0 s |
| [`showcase/cover_dadongbei_rnb_fugue.mp3`](showcase/cover_dadongbei_rnb_fugue.mp3) | 大东北我的家乡, He Yu | Contemporary R&B | 4725 | 246.4 s |
| [`showcase/cover_dadongbei_britpop_fugue.mp3`](showcase/cover_dadongbei_britpop_fugue.mp3) | Same score | Britpop | 5146 | 242.8 s |

Style prompts:

- **Y.M.C.A., R&B** — `contemporary R&B, neo-soul, smooth male lead vocal with stacked falsetto harmonies, Rhodes electric piano, muted wah funk guitar, deep round sub bass, crisp laid-back drums with swung hi-hats and finger snaps, warm sensual late-night groove, English vocal, 4/4`
- **Y.M.C.A., Britpop** — `britpop, 90s UK alternative rock, catchy overdriven electric guitar lead riff hook, jangly rhythm guitar, melodic bass, live drum kit, anthemic swaggering male vocal, Oasis and Blur energy, bright analog tape production, English vocal, 4/4`
- **Da Dong Bei, R&B** — `contemporary R&B, neo-soul, smooth Mandarin male lead vocal with stacked backing harmonies, Rhodes electric piano, muted funk guitar, deep round sub bass, crisp laid-back drums with swung hi-hats, warm soulful late-night groove, Mandarin vocal, 4/4`
- **Da Dong Bei, Britpop** — `britpop, 90s UK alternative rock, catchy overdriven electric guitar lead riff hook, jangly rhythm guitars, driving melodic bass, live drum kit, anthemic Mandarin male vocal, Oasis and Blur energy, bright analog tape production, Mandarin vocal, 4/4`

Each track is one generation with no splicing and no truncation: a YuE2 AR pass, a 32-step NAR pass, the v4 adapter, then 30 DiT steps and the MM3 vocoder at `dit_seed=7`. Wall clock was 171–204 seconds per track on a single RTX 4090.

Measured on the lossless renders, against the source recordings:

| Track | Energy above 8 kHz | 99% rolloff | Stereo side/mid | RMS |
|---|---:|---:|---:|---:|
| *Y.M.C.A., source* | 1.68% | 9.2 kHz | -12.5 dB | -13.3 dB |
| Y.M.C.A., R&B | 1.75% | 10.1 kHz | -5.9 dB | -17.8 dB |
| Y.M.C.A., Britpop | 0.89% | 7.6 kHz | -6.4 dB | -16.9 dB |
| *Da Dong Bei, source* | 1.26% | 8.8 kHz | -7.3 dB | -10.9 dB |
| Da Dong Bei, R&B | 1.49% | 9.5 kHz | -9.6 dB | -16.7 dB |
| Da Dong Bei, Britpop | 1.69% | 10.2 kHz | -6.3 dB | -17.2 dB |

The sources are commercial masters and the covers have no mastering stage, which accounts for most of the RMS gap. These four seeds were chosen by ear from a larger batch; the measurements above did not rank them, and the Britpop take of Y.M.C.A. is the darkest of the set on paper.

## One song, two arrangements

Start with the [s-AVE source excerpt](source_sAVE_excerpt.mp3), then listen to the [piano-and-strings cover](cover_sAVE_piano_fugue.mp3) and the [Britpop cover](cover_sAVE_britpop_fugue.mp3). The source is s-AVE by SawanoHiroyuki[nZk] with Aimer; the covers use its SheetSage2 transcription.

The [full 4:37 piano cover](cover_sAVE_piano_full_fugue.mp3) is a separate long-form generation, not a concatenation of the short demo. [To Know You](cover_toKnowYou_britpop_vocal_fugue.mp3) provides a 60-second vocal example with the first ten lyric lines supplied.

## Same latent, two decoders

Each `*_yue2_native.mp3` / `*_fugue.mp3` pair uses one YuE2-generated latent. The native version uses YuE2's VAE decoder; Fugue uses the adapter and MM3's DiT/vocoder. This holds the upstream generated representation fixed while changing the renderer.

| Pair | Source | Style prompt |
|---|---|---|
| `cover_sAVE_piano_*` | s-AVE recording, transcribed by SheetSage2 | intimate solo piano ballad with soft strings, 75 BPM, A minor, cinematic, instrumental |
| `cover_sAVE_britpop_*` | Same source recording | 1990s Britpop guitar rock, 75 BPM, A minor, driving drums, instrumental |
| `cover_toKnowYou_britpop_vocal_*` | To Know You, SawanoHiroyuki[nZk], with supplied lyrics | 1990s Britpop guitar rock, warm male lead vocal, driving drums |
| `britpop_A_*` | Transcribed MM3-generated Britpop track | 1990s Britpop guitar rock, 106 BPM, A harmonic minor, ... |
| `jazz54_B_*` | Transcribed MM3-generated 5/4 jazz instrumental | odd-meter jazz in 5/4, 102 BPM, A melodic minor, ... |

`cover_sAVE_piano_full_fugue.mp3` is the long-form exception to the paired naming convention; it has no native comparison in this directory.

## Format and provenance

The files are 160 kbps MP3 previews. The short instrumental pairs are approximately 30 seconds, the vocal pair 60 seconds, the full piano example 277 seconds, and the four `showcase/` tracks 243–284 seconds (4.7–5.4 MB each). Nothing here was remastered or loudness-matched. Numerical evaluation and the figures both used the generated WAV files, not these previews.

The source excerpt and covers of copyrighted songs are included as research demonstrations. Rights to the original compositions and recordings remain with their respective owners.
