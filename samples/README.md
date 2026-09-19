# Listening examples

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

The files are existing 160 kbps MP3 previews. The short instrumental pairs are approximately 30 seconds, the vocal pair 60 seconds, and the full piano example 277 seconds. They were not remastered or loudness-matched for this documentation revision. Numerical evaluation used the generated WAV files, not these MP3 previews.

The source excerpt and covers of copyrighted songs are included as research demonstrations. Rights to the original compositions and recordings remain with their respective owners.
