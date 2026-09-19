# A/B samples — same YuE2 latent, two decoders

Each pair: `*_yue2_native.mp3` is YuE2's own VAE decode; `*_fugue.mp3` is the identical latent rendered through the Fugue
adapter → MiniMax-Music3 DiT + vocoder. Melody, harmony and structure are the same by construction; listen for the top
octave, transients and stereo width.

| pair | source | style prompt |
|---|---|---|
| `britpop_A_*` | ABC transcribed from an MM3-generated Britpop track | 1990s Britpop guitar rock, 106 BPM, A harmonic minor, … |
| `jazz54_B_*` | ABC from an MM3-generated 5/4 jazz instrumental | odd-meter jazz in 5/4, 102 BPM, A melodic minor, … |
| `cover_sAVE_piano_*` | real recording (s-AVE, SawanoHiroyuki[nZk]/Aimer) → SheetSage2 | intimate solo piano ballad with soft strings, 75 BPM, A minor, cinematic, instrumental |
| `cover_sAVE_britpop_*` | same recording | 1990s Britpop guitar rock, 75 BPM, A minor, driving drums, instrumental |
| `cover_toKnowYou_britpop_vocal_*` | real recording (To Know You, SawanoHiroyuki[nZk]) → SheetSage2, **with lyrics** (first 10 lines) | 1990s Britpop guitar rock, warm male lead vocal, driving drums — 60 s |

30 s each (the vocal pair 60 s), 160 kbps mp3. The `cover_*` pairs are covers of copyrighted songs produced from their
transcriptions; they are included as research demonstrations only.
