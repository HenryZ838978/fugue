"""Render the real-song showcase figures.

The mel panels come from the MP3s in `samples/showcase/`, so they reproduce
from this repository alone:

    python3 paper/figures/render_showcase.py

The spectrum panel also draws the source recordings and the YuE2-native decodes
of the same latents. Those are not redistributed here: the sources are
copyrighted, and the native decodes are working files. Their averaged spectra
are stored in `showcase_spectra.json` and reused when the audio is absent. To
recompute everything from local copies:

    python3 paper/figures/render_showcase.py --sources DIR --renders DIR

Curves stop at 16 kHz so that lossless renders and MP3 previews stay comparable
above 8 kHz without the MP3 encoder's own lowpass entering the picture.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SHOWCASE = ROOT / "samples/showcase"
SPECTRA = HERE / "showcase_spectra.json"
INK = "#20282c"
MUTED = "#59666d"
GRID = "#e1e6e8"
NATIVE = "#4b678e"
FUGUE = "#137e78"
ADAPTER = "#b14c57"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titleweight": "bold",
        "svg.fonttype": "none",
        "svg.hashsalt": "fugue-showcase-v1",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)

SONGS = [
    {
        "key": "ymca",
        "title": "Y.M.C.A.",
        "credit": "Village People",
        "source": "ymca.44k.flac",
        "feature": 1,
        "window": 64.0,
        "covers": [
            ("R&B", "cover_ymca_rnb_fugue.mp3", "ymca.rnb.s2002", FUGUE),
            ("Britpop", "cover_ymca_britpop_fugue.mp3", "ymca.britpop.s1147", ADAPTER),
        ],
    },
    {
        "key": "dadongbei",
        "title": "Da Dong Bei, My Hometown",
        "credit": "He Yu",
        "source": "dadongbei.44k.flac",
        "feature": 0,
        "window": 60.0,
        "covers": [
            ("R&B", "cover_dadongbei_rnb_fugue.mp3", "dadongbei.rnb.s4725", FUGUE),
            ("Britpop", "cover_dadongbei_britpop_fugue.mp3", "dadongbei.britpop.s5146", ADAPTER),
        ],
    },
]
FMIN, FMAX, POINTS = 40.0, 16000.0, 96


def log_grid():
    return np.logspace(np.log10(FMIN), np.log10(FMAX), POINTS)


def average_spectrum(path, n_fft=4096):
    """Time-averaged power spectrum in dB, pooled into 1/12-octave log bands."""
    y, sr = librosa.load(str(path), sr=44100, mono=True)
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=2048)) ** 2
    mean = spec.mean(axis=1)
    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    grid = log_grid()
    ratio = (FMAX / FMIN) ** (0.5 / (POINTS - 1))
    banded = np.empty(POINTS)
    for i, center in enumerate(grid):
        sel = (freq >= center / ratio) & (freq < center * ratio)
        banded[i] = mean[sel].mean() if sel.any() else np.interp(center, freq, mean)
    db = 10 * np.log10(banded + 1e-12)
    return [round(v, 2) for v in (db - db.max()).tolist()]


def collect_spectra(sources, renders):
    """Recompute whatever audio is available locally; keep the stored rest."""
    stored = json.loads(SPECTRA.read_text()) if SPECTRA.exists() else {}
    out = dict(stored)
    for song in SONGS:
        src = sources / song["source"] if sources else None
        if src is not None and src.exists():
            out[f"{song['key']}/source"] = average_spectrum(src)
        for _, name, tag, _ in song["covers"]:
            render = renders / f"{tag}.graft.wav" if renders else None
            out[f"{song['key']}/{name}"] = average_spectrum(
                render if render is not None and render.exists() else SHOWCASE / name)
            native = renders / f"{tag}.yue2vae.mp3" if renders else None
            if native is not None and native.exists():
                out[f"{song['key']}/{name}/native"] = average_spectrum(native)
    return out


def spectrum_figure(spectra, out_stem):
    grid = log_grid()
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.0), sharex=True, sharey=True)
    for row, song in zip(axes, SONGS):
        src = spectra.get(f"{song['key']}/source")
        for ax, (label, name, _, _) in zip(row, song["covers"]):
            ax.axvspan(8000, FMAX, color="#f5f7f8", zorder=0)
            native = spectra.get(f"{song['key']}/{name}/native")
            if native is not None:
                ax.plot(grid, native, color=NATIVE, linewidth=1.7,
                        label="YuE2-native decode", zorder=2)
            ax.plot(grid, spectra[f"{song['key']}/{name}"], color=FUGUE, linewidth=1.9,
                    label="Fugue", zorder=3)
            if src is not None:
                ax.plot(grid, src, color=INK, linewidth=1.3, linestyle=(0, (4, 2.5)),
                        label="Source recording", zorder=4)
            ax.set_xscale("log")
            ax.set_xlim(FMIN, FMAX)
            ax.set_ylim(-72, 4)
            ax.set_xticks([100, 1000, 4000, 8000, 16000])
            ax.set_xticklabels(["100 Hz", "1 kHz", "4 kHz", "8 kHz", "16 kHz"])
            ax.grid(True, which="major", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_title(f"{song['title']}  ·  {label}", fontsize=12, pad=8)
    for ax in axes[1]:
        ax.set_xlabel("Frequency")
    for row in axes:
        row[0].set_ylabel("Average power, dB below peak")
    axes[0][0].legend(frameon=False, fontsize=9.5, loc="lower left")
    fig.suptitle("Same latent, two decoders, four full-length covers",
                 fontsize=16, weight="bold", x=0.005, ha="left", y=1.045)
    fig.text(0.005, 0.995,
             "Time-averaged spectra in 1/12-octave bands. In the shaded band above 8 kHz Fugue sits "
             "at or above the native decode of the same latent. Both stay under the\ncommercially "
             "mastered source around 1–4 kHz; these covers are unmastered. Curves stop at 16 kHz so "
             "lossless renders and MP3 previews remain comparable.",
             fontsize=10.5, color=MUTED, ha="left", va="top", linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    for ext in ("png", "svg"):
        fig.savefig(f"{out_stem}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def mel_panel(ax, path, t0, window, title, color):
    y, sr = librosa.load(str(path), sr=44100, mono=True, offset=t0, duration=window)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512,
                                         n_mels=160, fmax=FMAX)
    db = librosa.power_to_db(mel, ref=np.max)
    ax.imshow(db, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0,
              extent=(t0, t0 + window, 0, FMAX / 1000))
    ax.set_title(title, fontsize=11.5, color=color, pad=6)
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_xlabel("Seconds")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def mel_figure(out_stem, sources, renders, window=40.0):
    """One style per song, decoded three ways over the same 40-second window."""
    fig, axes = plt.subplots(2, 3, figsize=(13.4, 6.2))
    for row, song in zip(axes, SONGS):
        label, name, tag, _ = song["covers"][song["feature"]]
        t0 = song["window"]
        columns = [
            (sources / song["source"] if sources else None, "Source recording", INK),
            (renders / f"{tag}.yue2vae.mp3" if renders else None, "YuE2-native decode", NATIVE),
            (SHOWCASE / name, f"Fugue, {label}", FUGUE),
        ]
        for ax, (path, caption, color) in zip(row, columns):
            if path is None or not Path(path).exists():
                ax.axis("off")
                ax.text(0.5, 0.5, f"{caption}\nnot redistributed", ha="center", va="center",
                        fontsize=10, color=MUTED, transform=ax.transAxes)
                continue
            mel_panel(ax, path, t0, window, caption, color)
        row[0].set_ylabel(f"{song['title']}\nkHz", fontsize=11, weight="bold", linespacing=1.7)
    fig.suptitle("The same forty seconds, three ways", fontsize=16, weight="bold",
                 x=0.005, ha="left", y=1.06)
    fig.text(0.005, 1.012,
             "Mel spectrograms of one section per song. Left: the recording that was transcribed. "
             "Middle and right: a single YuE2 latent, decoded natively and through Fugue.\n"
             "The arrangement is new in both covers, so the note content departs from the source by design. Plotted to 16 kHz, as above.",
             fontsize=10.5, color=MUTED, ha="left", va="top", linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{out_stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", type=Path, default=None,
                    help="Directory holding the source recordings (not redistributed)")
    ap.add_argument("--renders", type=Path, default=None,
                    help="Directory holding <tag>.graft.wav and <tag>.yue2vae.mp3")
    args = ap.parse_args()

    spectra = collect_spectra(args.sources, args.renders)
    SPECTRA.write_text(json.dumps(spectra, indent=1) + "\n")
    spectrum_figure(spectra, str(HERE / "real-song-spectra"))
    mel_figure(str(HERE / "real-song-covers"), args.sources, args.renders)
    print(f"wrote real-song-spectra.png/.svg, real-song-covers.png, "
          f"showcase_spectra.json ({len(spectra)} curves)")


if __name__ == "__main__":
    main()
