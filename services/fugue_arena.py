"""Fugue Arena:上传一首歌 → 填目标 style(+歌词)→ 三列对比:原曲 / YuE2 原生(自家 VAE)/ Fugue(嫁接 MM3 DiT)。
薄客户端,只调 8650 / 8651 两个常驻服务。graphtokenizer env(有 gradio 6)。

用法: python fugue_arena.py --port 7860
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import gradio as gr
import requests

A = "http://127.0.0.1:8650"
B = "http://127.0.0.1:8651"
OUT = Path(os.environ.get("FUGUE_ARENA_OUT", "arena"))
OUT.mkdir(parents=True, exist_ok=True)

PRESETS = {
    "钢琴 ballad": "intimate solo piano ballad with soft strings, cinematic",
    "Britpop 摇滚": "1990s Britpop guitar rock, driving drums, deliriously euphoric",
    "爵士三重奏": "smooth jazz trio, upright bass, brushed drums, Rhodes piano",
    "Synthwave": "synthwave, retro analog synths, punchy drum machine, neon night drive",
    "民谣": "acoustic folk, fingerpicked guitar, warm and gentle, campfire",
    "管弦电影配乐": "orchestral film score, full strings and brass, epic and sweeping",
    "Lo-fi hip hop": "lo-fi hip hop, dusty drums, mellow Rhodes, vinyl crackle, chill",
    "重金属": "heavy metal, distorted guitars, double-kick drums, aggressive",
}
HELP = """**style 怎么填**(与 YuE2 一致,英文,逗号分隔的自由 tag):流派/年代 + 乐器 + 情绪/场景,可加 BPM、调式、拍号。
例:`1990s Britpop guitar rock, 106 BPM, A minor, driving drums, deliriously euphoric`。
BPM / 调式不填时沿用转谱出的原曲值。写 `instrumental` 或勾选"器乐"就不唱。

**lyrics 怎么填**:段落标记 `[Intro] [Verse] [Chorus] [Bridge] [Outro]`,每行一句,中英均可;器乐留空。"""


def to_mp3(wav, mp3):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "192k", str(mp3)], check=True)
    return str(mp3)


def run(audio, style, lyrics, instrumental, cot, seconds, seed, progress=gr.Progress()):
    if not audio:
        raise gr.Error("先上传一首歌")
    if not style.strip():
        raise gr.Error("style 不能为空")
    tag = re.sub(r"[^A-Za-z0-9_.-]", "-", Path(audio).stem)[:40] + f".{int(time.time())}"
    log = []
    t0 = time.time()
    progress(0.05, desc="SheetSage2 转谱中…")
    r = requests.post(f"{A}/transcribe", json=dict(audio_path=str(Path(audio).resolve())), timeout=1800).json()
    if not r.get("abc"):
        raise gr.Error(f"转谱失败: {r}")
    abc = r["abc"]
    m = re.search(r"(\d{2,3})\s*BPM", style)
    if m and not any(l.startswith("Q:") for l in abc.split("\n")):
        lines = abc.split("\n"); i = next((k for k, l in enumerate(lines) if l.startswith("L:")), 3); lines.insert(i + 1, f"Q:1/4={m.group(1)}"); abc = "\n".join(lines)
    if cot == "melody":
        abc = re.sub(r'"[^"]*"', "", abc)
    log.append(f"转谱 {r['sec']}s:{r.get('vocal_notes')} 人声音 / {r.get('instrumental_notes')} 器乐音 / {r.get('abc_measures')} 小节 / {r.get('key_lab', '')[:60]}")
    lyr = "[Instrumental]" if instrumental or not lyrics.strip() else lyrics.strip()
    progress(0.25, desc="YuE2 读谱生成中(AR + NAR)…")
    g = requests.post(f"{A}/generate", json=dict(style=style, lyrics=lyr, abc=abc, cot=cot, seed=int(seed),
                                                 max_sem=int(seconds * 25) if seconds > 0 else 9000, decode_yue2=True, tag=tag), timeout=7200).json()
    if "latent_path" not in g:
        raise gr.Error(f"YuE2 失败: {g}")
    log.append(f"YuE2 AR {g['t_ar']}s + NAR {g['t_nar']}s:{g['latent_frames']} 帧({g['latent_frames'] / 25:.0f}s)")
    progress(0.65, desc="Fugue 嫁接 MM3 DiT 渲染中…")
    gr_ = requests.post(f"{B}/graft", json=dict(latent_path=g["latent_path"], seed=7, steps=30, tag=tag), timeout=7200).json()
    if "wav_path" not in gr_:
        raise gr.Error(f"嫁接失败: {gr_}")
    log.append(f"MM3 DiT + vocoder {gr_['decode_sec']}s")
    progress(0.95, desc="转码…")
    src = to_mp3(audio, OUT / f"{tag}.src.mp3") if not audio.endswith(".mp3") else audio
    y = to_mp3(g["yue2_wav_path"], OUT / f"{tag}.yue2.mp3")
    f = to_mp3(gr_["wav_path"], OUT / f"{tag}.fugue.mp3")
    (OUT / f"{tag}.abc").write_text(abc)
    (OUT / f"{tag}.json").write_text(json.dumps(dict(tag=tag, style=style, lyrics=lyr, cot=cot, seed=seed, transcribe=r | {"abc": None}, yue2=g, graft=gr_), ensure_ascii=False, indent=1))
    log.append(f"总耗时 {time.time() - t0:.0f}s")
    return src, y, f, abc, "\n".join(log)


with gr.Blocks(title="Fugue Arena") as demo:
    gr.Markdown("# Fugue Arena — 真曲 cover:原曲 vs YuE2 vs Fugue(YuE2 读谱 × MiniMax Music 3 声学)")
    with gr.Row():
        with gr.Column(scale=1):
            audio = gr.Audio(label="上传要 cover 的歌(mp3/wav/flac/m4a)", type="filepath")
            preset = gr.Dropdown(list(PRESETS), label="风格预设(可改)", value="钢琴 ballad")
            style = gr.Textbox(label="style(英文 tag,逗号分隔)", value=PRESETS["钢琴 ballad"], lines=2)
            lyrics = gr.Textbox(label="lyrics(段落标记 + 每行一句;器乐留空)", lines=6)
            with gr.Row():
                instrumental = gr.Checkbox(label="器乐(不唱)", value=True)
                cot = gr.Radio(["full", "melody"], value="full", label="谱面模式(full 含和弦 / melody 仅旋律)")
            with gr.Row():
                seconds = gr.Slider(0, 300, value=60, step=10, label="生成时长 秒(0 = 整曲)")
                seed = gr.Number(value=0, label="seed", precision=0)
            btn = gr.Button("生成 cover", variant="primary")
            gr.Markdown(HELP)
        with gr.Column(scale=1):
            src_a = gr.Audio(label="① 原曲", type="filepath")
            yue_a = gr.Audio(label="② YuE2 原生(自家 VAE 解码)", type="filepath")
            fug_a = gr.Audio(label="③ Fugue(同一份 YuE2 latent → MM3 DiT + vocoder)", type="filepath")
            log = gr.Textbox(label="日志", lines=5)
            abc_out = gr.Textbox(label="SheetSage2 转出的 ABC 谱(喂给 YuE2 的)", lines=8)
    preset.change(lambda p: PRESETS[p], preset, style)
    btn.click(run, [audio, style, lyrics, instrumental, cot, seconds, seed], [src_a, yue_a, fug_a, abc_out, log])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=args.port, share=False, allowed_paths=[str(OUT)])
