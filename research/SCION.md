# SCION —— 把 YuE2 的符号侧嫁接到 MM3 的声学侧

> **scion** /ˈsaɪən/ 接穗：嫁接时取自母株的那段枝条。
> 砧木(rootstock) = MM3 的 DiT + Flow-VAE（声学质感）
> 接穗(scion)     = YuE2-3B 的 LM（已证明读得进 ABC 谱面）
>
> 立项 2026-09-19。前身是 `score_codec/` 的 r1~r4 四轮 ABC 通道实验（全部 ≈0）。

---

## 0. 一句话

**放弃在 MM3 的 qwen3-8B 上训读谱能力，改为把已经会读谱的 YuE2-3B 的 hidden
直接翻译成 MM3 DiT 期望的 condition，DiT 与 vocoder 全程冻结。**

---

## 1. 为什么埋掉 r1~r4

同一把 teacher-forcing c0 CE 尺子，四种设计，同一个零：

| 实验 | 摆放 | d_spec | 配对 |
|---|---|---|---|
| r1 / r2 | ABC 文本 BPE 当前缀 | ≈0 | — |
| 阶段 A | codec token 当前缀 | ≈0 | — |
| r3 | in-stream + prefix 双臂 | +0.0003 | 26/48 |
| r4 | YuE2 制度（plan loss + dropout + keep Q） | +0.0025 | 18/24 |
| **YuE2 对照** | **ABC 文本 BPE 当前缀（= r1 的摆放）** | **+0.340** | **24/24** |

比值 **136 倍**。r1 和 YuE2 摆放完全一样，所以摆放、token 空间、位置都不是瓶颈。

r4 把三条训练制度全补上了，`loss_plan` 从 23.12 降到 **0.49** —— 模型**学会了写 ABC**，
但 d_spec 仍只有 +0.0025。断点不在"看见"，在**"会写 ABC"到"用 ABC 写音频"之间**。

更早的 ablate 读数更难看：训练前 `base=2.1658` / `full=2.1955`，
**带 ABC 反而比不带差 0.030**，四档消融（full/other/flat/header）互相差在 0.002 以内 —— 
模型对 ABC 内容完全无差别响应。

### 已知陷阱（别再踩）

`instrumental` 子集的 `ruler(lyr_other - base) = +0.0000 (1/16)` **是伪影不是读数**：
3711 首 instrumental 的 lyrics 字段字面全是同一个字符串 `[instrumental]`，
"换成别的歌的 lyrics" 换了个一模一样的东西，对照组退化了。
任何基于 lyrics 通道的寄生/对照设计在 instrumental 上都无效。

### 剩下的自变量

四种设计穷尽之后，唯一没变过的是**模型本身**。SCION 就是换模型。

---

## 2. 机制测绘：MM3 的 LLM→DiT 桥

整座桥只有一个类，比预想的简陋得多：

`.../diffusers/models/condition_embedders/condition_embedder_minimax_music3.py`

```python
class MiniMaxMusic3ConditionEncoder(ModelMixin, ConfigMixin):
    def __init__(self, condition_hidden_dim=4096, num_condition_layers=8, out_dim=2048,
                 input_sampling_rate=24000,  input_hop_length=960,     # 25 Hz
                 output_sampling_rate=44100, output_hop_length=512):   # 86.13 Hz
        self.layer_weight_logits = nn.Parameter(torch.zeros(num_condition_layers))
        self.layer_scale         = nn.Parameter(torch.ones(1))
        self.proj                = nn.Conv1d(condition_hidden_dim, out_dim, kernel_size=3, padding=1)

    def forward(self, hidden_states):          # (B, frames, 8 * 4096)
        # → reshape (B, 8, 4096, T)
        layer_weights = torch.softmax(self.layer_weight_logits, dim=0)
        hidden_states = torch.einsum("blht,l->bht", hidden_states, layer_weights)
        hidden_states = self.layer_scale * hidden_states
        hidden_states = self.proj(hidden_states)
        hidden_states = F.interpolate(hidden_states, size=latent_length, mode="nearest")
        return hidden_states.transpose(1, 2)   # (B, latent_len, 2048)
```

### 实测权重（`condition_encoder/diffusion_pytorch_model.safetensors`）

```
softmax(layer_weight_logits) = [0.9061, 0.0130, 0.0126, 0.0128, 0.0140, 0.0135, 0.0137, 0.0142]
                                  ↑LM      ↑———— depth 的 7 层，合计 9.4%，几乎均匀 ————↑
layer_scale = 0.06851457059383392
proj.weight = (2048, 4096, 3)   proj.bias = (2048,)
```

三条读数：

1. **DiT 的条件 90.6% 来自 qwen3-8B 的 hidden。** 桥上没有第二个信息源。
2. **depth 的 7 层合计只占 9.4%，而且几乎均匀**（1.26%~1.42%）。均匀 = 模型没去区分它们
   = RVQ depth 那一段基本是残渣。**这半个直觉是对的。**
3. **`layer_scale = 0.0685`** 把条件整体压到 6.85% 幅度，再用 **nearest** 上采样
   （每 **3.445** 个 latent 帧共享同一个条件向量）。

### 正确的分工表述

> **音色的"身份"在 qwen3 hidden 里，音色的"质感"由 DiT + Flow-VAE 承担。**

DiT 拿到的是一张**低幅度（6.85%）、有效带宽只有 25Hz 的阶梯状粗路线图**，
86Hz 级别的所有声学细节它自己填。

**这正是手术能成立的理由**：要换掉的是"身份"那一层，"质感"那一层不用动。

---

## 3. 25Hz 是从哪儿来的（Henry 的洞察，2026-09-19）

> 人眼 24Hz → videogen 全是 25Hz → **音乐组基本上复用了视频组的数据管线**。

MM3 的 LM 帧率 `24000/960 = 25 Hz` 不是音乐学上的选择，是**视频管线的遗产**。

两个推论：

**(a) 25Hz 对音乐是个尴尬的帧率。** 120 BPM 下一个十六分音符 = 125 ms = **3.125 帧**，
不是整数。任何需要精确对齐节拍网格的符号信息压进这个通道都会被量化误差磨掉。
这可能是 ABC 的时间信息死在 r1~r4 的物理原因之一 —— 不是模型不想学，是通道不承载。

**(b) YuE2 的 VAE 也是 `48000/1920 = 25 Hz`。** 两边大概率撞上了同一份视频遗产。
**这是巧合，但是可利用的巧合** —— SCION 的时间轴对齐问题因此不存在。

---

## 4. 能做到吗 —— 能，接口对齐得反常

| | MM3 | YuE2 |
|---|---|---|
| LM hidden_size | 4096 | **2048** |
| DiT `condition_dim` | **2048** | — |
| LM / latent 帧率 | 24000/960 = **25 Hz** | 48000/1920 = **25 Hz** |
| VAE latent_dim | 128 | 64 |
| 采样率 | 44100 | 48000 |
| 层数 | 36 (LM) / 36 (DiT) | 28 |

- YuE2 的 hidden 宽度**正好等于** DiT 的 `condition_dim` = 2048
- 两边帧率**都是 25Hz**，时间轴天然对齐，插值逻辑直接抄 MM3 自己那行 `F.interpolate(mode="nearest")`

> **⚠️ 2026-09-19 开工后修正(见 §13)**:YuE2 的 **semantic audio tokenizer 未开源**(README / modeling_vae.py 明说),
> 任意音频拿不到 semantic token,也就拿不到 AR hidden。两侧能从**同一段音频**得到的 YuE2 表征只有 **VAE latent(64 维 @25Hz)**,
> 它同时是推理时 NAR ODE 的输出。接口因此定为 **latent64@25Hz → adapter → c25(2048@25Hz)→ MM3 自己的 nearest ↑3.445**。
> 下面"hidden 2048 = condition_dim 2048"的巧合不再是接口依据,保留作历史。

### 手术点

**YuE2 侧** —— `m-a-p--YuE2-3B/snapshots/master/modeling_yue2.py`

```python
692:  hidden_states, _ = self.model(...)
698:  nar_pred = self.llm2vae(hidden_states)   # [B, S, 64]  ← 拆掉 VAE = 不走这一层
```

"拆掉 YuE2 的 VAE"在代码上就是**不走 `llm2vae`，改走 adapter**。

**MM3 侧** —— 整体替换 `MiniMaxMusic3ConditionEncoder.forward()`，
输入从 `(B, T, 8×4096)` 改成 YuE2 的 `(B, T, 2048)`，其余（scale / interpolate）保留。

**冻结**：MM3 DiT（36 层）+ vocoder 全冻。YuE2-3B 先全冻，只训 adapter。

### 注意 chunking

`.../modular_pipelines/minimax_music3/before_denoise.py`：
AR frames 以 `_CHUNK_FRAMES = 200` / `_CHUNK_HOP = 100` 的窗口解码，
相邻窗重叠约 344 latent 帧，裁波形时保留前一窗尾部 86 个 latent 帧。
adapter 的输出要能塞进这个 chunk 逻辑，或者绕开它整条跑。

---

## 5. 有先例吗 —— 有，一个几乎同构

| 论文 | 关系 |
|---|---|
| [**Foley Control**](https://arxiv.org/pdf/2510.21581) | **最贴。** Stable Audio 的 DiT 完全冻结，只训 adapter 把视频编码器的 embedding 经 cross-attention 接进去。把 video encoder 换成 YuE2 = SCION。 |
| [**Freeze-Omni**](https://arxiv.org/html/2411.00774v1) | LLM hidden 驱动语音 decoder，LLM 全程冻结。**trick 值得偷**：用 NAR prefix decoder 把 hidden 转成 kv-cache 喂下游，连维度匹配的投影都省了。 |
| [**DITTO-TTS**](https://proceedings.iclr.cc/paper_files/paper/2025/file/80e77d9ed2f74dcaf1a42cb1a2593559-Paper-Conference.pdf) | **反向解法**：微调音频 codec 去对齐 LM 的 hidden 空间。留作备选（如果 adapter 方向训不动，就反过来微调 DiT 的 condition 入口）。 |
| [**DIFFA**](https://arxiv.org/pdf/2507.18452) | adapter 的具体配方：2 层 conv + 2 层 linear 做语义对齐，2 层 Q-Former 做声学，**分两阶段训**。 |
| [**LLM-Codec**](https://arxiv.org/pdf/2604.17852) | Gumbel-Softmax 量化桥，另一种接法。 |

### 两条警告

- [**Tracing Acoustic Information Loss in Audio-Conditioned LLMs**](https://arxiv.org/html/2609.05871)
  薄 adapter 常常不够，**readout alignment 本身就是主瓶颈**。别指望一个 Conv1d 解决问题。
- [**ALAS**](https://arxiv.org/pdf/2505.19937)
  **该抽哪一层很关键** —— Qwen2-Audio 的峰值对齐在第 **22** 层而非最后一层。
  **YuE2 有 28 层，别默认用最后一层，必须扫。** 这是第一阶段最便宜的超参。

---

## 6. 数据齐备吗 —— 齐备，但路径不是直觉那条

### 坏消息

**MM3 的 RVQ 量化器（音频 → c0..c7）没开源。**
`dav.pth`（492 MB，548 keys，前缀 `encoder.block.*`）是 **Flow-VAE**，
编出的是**连续 latent 而非离散码**（DEVLOG:44 已记录）。

→ 「任意真实音乐 → MM3 condition」这条路**不通**。

### 好消息：不需要它

adapter 的训练语料可以全部来自**自生成配对**，而这批数据已经在盘上：

```
/cache/zhangjing/musicodec_data/mm3-rvq-distill-corpus-8k/  →  pairs8k
  codes.i16.npy      (17191816, 8) int16   ← 1719 万帧 / 25Hz ≈ 191 小时，全 8 层
  walk.i16.npy       (2949443,)            ← Euler-walk 前缀 token（可逆，OCR 目标）
  walk1600.i16.npy   (2614185,)            ← maxT=1600 帧(64s) 截断重编
  timeline.i16.npy   (3865360, 2)          ← [frame, token]，25Hz 起音帧对齐的 in-stream 视图
  12837 首，gate: instrumental 3711 / three_ok 8121 / full_only 1005
```

> **⚠️ 修正(2026-09-19)**:下图 input 侧 "latent → YuE2-3B forward → h" 不成立(AR 需要 semantic token,见 §4 callout)。
> 实际 input = YuE2 VAE latent 本身;target 侧不变。另外**"任意真实音乐 → MM3 codes"并非完全不通**:社区 RVQ encoder
> (scragnog hotstep v1 + dav.pth,`/cache/zhangjing/models/open-rvq-encoder-minimax-music3--SimpleTuner`,mm3_continue.py 在用)是近似路径。

**两侧都绕开了量化器**：

```
自生成音频 A（wav + codes + caption/lyrics + ABC 全都在盘上）
 │
 ├─ target ：codes ──teacher-forcing LM+depth──▶ 8层 hidden ──ConditionEncoder──▶ c (2048 @86Hz)
 │           ↑ 码是 MM3 生成时自带的，不需要量化器
 │
 └─ input  ：A ──YuE2 VAE encode(开源)──▶ latent(64) ──YuE2-3B forward──▶ h (2048 @25Hz)
             ↑ YuE2 VAE 权重完整可用

训练目标：adapter(h) ≈ c      纯 L2 回归
不跑 DiT、不合成音频、零人工标注
```

**191 小时配对语料，零标注成本。**

### 域差不咬人

DEVLOG:74 记过「自生成音频与真实录音有域差（缓解：真实音频先过 dav VAE 重建）」。
对 adapter 训练**这个域差反而是共模的** —— adapter 学的是两个表征空间之间的映射，
两侧喂的是**同一段音频**，域差在输入输出上同时出现，被抵消。

真正会咬人的是**推理时**：用真实音乐（或用户哼唱）驱动 YuE2 时，
h 的分布与训练时的自生成音频不同。缓解手段同 DEVLOG：真实音频先过 dav VAE 重建一遍。

### 其他可用资产

```
ood216 : /cache/zhangjing/fugue/ss2/out/*/events.json   217 首人工 pick 的真音频，无 codes
         → 只能做 score-only 评测，不能做 adapter target
selfdistill/v2/ : probe_v0.pt / probe_v01.pt / probe_reader.pt / ar_batch_gen.py
         → 逆向 tokenizer 线（mel → 8 头分类）的 checkpoint，最新 Sep 11
         → 如果哪天需要"真实音频 → MM3 codes"，这条线是补量化器的方案
```

---

## 7. 最锋利的一问：你到底要 MM3 的什么

这个矛盾必须在动手前回答：

| 想要的 | 在哪 | 手术结论 |
|---|---|---|
| **vocoder / DiT 的音质**（保真、立体声、高频质感） | DiT + Flow-VAE，不受 condition 内容影响 | **手术成立** |
| **MM3 的编曲 / 音色审美** | 正是 qwen3 hidden 里的那 90.6% | **手术自相矛盾**（换成 YuE2 就没了） |

如果答案是第二行，SCION 在逻辑上就死了 —— 你要保留的东西正是你要换掉的东西。

---

## 8. 判决实验（零训练，不碰 GPU3）

### 实验 A：条件消融 —— DiT 到底承担了多少

拿一条已有的自生成样本，`frame_hiddens` 三种处理喂 DiT + vocoder：

1. **原样**（基线）
2. **`layer_scale` 置 0** —— 条件完全切断，纯 DiT 先验
3. **换成别的歌的 hidden**

然后听：

- ② 仍是好音质、只是内容乱 → **DiT 确实是音质来源，手术成立**，且收益上限就是这段差值
- ② 直接崩 → **音质本身编码在 condition 里**，adapter 要学的远不止翻译，
  得按 DIFFA 那种两阶段（语义 adapter + 声学 adapter）来

约 20 分钟。**这是最直接验证第 7 节假设的实验。**

### 实验 B：YuE2 VAE 自身重建质量

同一段音频走 YuE2 VAE `encode → decode`（48kHz），听重建保真度。

**如果 YuE2 VAE 重建已经够好听，这台手术就没必要做** —— 直接用 YuE2 全链路，
省掉整个 adapter。SCION 的全部价值建立在「MM3 的声学侧明显优于 YuE2 的声学侧」这个前提上，
这个前提**从没被验证过**。

> A 和 B 决定的是「做不做」，不是「怎么做」。先跑这两个。

---

## 9. 若判决通过：实施路线

### 阶段 0 —— 层扫描（最便宜的超参）

按 ALAS 的教训，扫 YuE2-3B 的 **28 层**，每层抽 h，用线性探针对 MM3 的 c 做回归，
看哪一层的 R² 最高。别默认最后一层。

### 阶段 1 —— 薄 adapter 基线

```
h (B, T, 2048) @25Hz  ──▶  Conv1d(2048→2048, k=3)  ──▶  ×layer_scale  ──▶  nearest ↑3.445  ──▶  c (B, L, 2048) @86Hz
```

抄 MM3 自己的结构，L2 回归，先看能不能拟合。
**预期：不够**（Tracing Acoustic Information Loss 的警告）。但这是必需的对照基线。

### 阶段 2 —— DIFFA 配方

语义 adapter（2 conv + 2 linear）+ 声学 adapter（2 层 Q-Former），分两阶段训。

### 阶段 3 —— 端到端听感

冻结 DiT + vocoder，跑完整链路：ABC → YuE2-3B → adapter → DiT → vocoder → wav。
用已有的 **gen→retrans→followscore 闭环**（DTW 归一化音程距离）评 follow 程度。

### ⚠️ 闭环尺子尚未校准

`ss2/gen_base/followscore.json`（n=12, seed=7, cfg=1.5, 30.02s）：

| var | dtw中位 | pc_cos中位 | n_gen中位 |
|---|---|---|---|
| full | 0.889 | 0.487 | 66 |
| flat | 1.097 | 0.612 | 65 |
| noabc | 0.905 | **0.634** | 54 |

配对 full < noabc **6/12** = 抛硬币。**但我们不知道同条件换 seed 的 dtw 方差是多少**，
所以这个 6/12 说明不了任何事。

**先决任务**：3 首 × full 条件 × 5 seed，算 within-condition dtw 的 σ，
定出噪声底，再看 +0.xx 的差值有没有意义。GPU 0/6/7 空着。

---

## 10. 风险清单

| # | 风险 | 缓解 |
|---|---|---|
| 1 | **teacher-forcing hidden ≠ 自由生成 hidden**（exposure bias） | selfdistill 已有截获自由生成 `frame_hiddens` 的能力，两种混训 |
| 2 | **跨音色空间翻译，一个 Conv1d 大概率不够** | 阶段 1 只作基线，预留 DIFFA 配方 |
| 3 | **MM3 的音质优势可能不在 DiT 上**（第 7 节的矛盾） | **实验 A/B 先验，这是前置条件不是风险** |
| 4 | 推理时真实音频与自生成语料的域差 | 真实音频先过 dav VAE 重建（DEVLOG:74） |
| 5 | chunking 逻辑（200/100 窗）与 adapter 输出不兼容 | 先绕开整条跑，跑通再适配 |
| 6 | 该抽 YuE2 哪一层未知 | 阶段 0 扫描，便宜 |
| 7 | followscore 闭环无噪声底，评不出真假 | 先校准 σ（GPU 0/6/7） |

---

## 11. 资产与路径速查

### 模型

```
MM3          /cache/zhangjing/models/MiniMax-Music3/
  transformer/       MiniMaxMusic3Transformer1DModel  36层 heads=32 head_dim=64
                     condition_dim=2048  in_channels=128  ff_inner_dim=8192  rotary_dim=32
  vocoder/           latent_channels=128  sr=44100  upsampling=[8,8,4,2]=512
                     decoder_input_dim=1024  decoder_hidden_dim=1536
  condition_encoder/ ← 手术点，见 §2
  rvq_depth_decoder/ hidden=4096  num_codebooks=8  num_layers=4  audio_vocab=1024  max_pos=16
  language_model/    hidden=4096  36层  vocab=200000
  dav.pth            492MB 548 keys  encoder.block.*  ← Flow-VAE，不是量化器
  flowmatching_vae.pth  9.8GB 374 keys  含 cond_layer_logits / cond_layer_scale / diffusion_transformer.*
  qwen_7B/           AbabForCausalLM 原生格式

YuE2-3B      /cache/zhangjing/models/_ms/models/m-a-p--YuE2-3B/snapshots/master/
  YuE2ForCausalLM  hidden=2048  28层  heads=16  kv_heads=8  head_dim=128
  intermediate=6144  vocab=184704  rope_theta=1e6  max_pos=24576
  latent_type="vae"  latent_dim=64  max_latent_frames=24576  timestep_shift=1.0
  modeling_yue2.py:468  lm_head
  modeling_yue2.py:692  hidden_states, _ = self.model(...)
  modeling_yue2.py:698  nar_pred = self.llm2vae(hidden_states)   ← 手术点

YuE2-Vae     /cache/zhangjing/models/_ms/models/m-a-p--YuE2-Vae/snapshots/master/
  YuE2VAE  sr=48000  downsampling_ratio=1920 (→25Hz)  latent_dim=64  audio_channels=2
  encoder latent_dim=128 / decoder latent_dim=64
  strides=[2,2,4,4,5,6]  c_mults=[1,2,4,8,16,32]
  decode_core_frames=1024  decode_halo_frames=16
  内部 OobleckEncoder / OobleckDecoder / SnakeBeta
  接口 encode(audio, sample=False, return_info=False) / decode / decode_tiled / decode_audio / forward
```

### 数据

```
pairs8k      /cache/zhangjing/musicodec_data/mm3-rvq-distill-corpus-8k/data
             构建脚本 /cache/zhangjing/fugue/score_codec/build_pairs8k.py
             12837 首  codes(17191816,8) ≈191h
ood216       /cache/zhangjing/fugue/ss2/out/*/events.json   217 首真音频，无 codes
out8k        /cache/zhangjing/fugue/ss2/out8k               FPS=25
selfdistill  /cache/zhangjing/fugue/selfdistill/v2/
```

### 前史脚本

```
/cache/zhangjing/fugue/score_codec/abc_lora4.py     r4 训练（GPU3，跑完为止，勿动）
/cache/zhangjing/fugue/score_codec/yue2_ruler.py    YuE2 对照尺子（+0.340 的来源）
/cache/zhangjing/fugue/score_codec/RUNS/r3_instream/eval_step3000.log
/cache/zhangjing/fugue/score_codec/RUNS/r4_yue2regime/train.log
/cache/zhangjing/fugue/ss2/ablate_{base,step1000,step2000}.json
/cache/zhangjing/fugue/ss2/gen_base/followscore.json
/cache/zhangjing/fugue/DEVLOG.md     §40-78 是数据齐备性的核心约束
```

### ⚠️ 服务器禁区

```
mcp/musicodec_service.py     conda env graphtokenizer  端口 8643
  依赖 musicodec_data/（SQLite + 内容寻址 store）        不可删
vLLM musicodec-reader        端口 8642
  依赖 MusicGraph/model/merged_v8_2_Llama-3.1-8B-Instruct/  不可删
GPU3  r4_yue2regime 在跑，让它跑完，不可打断
```

---

## 12. 待办（按优先级）

- [x] **实验 A**(§13.3:DiT 对 30% 条件噪声鲁棒)：条件消融（`layer_scale`=0 / 换歌 hidden），判定 DiT 是否是音质来源 — 20 min
- [x] **实验 B**(§13.4:MM3 VAE SI-SDR 17.7 vs YuE2 7.7 dB)：YuE2 VAE encode→decode 重建听感，判定手术有无收益空间 — 20 min
- [ ] **校准 followscore 噪声底**：3 首 × full × 5 seed，算 within-condition dtw σ（GPU 0/6/7）
- [x] 阶段 0：改判为接口 = VAE latent(§13.1),层扫描不适用;做了移位 / 上下文 / DAV 对照探针(§13.5-13.6)
- [x] 阶段 1：线性上限 R²var 0.38;Transformer adapter v1–v3 到 0.67(§13.9)
- [ ] 阶段 2：DIFFA 两阶段配方
- [x] 阶段 3：端到端 + followscore(§13.7b / §13.9);σ 校准仍欠
- [ ] （可选，诊断 r4）把 `loss_plan` 按 token 类别拆成 `[A-Ga-g]` 音高字母 vs `|/:数字` 结构符号，
      若音高部分没降则 r4 的制度①是假生效

---

## 13. 开工日志 —— 2026-09-19(r4 判为阴性对照,SCION 直接开工)

代码:本地 `Fugue/scion/`(source of truth)↔ 服务器 `/cache/zhangjing/fugue/scion/`。产物都在后者。

### 13.1 接口改判:接的是 YuE2 的 VAE latent,不是 LM hidden

读 YuE2 源码(`yue2/nar.py`, `modeling_yue2.py:600-705`)后两条事实:

1. YuE2 的 NAR **不是** `hidden → llm2vae` 一次回归,而是 LM 内部的 **flow-matching 速度场**(MoT 路由:每层独立的
   `nar_self_attn / nar_mlp`,NAR 位置注入 `vae2llm(x_t) + t_emb + pos_emb`,`llm2vae(norm(x))` 输出速度,32 步 midpoint)。
   692 行的 hidden 随 ODE 时间步与噪声变,不是一份干净的 condition。
2. **semantic audio tokenizer 未开源**(modeling_vae.py 原话 "This audio VAE is not the unreleased semantic audio tokenizer")。
   → 任意音频 → semantic token → AR hidden 这条路对配对训练不存在。

两侧都能从同一段音频拿到的 YuE2 表征只剩 **VAE latent(64 维 @25Hz,encoder 开源)**,而它正是推理时 NAR ODE 的输出。
所以:

```
训练:MM3 自生成 flac ─44.1k→48k─▶ YuE2 VAE encode ─▶ z (T',64)      ┐
      codes + caption/lyrics ─teacher-forcing LM+depth─▶ ConditionEncoder 去掉 interpolate ─▶ c25 (T,2048)  ┘ adapter(z) ≈ c25
推理:ABC+tags+lyrics ─▶ YuE2 AR ─▶ semantic ─▶ NAR ODE ─▶ z ─▶ adapter ─▶ c25 ─▶ nearest ↑3.445 ─▶ DiT ─▶ vocoder
```

YuE2 生成的 latent std 0.945,VAE 编码真实音频的 latent std 0.89–1.01 —— 同一尺度,没有幅度失配。

### 13.2 teacher-forcing 提取器已验证等价于生成时 hidden

`mm3_tf.py`:codes 喂回 LM(prompt + 逐帧 embedding,一次前向)+ depth 解码器(每帧 8 位置 causal,teacher-forced),
hidden 布局与 ar_batch_gen 的 hidden.pt 同;`cond25()` 复现 ConditionEncoder 到 proj 为止,与官方前向 bit 级一致
(nearest 上采样 max|Δ|=0;唯一差异是 Conv1d k=3 在 200 帧窗边界的 zero-pad,只影响每窗首尾一帧,而这些帧在拼接时被裁掉)。
0.34 s/首,全库 12837 首 ≈ 75 min 单卡。

**验证**:TF hidden 重新解码 vs 原 flac 的 log-mel L1 = **0.686**;同条件换 seed(7 vs 8)= **0.698**。二者相等 → 提取无损。

### 13.3 实验 A(条件消融,mf-1-000415 Britpop 有人声,前 30s)

| 变体 | log-mel L1 vs 原 flac | 读法 |
|---|---|---|
| orig(TF hidden) | 0.686 | = seed 噪声底 0.698 |
| **noise10**(c25 加 10% 方差白噪) | 0.697 | 与噪声底无差 |
| **noise30**(30%) | 0.729 | 仅 +0.04 |
| zero(condition 置零 = 纯 DiT 先验) | 2.279 | 内容全变 |
| swap(换歌 hidden) | 5.077 | ≈ 两首歌本身距离 5.114 |
| davrecon(vae.npy → vocoder,不走 DiT) | 0.210 | MM3 VAE 重建上限 |

**DiT 对 condition 误差相当鲁棒**:30% 方差的噪声几乎听不出谱距离变化。这是 adapter 不必完美的依据。
音频回传本地 `Fugue/scion/listen/`(mp3),待耳判 zero 是否"音质好内容乱"。

### 13.4 实验 B(声学侧对照)—— 前提成立

同段音频两套 VAE encode→decode(`expB_yue2vae.py` / `expB_mm3vae.py` / `expB_metrics.py`,统一 44.1k 比较):

| 段 | 系统 | log-mel L1 | lo<4k | mid 4-10k | hi>10k | SI-SDR |
|---|---|---|---|---|---|---|
| MM3 自生成 | YuE2 VAE | 0.669 | 0.74 | 0.54 | 0.62 | 3.2 dB |
| MM3 自生成 | MM3 Flow-VAE | 0.210 | 0.11 | 0.27 | 0.39 | 20.5 dB |
| 真曲 s-ave | YuE2 VAE | 0.733 | 0.76 | 0.84 | 0.53 | 7.7 dB |
| 真曲 s-ave | MM3 Flow-VAE | 0.997 | **0.23** | 0.82 | 3.18* | **17.7 dB** |

*真曲源是 AAC(≈16k 截止),MM3 解码器补出了高频,log 域被放大,属指标伪影。
MM3 声学侧在低频段与波形保真上大幅领先。DAV encoder 现场编码 vs zip 内 vae.npy 差 0.0005,加载正确。

### 13.5 对齐坑:MM3 音频不是均匀时间轴

200 帧窗 / 100 帧 hop,每窗 int(200×3.4453)=689 latent,拼接后每 100 帧占 **345** latent(名义 344.53),
第 k 窗比名义时刻晚 0.47k latent ≈ **0.136k 个 YuE2 帧**;105s 的歌尾部差 3 帧(YuE2 2628 帧 vs MM3 2625 发射帧,实测吻合)。
`data.align_index()`:m_j = round(j + 0.1358·k(j)),k = 0 if j<125 else (j-25)//100。

线性探针(w=1)证明这不是可忽略的细节:**stitched 0.347 vs naive 0.262**。移位扫描峰在 shift=0(+1 为 0.336)。

### 13.6 阶段 0/1:线性上限与第一个 adapter

线性 ridge(YuE2 latent ±w 帧 → c25,300 首 val 训 / 100 首 val 测):

| w(上下文帧) | 0 | 1 | 2 | 4 | 8 |
|---|---|---|---|---|---|
| R²var | — | 0.347 | 0.371 | 0.381 | **0.384** |
| R²ch(逐通道均值) | — | 0.115 | 0.135 | 0.146 | 0.150 |

诊断:MM3 自家 DAV latent(128 维 @86Hz,按拼接时间轴池化到 25Hz)做同样探针只有 **0.14–0.20** ——
YuE2 latent 作为接口比 MM3 自己的波形级 latent 更"语义",没吃亏。

`train_adapter.py`(Linear → 2×Conv k5 → 深度可分 conv 相对位置 → N 层双向 Transformer → Linear,标准化空间 MSE):
v1 = 6 层 d512(23.7M),仅 1050 首 train(整库还在提取),12k 步:R²var **0.593** / R²ch 0.369(step 7000 平台,过拟合)。
c25 每通道 std 0.125–4.32 重尾;R²var 由大方差通道主导,是 DiT 实际看到的尺度。

### 13.7 第一次嫁接解码(`graft_decode.py`,adapter v1 step4000 R²var 0.58)

- 经嫁接的重建(mf-1-000415 的 MM3 音频 → YuE2 VAE → adapter → DiT):该曲 R²var 0.446;graft vs 原 flac **0.867**
  (TF-orig 0.686,noise30 0.729)。结构化的 adapter 残差比同量白噪更伤,但这是千首数据的第一版。
- 真接穗(`yue2_gen.py`:同一份 SheetSage2 ABC + style + lyrics → YuE2 → 750 帧 latent,AR 15s + NAR 3s)→ graft:
  `graft/mf-1-000415.yue2full.s0.v1e.graft.mp3`,与 YuE2 自家 VAE 解码 `...yue2vae.mp3` 同 latent 对听。已回传本地。

### 13.7b 闭环尺子:YuE2 的读谱能力穿过嫁接活下来了(`follow/` + ss2/retrans.py + scion/followscore.py)

各变体 30s wav → SheetSage2 回转谱 → 与输入 ABC 的 DTW 归一化音程距离(越小越跟谱;0 = 转谱完全一致):

| mf-1-000415(Britpop)| Ins 声部 dtw(94 音)| Vocal 声部 dtw(34 音)|
|---|---|---|
| ref(原 flac 回转谱) | 0.000 | 0.200 |
| tforig(TF hidden 重解码) | 0.065 | 0.200 |
| graftrecon(原曲 → YuE2 VAE → adapter v1 → DiT) | 0.065 | **0.197** |
| **yue2full**(YuE2 按 ABC 生成,自家 VAE) | **0.064** | (只唱了 7 个音,n 太小) |
| **graftfull**(同一 latent → adapter v1 → DiT) | **0.063** | (5 个音) |
| graftfullre(latent 先 decode→encode) | 0.064 | — |
| yue2off / graftoff(不给谱) | 1.081 / 1.006 | 0.515 / 0.456(pc_cos 0.34) |
| zero(纯 DiT 先验) | 0.527 | — |

| mf-1-001136(5/4 爵士器乐)| Ins dtw(98 音)|
|---|---|
| ref / tforig / graftrecon | 0.055 / 0.178 / 0.231 |
| yue2full / graftfull | 0.405 / 0.473 |

三条读法:
1. **嫁接不丢谱**:Ins 线 yue2full 0.064 → graftfull 0.063;爵士 0.405 → 0.473(略降,仍远好于无谱的 ~1.0)。
2. **嫁接重建 = 同条件换 seed**:graftrecon 的 Vocal dtw 0.197 与 tforig 0.200 相同 —— adapter v1(R² 0.58)的残差没伤到旋律。
3. 无谱对照 dtw ≈ 1.0、zero 0.53:尺子能分辨。这正是 §9 说"闭环尺子尚未校准"时缺的那组对照。

chroma 一致性(CQT 12 维逐帧余弦超出打乱基线的幅度;同条件换 seed 的天花板 +0.055):
graftrecon vs ref +0.053;同 latent 的 yue2vae 解码 vs graft 解码 +0.047(Britpop)/ +0.439(爵士);zero/swap ≈ +0.008。

真曲 cover 首试(s-AVE,ood216 SheetSage2 谱,`yue2_gen.py --abc_file`,style 换成钢琴 ballad / Britpop):
YuE2 原声 vs 嫁接 chroma +0.36 / +0.34;音频已回传(`listen/s-ave.*`)。跟真曲旋律的 DTW 待 retrans。


### 13.9 全量 adapter:v2 / v3 / v4

整库两侧提取完成(12837 首 × 两侧,0 失败;c25 66GB fp16,yue2lat 2.2GB;TF 52.7 min,VAE 73.9 min 各单卡)。
三个 66.6M 的 adapter(8 层 d768,crop 768 帧,bs 24,20k 步 ≈ 66 min):

| run | 差异 | val R²var | R²ch | graft 重建 log-mel vs ref(Britpop / 爵士)| 该曲 R²var |
|---|---|---|---|---|---|
| v1 | 6 层 d512,仅 1050 首 | 0.593 | 0.369 | 0.867 / 0.893 | 0.446 / 0.556 |
| v2 | 全量,标准化 MSE | 0.665 | 0.464 | 0.829 / 0.856 | 0.546 / 0.637 |
| **v3** | 全量,通道损失按方差加权 α=0.5(`--var_weight`) | **0.672** | 0.462 | **0.787 / 0.829** | 0.575 / 0.656 |
| **v4** | v3 + 输入 latent 噪声增广 0.15(`--x_noise`,对齐 NAR latent 域差) | 0.671 | 0.461 | **0.740** / 0.847 | 0.623 / 0.648 |

同条件换 seed 的 log-mel 噪声底 0.69–0.70;v3 的嫁接重建离它还差 0.09,但 DiT 对条件误差鲁棒(§13.3),内容尺子已到顶:

闭环(SheetSage2 回转谱 DTW,Ins 声部):
- 嫁接重建:Britpop v2/v3 **0.000**(转谱与原曲完全一致;v1 0.065),Vocal 0.197(原样 0.200);爵士 v1 0.231 → v2 0.198 → v3 0.200(tforig 0.178)
- YuE2 latent → 嫁接:Britpop yue2full 0.064 → graft v1/v2/v3 全 0.064(v4 0.079);爵士 yue2full 0.405 → v1 0.473 / v2 0.443 / **v3 0.371** / v4 0.390;真曲钢琴 cover yue2 0.353 → v1 0.434 / v2 0.414 / v3 0.434 / **v4 0.369**
- **v4 判读**:噪声增广对推理时的 NAR latent 域差确有帮助——真曲 cover(最 OOD 的输入)从 0.43 降到 0.37,Britpop 嫁接重建 log-mel 0.740 是历版最接近噪声底 0.69 的;chroma 一致性 +0.059 / +0.058(天花板 +0.055)。**v4 = 当前默认 adapter**(`runs/v4/best.pt`)。
- r4(阴性对照)于 09-19 01:59 跑完 step 6000,ALL d_spec 最终 +0.0006 / −0.0007 / +0.0016(三次 eval),确认 ≈0;GPU3 已释放。
- 无谱对照 ~1.0–1.08,zero 0.53

chroma 一致性(v2):ref vs 嫁接重建 +0.054(天花板 +0.055);同 latent yue2vae 解码 vs 嫁接 +0.056 / 爵士 +0.445。

**判决(客观侧)**:接穗的读谱能力零损穿过嫁接;内容保持到"同条件换 seed"的天花板。剩下唯一没被尺子回答的是 §7 那一行——
MM3 声学侧的音质是否真的在嫁接输出里兑现。这只能耳判:`Fugue/scion/listen/`(35 条 mp3 + README),核心 A/B 是
`*.yue2full.s0.yue2vae.mp3`(YuE2 自家 VAE)vs `*.yue2full.s0.v2.graft.mp3`(同 latent,MM3 DiT+vocoder)。

### 13.10 下一步(按耳判结果分叉)

- **若嫁接明显更好听** → 直接进入产品线:真曲 → SheetSage2 → YuE2(换 tags 改风格)→ v3/v4 adapter → MM3。
  待补:整曲(>30s)的 YuE2 生成 + chunk 拼接;lyrics 通道(YuE2 歌词→人声);噪声增广已证有效(v4),可再扫 x_noise ∈ {0.1, 0.25}。
- **若差别不大 / 有 artifact** → adapter 的残差是主因:(a) DIFFA 两阶段(语义+声学 Q-Former);(b) 用 DiT 的梯度做感知损失
  (冻结 DiT,在 latent 空间比 velocity 预测,而非只回归 c25);(c) 反向解法 DITTO:微调 ConditionEncoder 入口吃 YuE2 latent。
- 无论哪支:followscore 的 within-condition σ 仍待校准(3 首 × 5 seed),现在只有单 seed 读数。
- 工程:`run_eval_v2.sh <run> <gpu>` 是通用评测入口(解码 → 回转谱 → followscore → chroma);
  `yue2_gen.py --abc_file/--style/--lyrics` 是真曲 cover 入口;所有 wav→mp3 回传走 tar|base64。


### 13.11 耳判结果 + 音质尺子(2026-09-20)

**Henry 耳判**:核心 A/B `mf-1-000415.yue2full.s0` 的 YuE2-VAE 版"单听没什么",v4 嫁接版"一出来直接降维打击,
飞机免费耳机 vs AirPods Pro"。§7 的问题答了:MM3 声学侧的音质**在嫁接输出里兑现了**。同时他听出嫁接版风格偏移
(YuE2 大调布鲁斯 → MM3 小调 traditional hard rock),自己也标注"只是一次抽卡"。

**尺子 1:Audiobox Aesthetics(Meta,PQ/PC/CE/CU 0–10)+ CLAP 探针**(`quality_score.py` → `quality_scores.json`)

| 同一 latent | PQ | CE | CU | CLAP maj−min |
|---|---|---|---|---|
| YuE2 VAE s0 / **v4 graft** | 8.43 / 8.09 | 7.66 / 7.20 | 8.23 / 8.05 | +0.166 / +0.163 |
| YuE2 VAE s1 / v4 graft | 7.81 / 7.02 | 7.07 / 6.48 | 7.75 / 7.05 | +0.218 / +0.179 |
| YuE2 VAE s2 / v4 graft | 7.63 / 6.93 | 6.89 / 6.35 | 7.48 / 6.92 | +0.154 / +0.125 |
| 爵士 YuE2 / v4 graft | 8.27 / 8.09 | 7.96 / 7.87 | 8.29 / 8.21 | |
| s-AVE piano YuE2 / v4 graft | 8.07 / 7.53 | 7.69 / 6.98 | 7.89 / 7.39 | |
| 参照:MM3 原曲 ref / 真曲 s-AVE | 7.68 / 8.31 | 7.07 / 7.72 | 7.57 / 7.65 | |
| 参照:MM3 zero-cond | 6.51 | 4.80 | 5.42 | |

**Audiobox 与耳判方向相反**:它给 YuE2-VAE 版一律高 0.2–0.8。但它也给 MM3 自己的原曲 7.68 < YuE2 生成 8.43,
给真曲→YuE2 VAE 往返 8.34 ≥ 真曲原声 8.31(而实验 B 测得 SI-SDR 只有 7.7 dB)。
读法:**Audiobox PQ 对 VAE 重建的高频/立体声损失不敏感,反而偏好 YuE2 VAE 那种平滑输出**;它不是这台手术要的尺子。
DiT 换 4 个 seed(dit11–14)PQ 8.12–8.19,方差可忽略。v1→v2→v3 单调 7.87→8.18→8.21,与 R² 同向,v4 8.09 略回落。

**尺子 2:频谱/立体声物理量**(与耳判同向,解释"AirPods vs 飞机耳机")

| 同一 latent | 能量占比 >8k | >12k | 99% rolloff | side/mid |
|---|---|---|---|---|
| YuE2 VAE s0 → **v4 graft** | 0.43% → **0.95%** | 0.08% → **0.32%** | 13.2k → **18.3k** | −12.9 → **−8.4 dB** |
| YuE2 VAE s1 → v4 graft | 2.03% → 5.67% | 0.59% → 3.32% | 17.2k → 19.8k | −10.9 → −5.7 dB |
| 爵士 YuE2 → v4 graft | 0.14% → 0.42% | 0.06% → 0.21% | 14.4k → 19.9k | −8.7 → −6.3 dB |
| piano YuE2 → v4 graft | 0.13% → 0.38% | 0.01% → 0.07% | 7.0k → 19.3k | −4.4 → −4.6 dB |
| 参照:MM3 原曲 / 真曲 s-AVE | 0.37% / 0.26% | 0.07% / 0.03% | 16.6k / 12.4k | −5.9 / −6.1 dB |

YuE2 VAE 输出 **12k 以上几乎为空、立体声宽度比正常唱片窄 4–7 dB**;嫁接后高频与宽度回到 MM3 原曲/真曲的量级。
这就是耳朵听到的东西。**后续音质尺子用这两组物理量 + SI-SDR(§13.4),Audiobox 只作参考。**

**风格偏移是不是 latent 没对齐**:SheetSage2 回转谱两版都判 K:Dm、Dm 和弦为主(输入 ABC 即 K:Dm),CLAP major−minor
探针 +0.166 vs +0.163,blues−hardrock 探针 +0.127 vs +0.089(轻微向 hard rock 偏)。**调式没变,偏的是音色/混音质感**,
与 §13.5 的对齐验证(移位扫描峰在 0)一致。v4 的 x_noise 让 adapter 对 latent 细节更"钝",风格质感更多由 DiT 先验补——
如果想压这个偏移,下一档是 x_noise 0.1 或直接用 v3(blues−hr +0.091)。

**评测集**:Henry 提到 100% 有现成评测集(如需下载走 voxlab 中转)。待他指定后跑 `quality_score.py` 同款脚本。


## 14. 配方锁定(2026-09-20)—— Fugue 成了

Henry 耳判定案:`s-ave.piano.s0.v4.graft`(真曲 s-AVE → SheetSage2 谱 → YuE2 换 style → v4 嫁接)**风格完全 follow prompt、
旋律听不出错、音质远超 YuE2**;`s-ave.britpop.s0.v4.graft` 同样过关(Henry 09-20 更正:两首最佳都是 v4 嫁接版,非 YuE2-Vae 版)。cover 能力 > MM3(MM3 读不进谱,r1–r4 全零),音质 > YuE2。

### 14.1 冻结件

```
/cache/zhangjing/fugue/scion/ckpt/scion-v4-20260920/     254MB,MD5SUMS 在目录内
  best.pt        92d49764b9b30f5e08c163674d43363c   adapter v4(8 层 d768 heads12,66.6M;--var_weight 0.5 --x_noise 0.15;20k 步,val R²var 0.6705)
  stats.npz      9412d06e…                          x/y 逐通道 z-score 统计(推理必需)
  args.json / train.log / data.py / train_adapter.py / graft_decode.py / yue2_gen.py / mm3_tf.py / yue2_encode.py
```
上游全冻:YuE2-3B + YuE2-Vae encoder(训练侧)、MM3 condition_encoder(仅 proj 权重用于造 target)、MM3 DiT + vocoder。
训练数据:pairs8k 全库 12837 首自生成配对,零人工标注。

### 14.2 推理管线(单 4090,30s 片段 ≈ 30s 墙钟)

```
真曲 ──SheetSage2──▶ ABC(+lyrics)  ~3s
  ──YuE2-3B AR──▶ semantic tokens   11–15s(torch backend)
  ──YuE2 NAR(32 步 midpoint)──▶ latent64 @25Hz   2–3s
  ──adapter v4──▶ c25 @25Hz   <0.1s
  ──nearest ↑3.445──▶ MM3 DiT(30 步,CFG 1.7,200/100 窗)──▶ latent128 @86Hz   ~10s
  ──MM3 vocoder──▶ 44.1k 立体声   ~1s
```
不用的:YuE2 的 VAE decoder(整段)、MM3 的 LM + depth decoder(只在训练时造 target)。
Henry 的表述"ABC → YuE2 → DiT → VAE"对,精确写法是 **ABC → YuE2(AR+NAR)→ adapter → MM3 DiT → MM3 vocoder**。

### 14.3 Eval 计划(对齐 YuE2 model card 的两套协议)

YuE2 卡片给了现成的比较框架,而且 **MM3 在 WildSongBench 表里、不在 SHS100K cover 表里**(因为它读不进谱)——这恰是我们的位置:

| 协议 | 量什么 | 工具 | 我们的对照 |
|---|---|---|---|
| **SHS100K zero-shot cover** | 身份保持:CLEWS mAP / Hit@1、Discogs-VINet mAP(对 10,545 首检索库);MuLan 风格相似;SongBench Musicality | CLEWS、Discogs-VINet(GitHub raraz15)、MuQ-MuLan | YuE2 full-score 0.647 / 71.3%;SongEcho 0.419 / 48.4%;ACE-Step 0.024。**预期:身份 ≈ YuE2(latent 未变,DTW 已证),Musicality/音质 ↑** |
| **WildSongBench** | 192 prompt 全曲:SongBench Avg(7 维)、Q3O、MuLan、AllMusicCaps、PER | 数据集自带 minimal evaluator | MM3 6.283 / YuE2 6.732 / Suno v5 6.872 |
| 物理音质(本线自定) | >8k / >12k 能量占比、99% rolloff、side/mid、(有参照时)SI-SDR | `quality_score.py` + §13.11 脚本 | Audiobox 不可用(§13.11) |
| 内容闭环(本线自定) | SheetSage2 回转谱 DTW、chroma 一致性 | `followscore.py` `chroma_agree.py` | 已有 |

落地顺序:
1. **先跑本地版**:ood216(217 首真曲,已有 SheetSage2 谱)当 cover 源,检索库 = ood216 + refs_full 真音频;
   身份用 Discogs-VINet(权重走 voxlab 中转),风格用 CLAP(MuLan 待下),音质用物理量。三臂:YuE2-Vae / YuE2-Vae-legacy / **graft v4**,
   + MM3 caption-only(证"MM3 做不了 cover")。每首 2 style × 2 seed,与 YuE2 协议同构。
2. WildSongBench 的 192 prompt + evaluator 走 hf-mirror(今日连接被重置,晚点重试)或 voxlab;
   我们的"全曲生成"= YuE2 plan 自写谱 + graft,可直接进 WSB 表。
3. SHS100K 全量需 10,545 首参考音频,国内离线机拿不到 → 论文里报 ood216 协议 + WSB,SHS100K 子集视 voxlab 能力而定。
待 Henry 指定"100% 有的评测集"名字。

### 14.4 Infra 优化清单(按收益/成本)

| # | 项 | 现状 | 做法 | 预期 |
|---|---|---|---|---|
| 1 | **单进程/双服务** | 两个 conda env(graphtokenizer 装 yue2 / mm3 装 diffusers),文件中转 | 仿 musicodec_service:YuE2 服务 + MM3 服务常驻同一张卡(6+5 GB),HTTP 传 latent;或把 `yue2_infer-0.1.5.whl` 装进 mm3 env 试单进程 | 去掉启动开销(每次 20–40s 载模型)与人工串脚本 |
| 2 | **YuE2 AR 提速** | torch backend 11–15s/30s | 包内自带 `backend="vllm"`、`cuda_graph.py`、fp8 量化 | 3–5× |
| 3 | **整曲** | 目前 `--max_sem 750` / `--frames 750` 人为截 30s | 去掉上限(YuE2 context 24576、MM3 chunk 逻辑本就支持任意长),跑一首 3 分钟真曲验证 | 产品形态 |
| 4 | **人声路径** | 至今测试要么器乐、要么 YuE2 只唱了 5–7 个音 | 选一首词多的真曲,lyrics 走 YuE2,验证人声穿过 graft 的可懂度(PER) | 决定能否上 WSB |
| 5 | `cot="melody"` | 我们用 full(带和弦);YuE2 卡片推荐 cover 用 melody-only | 两种都跑,对比身份/风格 | 可能风格更自由 |
| 6 | DiT 步数 | 30 步 ×CFG ×7 窗 ≈10s | 20 步 A/B;窗间有依赖不能并行 | ≤ 1.5× |
| 7 | 训练侧(不动配方) | — | 若第二轮再训:DiT 空间感知损失、x_noise 扫 {0.1, 0.25} | — |

### 14.5 RepE 干预:第二轮

判断:**先封板 + eval + 论文,RepE 放第二轮。** 理由:
- 论文的主张是"冻结符号规划器 + 冻结声学渲染器 + 66M 自蒸馏 adapter,零标注",eval 数字是它唯一缺的东西;RepE 是可分离的第二贡献。
- 现在唯一想用 RepE 压的是 §13.11 那点音色偏移(blues→hard rock),而它不影响身份/风格 follow,不在关键路径上。
- 但有一个**几乎免费、不碰冻结件的探针**可与 eval 并行:MM3 已有的 steering 轴(selfdistill/v2/axes_library.npz)住在 LM hidden 空间,
  c25 = proj(layer_scale · w₀ · hidden) 是线性映射,所以 **hidden 空间的轴可直接投到 c25 空间**,在 graft_decode 里加 `c_pred + α·proj(axis)`,
  纯推理时干预。若一天内能看到 CLAP 探针按 α 单调移动,就作为论文的一个 section;否则留给第二轮系统做(含在 YuE2 latent 空间找轴)。


## 15. 收尾(2026-09-20 下午)—— infra / eval / 三件套 / arena

### 15.1 Infra:两个常驻服务 + CLI + Arena(单卡)

| 组件 | env | 端口 | 显存 | 文件 |
|---|---|---|---|---|
| 服务 A:SheetSage2 转谱 + YuE2 AR/NAR + YuE2-Vae 解码 | graphtokenizer | 8650(GPU7)/ 8660(GPU6) | 9.8 GB | `fugue_yue2_service.py` |
| 服务 B:adapter v4 + MM3 DiT + vocoder | mm3 | 8651 / 8661 | 4.9 GB | `fugue_graft_service.py` |
| 编排 CLI | mm3 | — | — | `fugue_cover.py --audio x.flac --style "…" [--lyrics] [--seconds 0]` |
| Arena(Gradio 6) | graphtokenizer | 7877 | — | `fugue_arena.py`;`run_arena.sh`;cloudflared 隧道见 `cf_arena.log`(从本地访问不稳,内网 10.158.0.7:7877 稳) |

`run_services.sh`(`GPU=6 PA=8660 PB=8661` 可起第二组)。30s 片段端到端 50s(转谱 5 + AR 14 + NAR 4 + graft 22);
**整曲 4:37 跑通**:AR 54s + NAR 23s + graft 118s ≈ 3.3 min(`covers/s-ave.full.piano.*`,本地 listen/ 有 mp3)。
两个坑已修:① adapter 全长 attention 在 6926 帧 OOM → `predict_c25` 分块(win 768 + pad 128,更贴训练分布);
② 服务 A 转谱后 `torch.cuda.empty_cache()`,否则 SheetSage2 峰值显存留在 cache 里挤掉服务 B。

### 15.2 Eval(ood216 协议,对齐 YuE2 卡片的 SHS100K 表)

`eval_gen.py`:210 首真曲(有 SheetSage2 谱)× 2 目标 style(6 选 2,器乐)× 前 60s,三臂 fullscore(YuE2-Vae 解码 / v4 嫁接,同 latent)
+ noscore 对照(cot=off,只 style A);双卡 40s/条,630 条 ≈ 3.5h。`eval_score.py`:Discogs-VINet 身份检索(权重经 ghproxy 拉到
`/cache/zhangjing/models/discogs-vinet`,`vinet_embed.py` 不依赖 essentia、键名 features.*→front_end.* 映射,strict 加载)、
CLAP 风格、物理音质量、Audiobox(参考)、SheetSage2 回转谱 DTW。`run_eval_score.sh` 排队自动跑。
**上界**:原曲自身 60s 截段检索 Hit@1 0.819 / MRR 0.861(n=210)。**早期读数(n=32)**:fullscore graft Hit@1 0.31 / MRR 0.39,
yue2vae 0.28 / 0.39;noscore 两臂 0.00,中位 rank ~110/210 = 随机。→ 身份 ≈ YuE2,与预期一致。最终表填 §15.4。

### 15.3 三件套

- **ckpt**:`ckpt/scion-v4-20260920/adapter.safetensors`(fp32,266MB,md5 3c2c1f8f…)+ `stats.npz` + `config.json`,经 MCP base64 分片
  (3MB×85 片,4 片/次)搬到本地 `Fugue/release/hf/fugue-scion-v4/`,md5 全对。cloudflared 隧道大文件不稳,弃用。
  HF 上传待 Henry 本地 `hf auth login` 后:`hf upload HenryZ838978/fugue-scion-v4 release/hf/fugue-scion-v4 .`(model card 已写)。
- **repo**:`Fugue/release/fugue/` → **github.com/HenryZ838978/fugue(private,明天翻 public)**。结构:`fugue/`(最小推理包:
  adapter.py / graft.py / scion.py / cover.py,路径走 env 或 HF id;在服务器上验证与研究脚本逐元素一致,wav max|Δ| 2e-5)、
  `services/`(路径已参数化)、`research/`(scion 全部脚本 + SCION.md)、`samples/`(4 对 A/B mp3)、`paper/fugue.md`。
- **paper**:`paper/fugue.md` 草稿 v0.1,§5.5 表待 eval 填。

### 15.4 Eval 结果(ood216,209 首 × 2 style × 60s,单 seed;068-slash 谱 48k token 超上下文剔除)

| 臂 | n | Hit@1 | Hit@10 | MRR | >8k | >12k | rolloff99 | side/mid | CLAP-style | DTW |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 供谱 · YuE2-Vae | 418 | 0.557 | 0.809 | 0.646 | 0.5 % | 0.1 % | 13.8 kHz | -10.9 dB | 0.319 | 0.470 |
| 供谱 · **Fugue v4** | 418 | 0.524 | 0.775 | 0.609 | 1.0 % | 0.2 % | 17.6 kHz | -4.8 dB | 0.304 | 0.523 |
| 无谱 · YuE2-Vae | 210 | 0.000 | 0.033 | 0.022 | | | | | 0.436 | 1.686 |
| 无谱 · Fugue | 210 | 0.000 | 0.033 | 0.022 | | | | | 0.418 | 1.622 |
| 上界:原曲自身 60s 截段 | 210 | 0.819 | 0.933 | 0.861 | | | | | | |

配对(同 latent,n=418):rank 完全相同 239 对,Fugue 更好 63,YuE2-Vae 更好 116;Hit@1 差 -0.033(bootstrap 95% CI [-0.060, -0.010]),
MRR 保持 YuE2 的 94 %;YuE2 命中 rank 1 的样本里 Fugue 仍命中 91.0 %。物理音质四项 Fugue 胜率 99.3 %(>12k 能量 3.1×、
rolloff +3.8 kHz、side/mid +6.1 dB)。CLAP-style 配对差 -0.019、回转谱 DTW 差 +0.000 —— 持平。Audiobox PQ 反向 -0.28(同 §13.11)。

**口径(Henry 09-20)**:身份差 ~3% 是 R² 0.67 有损映射的信息论代价,对 gen-style 模型不构成"更差";不做 DiT 感知损失追这 3%。
DOI 之后的自然下一步是 RepE + DiT 侧控制。


## Sources

- [Foley Control: Aligning a Frozen Latent Text-to-Audio Model to Video](https://arxiv.org/pdf/2510.21581)
- [Freeze-Omni: A Smart and Low Latency Speech-to-speech Dialogue Model with Frozen LLM](https://arxiv.org/html/2411.00774v1)
- [DITTO-TTS: Diffusion Transformers for Scalable Text-to-Speech](https://proceedings.iclr.cc/paper_files/paper/2025/file/80e77d9ed2f74dcaf1a42cb1a2593559-Paper-Conference.pdf)
- [DIFFA: Large Language Diffusion Models Can Listen and Understand](https://arxiv.org/pdf/2507.18452)
- [LLM-Codec](https://arxiv.org/pdf/2604.17852)
- [Where Does the Sound Go? Tracing Acoustic Information Loss in Audio-Conditioned LLMs](https://arxiv.org/html/2609.05871)
- [ALAS: An Automatic Latent Alignment Score for Audio](https://arxiv.org/pdf/2505.19937)
