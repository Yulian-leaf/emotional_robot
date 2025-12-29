# 三模态进一步训练（Text / Audio / Image）

本项目包含三种模态的数据：
- 文本：`txt_Emotional_Dataset/{train,val,test}.txt`
- 语音：`audio_RAVDESS/Actor_*/**.wav`（RAVDESS 命名规范）
- 图像：`img_FER/{train,test}/{class}/...`

下面提供三套脚本（位于 `scripts/`）用于进一步微调，并将模型导出到本地目录，供后端通过环境变量加载。

## 0) 环境准备（Windows/conda）

建议在你当前的 `ne` 环境中安装训练依赖：

```powershell
pip install -r requirements-train.txt
```

> `torch` / `torchaudio` / `torchvision` 建议按你的 CPU/GPU 版本单独安装（conda 或 pip）。
>
> 已知限制：你当前环境里 `soundfile` 依赖的 `libsndfile.dll` 缺失，会导致 `datasets`/`evaluate` 以及 `transformers.Trainer` 导入失败。
> 本仓库提供的训练脚本已避免依赖这些库，使用纯 PyTorch 训练循环。

## 1) 文本情感微调

数据：`txt_Emotional_Dataset/train.txt` 等。你的 `train.txt` 目前是 `text;label`（分号分隔）。脚本也支持 `label\ttext` 等常见格式。

运行：

```powershell
python scripts\train_text_hf.py --data_dir txt_Emotional_Dataset --output_dir models\text-emotion
```

接入后端：

```powershell
$env:TEXT_MODEL_ID = "d:\python-learn\program\models\text-emotion"
# 若需要 ASR：
$env:TEXT_ASR_MODEL = "openai/whisper-base"  # 或你正在用的 ASR
```

## 2) 语音情感微调（RAVDESS）

第一步：生成清单（CSV）

```powershell
python scripts\prepare_ravdess.py --ravdess_dir audio_RAVDESS --out_csv data\ravdess.csv
```

第二步：训练

```powershell
python scripts\train_audio_hf.py --csv data\ravdess.csv --output_dir models\audio-ser
```

接入后端：

```powershell
$env:AUDIO_MODEL_ID = "d:\python-learn\program\models\audio-ser"
# 建议禁用文本回退，避免 audio==text：
$env:AUDIO_ALLOW_TEXT_FALLBACK = "0"
```

## 3) 图像表情微调（FER folder）

```powershell
python scripts\train_image_hf.py --data_dir img_FER --output_dir models\image-fer
```

接入后端：

```powershell
$env:IMAGE_MODEL_ID = "d:\python-learn\program\models\image-fer"
```

## 4) 快速烟测（不完整训练）

所有训练脚本都支持 `--smoke_test 1`：只加载少量样本并跑一次 forward，确认环境与数据没问题。

---

如果你希望把三模态标签统一成 7 类（angry/disgust/fear/happy/neutral/sad/surprise），我们也可以在脚本里开启 `--canonical 1` 进行映射。
