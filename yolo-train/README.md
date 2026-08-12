# yolo-train

YOLO26 夜間偵測消融：資料集（`wo_enhance` / `w_enhance`）× 權重（nofinetune / finetune），並可匯出 TensorRT engine。

## 本機資料與權重（不在 git 內）

| 內容 | 路徑 |
|------|------|
| 未增強資料集 | `wo_enhance/` |
| 增強後資料集 | `w_enhance/` |
| 預訓練權重 | `pretrained_model/yolo26{n\|s\|m\|l\|x}.pt` |
| Fine-tuned 權重 | `fine_tuned_model/yolo26{n\|s\|m\|l\|x}_trained.pt` |
| 推論測資（可選） | `dark_rgb/` |

`runs/`、`experiments/`、`*.pt`／`*.engine`／`*.onnx` 亦不進版控。細節見[根 README](../README.md)。

## Ultralytics runs 目錄

全域 `runs_dir` 若指到別的路徑（例如舊的 `E:\old_folder\runs`），結果會寫出本專案。建議先設成本 repo：

```bash
uv run --package yolo-train yolo settings

# 例如：如果專案再 E:\
uv run --package yolo-train yolo settings runs_dir="E:\fast-clahe\yolo-train\runs"
```

本專案的 `main.py` 也會在 `train`／`val`／`predict` 傳入 `project=<yolo-train>/runs`（絕對路徑），降低跑錯目錄的機會。

## 消融 2×2（四選一 val）

| Config | 資料 | 權重 |
|--------|------|------|
| [`configs/wo_enhance_nofinetune.toml`](configs/wo_enhance_nofinetune.toml) | `wo_enhance` | pretrained |
| [`configs/w_enhance_nofinetune.toml`](configs/w_enhance_nofinetune.toml) | `w_enhance` | pretrained |
| [`configs/wo_enhance_finetune.toml`](configs/wo_enhance_finetune.toml) | `wo_enhance` | fine-tuned |
| [`configs/w_enhance_finetune.toml`](configs/w_enhance_finetune.toml) | `w_enhance` | fine-tuned |

其他：

| Config | 模式 | 說明 |
|--------|------|------|
| [`configs/w_enhance_train.toml`](configs/w_enhance_train.toml) | `train` | 在 `w_enhance` 上微調 |
| [`configs/export_engine.toml`](configs/export_engine.toml) | `export` | 匯出 FP16 `.engine` |
| [`configs/predict.toml`](configs/predict.toml) | `predict` | 對 `dark_rgb` 推論 |

## 執行

```bash
# 四選一驗證（預設 config 為 wo_enhance_nofinetune）
uv run --package yolo-train --directory yolo-train python main.py --config configs/wo_enhance_nofinetune.toml --model-size n
uv run --package yolo-train --directory yolo-train python main.py --config configs/w_enhance_finetune.toml --model-size n

# 訓練
uv run --package yolo-train --directory yolo-train python main.py --config configs/w_enhance_train.toml --model-size s
```

- `--model-size`：`n` / `s` / `m` / `l` / `x` / `all`
- `--model`：直接覆寫權重路徑

## 結果記錄

`val` 寫入（目錄名 = config 的 `name`，不附 size 後綴）：

```
experiments/<name>/
  metrics.json
  config.snapshot.toml   # 含本次實際 model 路徑
```

同 config 再跑不同 size 會覆寫同目錄；以 snapshot 分辨用了哪個權重。

## 匯出 TensorRT `.engine`（FP16）

```bash
uv run --package yolo-train --directory yolo-train python main.py --config configs/export_engine.toml --model fine_tuned_model/yolo26n_trained.pt
```

`.engine` 綁定本機 GPU／TensorRT；改 `quantize`／`batch` 後需重匯。

## 下載 Roboflow 資料集

```bash
set ROBOFLOW_API_KEY=你的金鑰
uv run --package yolo-train --directory yolo-train python download_dataset.py
```

腳本會把下載目錄重新命名為 `w_enhance/`。

研究報告見 [docs/report.md](../docs/report.md)。
