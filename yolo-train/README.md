# yolo-train

YOLO26 夜間偵測消融：Baseline（無增強、無微調）vs Enhance + fine-tune，並可匯出 TensorRT engine。

## 本機資料與權重（不在 git 內）

資料集、`pretrained_model/`、`fine_tuned_model/`、`runs/`、`experiments/`、`*.pt`／`*.engine`／`*.onnx` 皆不進版控。clone 後請自行準備，細節見[根 README「版控範圍」](../README.md#版控範圍什麼在-git什麼要本機準備)。

## 權重路徑慣例

| 路線 | 路徑 |
|------|------|
| Baseline（pretrained） | `pretrained_model/yolo26{n\|s\|m\|l\|x}.pt` |
| Fine-tuned（enhance 訓練後） | `fine_tuned_model/yolo26{n\|s\|m\|l\|x}_trained.pt` |

對應舊 runs 註記：`val 1–5` ≈ baseline 驗證；`train 2–6` ≈ enhance 訓練產物。請改用下方具名 config，勿再靠註解開關。

## 實驗設定檔

| Config | 模式 | 說明 |
|--------|------|------|
| [`configs/baseline_val.toml`](configs/baseline_val.toml) | `val` | w/o enhance & fine-tune，資料 `rgb-dark.v1i.yolo26` |
| [`configs/enhance_train.toml`](configs/enhance_train.toml) | `train` | enhance 路線訓練，資料 `lod-dataset.v2i.yolo26` |
| [`configs/enhance_val.toml`](configs/enhance_val.toml) | `val` | enhance + fine-tune 驗證 |
| [`configs/export_engine.toml`](configs/export_engine.toml) | `export` | 匯出 TensorRT `.engine` |
| [`configs/predict.toml`](configs/predict.toml) | `predict` | 對 `dark_rgb` 做推論 |

## 執行

```bash
uv run --package yolo-train --directory yolo-train python main.py --config configs/baseline_val.toml --model-size n
uv run --package yolo-train --directory yolo-train python main.py --config configs/enhance_val.toml --model-size all
uv run --package yolo-train --directory yolo-train python main.py --config configs/enhance_train.toml --model-size s
```

- `--model-size`：`n` / `s` / `m` / `l` / `x` / `all`
- `--model`：直接覆寫權重路徑（略過 size 展開）
- `--config`：預設 `configs/baseline_val.toml`

## 結果記錄

`val` 模式會寫入：

```
experiments/<name>_<size>/
  metrics.json          # mAP、延遲、FPS、路徑、時間戳
  config.snapshot.toml  # 本次生效設定
```

## 匯出 TensorRT `.engine`（FP16）

依賴已含 `tensorrt-cu12`、`nvidia-modelopt` 與 ONNX 工具鏈。預設匯出 **FP16**（`quantize = 16`）、`batch = 1`（見 `configs/export_engine.toml` 的 `[export]`）。驗證設定同樣使用 `batch = 1`，方便與 `.pt` 公平比較單張延遲。

```bash
uv run --package yolo-train --directory yolo-train python main.py --config configs/export_engine.toml --model fine_tuned_model/yolo26n_trained.pt
```

成功後檔案在權重同目錄，例如 `fine_tuned_model/yolo26n_trained.engine`。`.engine` 綁定本機 GPU／TensorRT 版本；改 `quantize`／`batch` 後需重匯。

公平比較範例（兩邊都是 batch=1）：

```bash
uv run --package yolo-train --directory yolo-train python main.py --config configs/enhance_val.toml --model fine_tuned_model/yolo26x_trained.pt
uv run --package yolo-train --directory yolo-train python main.py --config configs/enhance_val.toml --model fine_tuned_model/yolo26x_trained.engine
```

## 下載 Roboflow 資料集

```bash
set ROBOFLOW_API_KEY=你的金鑰
uv run --package yolo-train --directory yolo-train python download_dataset.py
```

研究報告見 [docs/report.md](../docs/report.md)。
