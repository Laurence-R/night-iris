# fast-clahe

夜間 HDR 前處理（16-bit CLAHE）與 YOLO26 偵測消融實驗的 monorepo。

| 子專案 | 職責 |
|--------|------|
| [`raw-clahe`](raw-clahe/) | GPU 加速 HDR CLAHE → LDR，延遲評測與增強影像輸出 |
| [`yolo-train`](yolo-train/) | Baseline / enhance+fine-tune 訓練、驗證、TensorRT 匯出 |

資料流：

```
HDR (uint16) → raw-clahe → enhanced LDR → yolo-train（訓練／驗證）
```

## 環境

- Python **3.11**
- PyTorch **cu126**
- 在 repo 根目錄安裝兩個 workspace member：

```bash
uv sync --all-packages
```

## 版控範圍：什麼在 git、什麼要本機準備

本 repo **只追蹤程式、設定檔與文件**。大型資料、權重與執行產物刻意不進 git（見根目錄 [`.gitignore`](.gitignore)）。clone 之後需自行準備下列內容，實驗才能完整重跑。

### 會進 GitHub 的內容

| 類型 | 路徑 |
|------|------|
| Workspace 設定 | `pyproject.toml`、`uv.lock`、`.python-version` |
| 文件 | `README.md`、`docs/report.md`、`docs/ppt.md`、各子專案 README |
| 前處理程式與設定 | `raw-clahe/*.py`、`raw-clahe/configs/` |
| 偵測程式與設定 | `yolo-train/*.py`、`yolo-train/configs/` |

### 需本機預先準備（不進版控）

| 內容 | 建議放置路徑 | 說明 |
|------|----------------|------|
| 夜間 HDR／16-bit 影像 | `raw-clahe/dataset/` | CLAHE 輸入；張數可對齊報告（約 2230） |
| YOLO 訓練資料集 | `yolo-train/lod-dataset.v2i.yolo26/` | enhance 路線訓練用（可由 Roboflow 下載） |
| YOLO 驗證資料集 | `yolo-train/rgb-dark.v1i.yolo26/` | baseline／enhance 驗證用 |
| 預訓練權重 | `yolo-train/pretrained_model/yolo26{n,s,m,l,x}.pt` | Baseline 與訓練起點 |
| Fine-tuned 權重 | `yolo-train/fine_tuned_model/yolo26*_trained.pt` | enhance 訓練後權重（可自行訓練產生） |
| 推論測資（可選） | `yolo-train/dark_rgb/` | `predict` 模式用的圖片資料夾 |
| Roboflow API Key | 環境變數 `ROBOFLOW_API_KEY` | 僅本機／CI secret，勿寫進程式 |

下載訓練資料集範例：

```bash
set ROBOFLOW_API_KEY=你的金鑰
uv run --package yolo-train --directory yolo-train python download_dataset.py
```

權重命名慣例見 [`yolo-train/README.md`](yolo-train/README.md)。

### 專案自動產出（不進版控，可刪除後重跑）

| 產出 | 路徑 | 如何產生 |
|------|------|----------|
| 增強 LDR、設定快照 | `raw-clahe/result/` | 跑 `raw-clahe` 前處理 |
| 延遲圖表 | `raw-clahe/preprocessing_latency_*.png` | 同上 |
| 驗證／訓練曲線與預測圖 | `yolo-train/runs/` | Ultralytics `train`／`val`／`predict` |
| 實驗指標 JSON | `yolo-train/experiments/*/metrics.json` | `val` 模式結束後寫入 |
| ONNX／TensorRT engine | `*.onnx`、`*.engine`（多在權重同目錄） | `export` 模式；engine 綁定本機 GPU／TensorRT，換機器需重匯 |
| 虛擬環境 | `.venv/` | `uv sync --all-packages` |

這些檔案體積大或與機器綁定，適合放本機或外部儲存，不適合 commit。

## 常用指令

前處理（在 `raw-clahe` 工作目錄）：

```bash
uv run --package raw-clahe --directory raw-clahe python main.py --config configs/default.toml
uv run --package raw-clahe --directory raw-clahe python main.py --config configs/default.toml --no-bilateral --count 100
```

偵測實驗（在 `yolo-train` 工作目錄）：

```bash
# Baseline：未增強、未微調
uv run --package yolo-train --directory yolo-train python main.py --config configs/baseline_val.toml --model-size n

# Enhance 路線：訓練 / 驗證
uv run --package yolo-train --directory yolo-train python main.py --config configs/enhance_train.toml --model-size s
uv run --package yolo-train --directory yolo-train python main.py --config configs/enhance_val.toml --model-size all
```

驗證指標會寫入 `yolo-train/experiments/<name>_<size>/metrics.json`。

## 文件

- [實驗報告](docs/report.md)
- [簡報大綱](docs/ppt.md)
- [raw-clahe 操作說明](raw-clahe/README.md)
- [yolo-train 操作說明](yolo-train/README.md)
