# fast-clahe

夜間 HDR 前處理（16-bit CLAHE）與 YOLO26 偵測消融實驗的 monorepo。


| 子專案                         | 職責                                             |
| --------------------------- | ---------------------------------------------- |
| `[raw-clahe](raw-clahe/)`   | GPU 加速 HDR CLAHE → LDR，延遲評測與增強影像輸出             |
| `[yolo-train](yolo-train/)` | Baseline / enhance+fine-tune 訓練、驗證、TensorRT 匯出 |


資料流：

```
HDR → raw-clahe → enhanced LDR → yolo-train（訓練／驗證）
```

## 環境

- Python **3.11**
- PyTorch **cu126**
- 在 repo 根目錄安裝兩個 workspace member：

```bash
uv sync --all-packages
```

## 本機預先準備之檔案

本 repo **只追蹤程式、設定檔與文件**。大型資料、權重與執行產物不進 git（見根目錄 `[.gitignore](.gitignore)`）。clone 之後需自行準備下列內容，實驗才能完整重跑。


| 內容               | 建議放置路徑                                             | 說明                                                             |
| ---------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| 夜間 HDR／16-bit 影像 | `raw-clahe/dataset/`                               | CLAHE 輸入；張數可對齊報告（約 2230）                                       |
| Enhance 路線用之資料集  | `yolo-train/w_enhance/`                            | 已經透過 Roboflw 做好標註，可馬上供 YOLO 做訓練、驗證與測試，模擬 YOLO 在增強後的影像上的表現。     |
| Baseline 路線用之資料集 | `yolo-train/wo_enhance/`                           | 已經透過 Roboflw 做好標註，可馬上供 YOLO 做訓練、驗證與測試，模擬 YOLO 在未增強影像上的表現。      |
| 預訓練權重            | `yolo-train/pretrained_model/yolo26{n,s,m,l,x}.pt` | YOLO26 的預訓練權重，由日間資料集 COCO Dataset 所訓練的 YOLO26 模型。              |
| Fine-tuned 權重    | `yolo-train/fine_tuned_model/yolo26*_trained.pt`   | 利用增強後的資料重新微調的 YOLO26 模型，可透過 `yolo-train` 將 enhance 之後的資料集對模型微調 |
| 推論測資（可選）         | `yolo-train/dark_rgb/`                             | `predict` 模式用的圖片資料夾                                            |
| Roboflow API Key | 環境變數 `ROBOFLOW_API_KEY`                            | 僅本機／CI secret，勿寫進程式                                            |


下載訓練資料集範例：

```bash
set ROBOFLOW_API_KEY=你的金鑰
uv run --package yolo-train --directory yolo-train python download_dataset.py
```

權重命名慣例見 `[yolo-train/README.md](yolo-train/README.md)`。

### 專案自動產出


| 產出                   | 路徑                                      | 如何產生                                        |
| -------------------- | --------------------------------------- | ------------------------------------------- |
| 增強後 LDR、執行時的 config  | `raw-clahe/result/`                     | 跑 `raw-clahe` 前處理                           |
| 處理時長圖表               | `raw-clahe/preprocessing_latency_*.png` | 同上                                          |
| 驗證／訓練曲線與預測圖          | `yolo-train/runs/`                      | Ultralytics `train`／`val`／`predict`         |
| 實驗指標 JSON            | `yolo-train/experiments/<config.name>/metrics.json` | `val` 模式結束後寫入                               |
| ONNX／TensorRT engine | `*.onnx`、`*.engine`（多在權重同目錄）            | `export` 模式；engine 綁定本機 GPU／TensorRT，換機器需重匯 |
| 虛擬環境                 | `.venv/`                                | `uv sync --all-packages`                    |


這些檔案體積大或與機器綁定，適合放本機或外部儲存，不適合 commit。

## 常用指令

前處理（在 `raw-clahe` 工作目錄）：

```bash
uv run --package raw-clahe --directory raw-clahe python main.py --config configs/default.toml
uv run --package raw-clahe --directory raw-clahe python main.py --config configs/default.toml --no-bilateral --count 100
```

偵測消融（在 `yolo-train` 工作目錄；四選一 val）：

```bash
uv run --package yolo-train --directory yolo-train python main.py --config configs/wo_enhance_nofinetune.toml --model-size n
uv run --package yolo-train --directory yolo-train python main.py --config configs/w_enhance_nofinetune.toml --model-size n
uv run --package yolo-train --directory yolo-train python main.py --config configs/wo_enhance_finetune.toml --model-size n
uv run --package yolo-train --directory yolo-train python main.py --config configs/w_enhance_finetune.toml --model-size n

# 在 w_enhance 上微調
uv run --package yolo-train --directory yolo-train python main.py --config configs/w_enhance_train.toml --model-size s
```

驗證指標會寫入 `yolo-train/experiments/<config.name>/metrics.json`。

### Ultralytics runs 目錄

若結果寫到別的路徑（例如舊的 `E:\yolo-train\runs`），請先設定：

```bash
uv run --package yolo-train yolo settings runs_dir="E:\fast-clahe\yolo-train\runs"
```

細節見 [`yolo-train/README.md`](yolo-train/README.md)。

## 文件

- [raw-clahe 操作說明](raw-clahe/README.md)
- [yolo-train 操作說明](yolo-train/README.md)

