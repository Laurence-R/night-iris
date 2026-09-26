# gpu-clahe

`clahe.py` 是 Night-Iris 前處理使用的 GPU CLAHE kernel。`night-iris/enhance.py` 直接載入這個檔，對 8-bit LDR 的 Y 通道做對比拉伸。

`main.py` 是另一條測試：讀 16-bit 影像，做白平衡、CLAHE、可選的雙邊濾波、線性 tone mapping，並量測延遲。它不產出 YOLO 訓練圖。讀圖用 OpenCV 的 `IMREAD_UNCHANGED`，不讀 EXIF，也不把輸出對到某一個偵測權重。

測試路徑：白平衡 → Y 通道 CLAHE →（可選）雙邊濾波 → 線性 tone mapping → uint8。

## 準備資料

在 `gpu-clahe/dataset/` 放入 16-bit 影像。影像必須已經是設定檔的 `height`×`width`（預設 800×1200）；這條測試不會自行縮放。

```
Night-Iris/
├── gpu-clahe/
│   ├── dataset/
│   │   ├── img_1.png
│   │   ├── img_2.png
```

## 設定檔

| 檔案 | 說明 |
|------|------|
| [`configs/default.toml`](configs/default.toml) | 50 張，雙邊濾波關閉 |
| [`configs/full_dataset.toml`](configs/full_dataset.toml) | `count = 2230`，雙邊濾波開啟，用來跑完整批的延遲 |

兩份設定的欄位相同：`paths`、`image`、`clahe`、`bilateral`、`white_balance`。CLAHE 的 `clip_limit` 用 kernel 預設 4.0，與 `night-iris` 設定裡的 4.0 相同，但不是從同一份 toml 讀的。

## 執行

在 repo 根目錄：

```bash
uv run --package gpu-clahe --directory gpu-clahe python main.py --config configs/default.toml
```

```bash
uv run --package gpu-clahe --directory gpu-clahe python main.py --config configs/default.toml --no-bilateral --count 100
uv run --package gpu-clahe --directory gpu-clahe python main.py --input dataset --output result
```

CLI 參數：`--config`、`--input`、`--output`、`--count`、`--no-bilateral`。

跑完會有：

- `result/enhanced/`：測試輸出影像
- `result/run_config.toml`：這次生效的設定
- 工作目錄下的 `preprocessing_latency_*.png`：延遲圖
