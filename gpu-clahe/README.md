# gpu-clahe

`clahe.py` 是 Night-Iris 前處理使用的 GPU CLAHE kernel。`night-iris/enhance.py` 直接載入這個 Kernel，並對 8-bit LDR 的 Y 通道做對比拉伸。

`main.py` 是另一條測試：它會先讀取未處理影像，將圖片轉成 YUV 色彩空間之後，對 Y 通道做 CLAHE、做雙邊濾波 (Optional)、線性 Tone Mapping，最後繪製出量測延遲。

## 準備資料

請先在 `gpu-clahe/` 目錄底下創建一個 `unprocessed-img/` 目錄，並在 `gpu-clahe/unprocessed-img/` 放入要處理的影像。影像必須已經是設定檔的 `width`×`height` (default.toml 預設 2448×2048)。

```
Night-Iris/
├── gpu-clahe/
│   ├── unprocessed-img/
│   │   ├── img_1.png
│   │   ├── img_2.png
```

有幾點要注意：

- LDR 必須是 8-bit 三通道
- HDR 必須是 16-bit 三通道
- Image Size 一定要跟的設定檔中的 `width`×`height` 相同

## 設定檔


| 檔案                                             | 說明                                   |
| ---------------------------------------------- | ------------------------------------ |
| `[configs/default.toml](configs/default.toml)` | 跑 50 張 LDR 影像，影像尺寸為 2448x2048，雙邊濾波關閉 |


設定欄位是 `[paths]`、`[image]`、`[clahe]`、`[bilateral]`

## 執行

在 repo 根目錄。`--dynamic-range` 覆寫設定檔：`ldr` 是 8-bit，`hdr` 是 16-bit。

```bash
uv run --package gpu-clahe --directory gpu-clahe python main.py --config configs/default.toml --dynamic-range hdr
```

```bash
uv run --package gpu-clahe --directory gpu-clahe python main.py --config configs/default.toml --dynamic-range ldr
uv run --package gpu-clahe --directory gpu-clahe python main.py --config configs/default.toml --dynamic-range hdr --no-bilateral --count 100
uv run --package gpu-clahe --directory gpu-clahe python main.py --input unprocessed-img --output result --dynamic-range ldr
```

CLI 參數：`--config`、`--input`、`--output`、`--count`、`--no-bilateral`、`--dynamic-range {ldr,hdr}`。

跑完產生以下的檔案：

- `result/processed-img/`：處理後的影像
- `result/run_config.toml`：這次處理過程的設定快照
- `result/latency/preprocessing_latency_chart.png`：每張圖的前處理總時常點圖
- `result/latency/preprocessing_latency_breakdown.png`：每張圖在各階段延遲堆疊圖

