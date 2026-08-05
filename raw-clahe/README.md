# raw-clahe

GPU-based 的夜間 HDR CLAHE 前處理：白平衡 → Y 通道 CLAHE →（可選）雙邊濾波 → 線性 tone mapping → uint8 LDR。

## 準備資料
請在 `raw-clahe` 底下建立 `dataset/` 作為欲處理之 16-bit HDR 影像的放置區。
本研究使用的影像資料集為 [LOD Dataset](https://github.com/ying-fu/LODDataset) 若有需要可以點擊連結前往該 repo

建好 `dataset/` 後的檔案結構：

```
fast-clahe/
├── dataset/
│   ├── img_1.png
│   ├── img_2.png
│   ...
...
```
## 設定檔

| 檔案 | 說明 |
|------|------|
| [`configs/default.toml`](configs/default.toml) | 預設實驗（50 張、bilateral 開啟） |
| [`configs/full_dataset.toml`](configs/full_dataset.toml) | 全量延遲評測（`count = 2230`，對齊報告規模） |

## 執行

在 repo 根目錄：

```bash
uv run --package raw-clahe --directory raw-clahe python main.py --config configs/default.toml
```

常用指令：

```bash
# 關閉雙邊濾波、限制張數
uv run --package raw-clahe --directory raw-clahe python main.py --config configs/default.toml --no-bilateral --count 100

# 自訂輸入／輸出目錄
uv run --package raw-clahe --directory raw-clahe python main.py --input dataset --output result
```

CLI 參數：`--config`、`--input`、`--output`、`--count`、`--no-bilateral`。

跑完後會有以下的檔案產出：

- 增強影像：`result/enhanced/`
- 生效設定快照：`result/run_config.toml`
- 延遲圖：工作目錄下的 `preprocessing_latency_*.png`

研究報告見 [docs/report.md](../docs/report.md)。
