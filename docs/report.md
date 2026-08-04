# 夜間 HDR 前處理模組實驗報告

**專案名稱：** raw-clahe  
**核心方法：** 16-bit Domain CLAHE + Linear Tone Mapping  
**後端偵測模型：** YOLO26（n / s / m / l / xl）  
**硬體環境：** CUDA GPU（約 10 GB VRAM）、32 GB 系統記憶體  

---

## 1. 摘要

本實驗針對夜間場景感知問題，提出一套以 **GPU 加速傳統影像增強** 為核心的前處理模組。輸入為未白平衡的 **16-bit、三通道 HDR** 影像，在高動態範圍域中執行 **CLAHE（Contrast Limited Adaptive Histogram Equalization）**，再經線性色調映射轉成 LDR，供後端 YOLO26 進行物件偵測。

實驗結果顯示：

1. **偵測精度大幅提升**：夜間原始影像上 baseline YOLO26 幾乎失效（mAP50 ≈ 0.001），經前處理增強並搭配 fine-tune 後，mAP50 可達 **0.672～0.757**。
2. **前處理可達 real-time**：在 2230 張 800×1200 影像上，P99 前處理延遲約 **2.39 ms**，超過 30 ms 的尖峰次數為 **0**，無需 multi-threading。
3. **端到端推論時間未因前處理變慢**：enhance + fine-tune 設定下，YOLO26 推論時間反而低於 baseline（例如 xl：10.9 ms → 6.4 ms）。

---

## 2. 研究背景與目標

### 2.1 問題陳述

夜間環境中，相機感測器常面臨：

- 動態範圍不足、暗部細節淹沒於雜訊
- 未白平衡的 RAW / HDR 資料色偏嚴重
- 主流偵測模型多在 LDR（8-bit）日間資料上訓練，對夜間 HDR 泛化能力差

若僅依賴後端模型重新訓練，成本高、部署週期長，且仍可能受輸入動態範圍壓縮失真影響。

### 2.2 初始研究目標

> 期望以**單一前處理模組**，讓後端模型在**未經訓練**的情況下，即可克服夜間感知問題。

### 2.3 研究方向調整

實務驗證後，研究方向收斂為：

| 面向 | 內容 |
|------|------|
| 輸入 | 未白平衡的 16-bit、3-channel HDR |
| 增強域 | 在 16-bit domain 執行 CLAHE（保留高動態資訊） |
| 輸出 | Linear tone mapping → LDR（uint8），供偵測模型使用 |
| 後端 | YOLO26 系列；分為 non-fine-tuned / fine-tuned |
| 工程目標 | GPU 記憶體預配置與非同步管線，達成穩定 real-time |

目前報告中的偵測結果以 **「前處理增強 + fine-tune」** 為主；初始「免訓練」目標仍作為動機與後續工作方向保留。

---

## 3. 方法

### 3.1 整體 Pipeline

```
RAW / HDR (uint16, RGB, 未白平衡)
        │
        ▼
  [CPU] 批次載入至記憶體陣列
        │
        ▼
  [H2D] Pinned Memory → GPU float32 [0,1]
        │
        ▼
  Grey-World 白平衡（可選步驟；實作於 loader）
        │
        ▼
  RGB → YUV
        │
        ▼
  CLAHE on Y（grid 8×8, clip_limit=4.0）
        │
        ▼
  YUV → RGB
        │
        ▼
  （可選）Bilateral Filter
        │
        ▼
  Linear Tone Mapping → uint8 LDR
        │
        ▼
  [D2H] → CPU → PNG / 後端 YOLO26
```

### 3.2 為何在 16-bit Domain 做 CLAHE

| 做法 | 優點 | 缺點 |
|------|------|------|
| 先壓成 8-bit 再 CLAHE | 實作簡單、相容 OpenCV | 暗部量化誤差大，夜間細節已損失 |
| **先 CLAHE 再 Tone Mapping（本方法）** | 在高動態域拉伸局部對比，暗部資訊較完整 | 需自行實作 GPU CLAHE，記憶體與穩定性要求高 |

夜間 HDR 的有效資訊多集中在低碼值區間；若先量化到 8-bit，CLAHE 只能在已量化的灰階上操作。本方法先在 float32（正規化自 uint16）上做局部直方圖均衡，再映射到 LDR，較符合「保留動態範圍再增強」的直覺。

### 3.3 CLAHE 實作要點

- 色彩空間：RGB ↔ YUV（BT.470-5），僅對 **Y 通道** 做 CLAHE，避免色度失真
- `grid_size = (8, 8)`，`clip_limit = 4.0`
- 直方圖以 `scatter_add_` 批次計算，避免 Python 迴圈逐 tile 累加
- 雙線性插值與 LUT 映射皆使用預配置 buffer + in-place 運算

### 3.4 Tone Mapping

採用 **線性曝光映射**（`tm_linear_inplace`）：

\[
I_{\text{LDR}} = \mathrm{clamp}(I_{\text{HDR}} \times \text{exposure} \times 255,\ 0,\ 255)
\]

優點是計算極輕（約 0.1 ms/張）、可預測，且不引入額外非線性色偏，適合即時前處理。

### 3.5 可選雙邊濾波

以 `USE_BILATERAL` flag 控制，預設關閉。  
原因：CLAHE 會放大暗部雜訊；雙邊濾波若參數過保守（小 kernel、小 `sigma_color`），視覺與偵測收益有限，卻會顯著增加 GPU 暫存與延遲。後續可作為獨立消融實驗。

---

## 4. GPU 優化設計

本專案強調：**不靠 multi-threading，靠記憶體與同步策略達成 real-time**。

### 4.1 預配置 Workspace（「挖記憶體格子」）

所有固定尺寸的中間張量在進入主迴圈前一次配置：

- CLAHE：`hist_tiles`、`luts`、`interp_tiles`、`eq_output` 等
- Pipeline：`yuv_buffer`、`rgb_out`、`gpu_batch`、`out_gpu`、pinned host buffers

迴圈內以 `copy_`、`mul_`、`out=` 覆寫，避免 CUDA Caching Allocator 反覆配置／釋放造成碎片與尖峰延遲。

### 4.2 Pinned Memory + 批次 H2D / D2H

- Host 端使用 `pin_memory()` buffer
- Chunk 級批次上傳與結果下載（`non_blocking=True`）
- 減少 pageable → device 的隱性同步開銷

### 4.3 Chunk 級同步，而非逐張同步

早期瓶頸來自每張影像後呼叫 `torch.cuda.synchronize()`，造成：

1. GPU 頻繁閒置  
2. DVFS 降頻  
3. 延遲曲線出現「斷崖／區塊階躍」

現行策略：同一 chunk 內連續 enqueue（H2D → CLAHE → TM → D2H），**僅在 chunk 結尾 synchronize 一次**，以 CUDA Event 量測各階段耗時。

### 4.4 延後磁碟 I/O

GPU 結果先累積於 `all_results`（NumPy），全部處理完成後再寫 PNG，避免 chunk 間因 `cv2.imwrite` 阻塞導致 GPU 降頻。

### 4.5 其他工程設定

- `torch.backends.cudnn.benchmark = True`
- `gc.disable()`（實驗跑批期間）
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64`（降低碎片）
- Warmup 20 張，穩定 kernel 選擇與時脈

---

## 5. 實驗設定

### 5.1 前處理模組（本專案）

| 項目 | 數值 |
|------|------|
| 影像解析度 | 800 × 1200 |
| 測試張數 | 2230 |
| Batch / Chunk size | 200 |
| Warmup | 20 |
| CLAHE grid | 8 × 8 |
| Clip limit | 4.0 |
| Bilateral | 關閉（主結果） |
| 計時方式 | CUDA Event |

### 5.2 後端偵測（YOLO26）

| 項目 | 說明 |
|------|------|
| 模型系列 | YOLO26：n / s / m / l / xl |
| 對照組 | w/o enhance & fine-tune（夜間原始輸入） |
| 實驗組 | w enhance & fine-tune（CLAHE 增強 LDR + 微調） |
| 指標 | mAP50、mAP50-95、Inference Time (ms) |

---

## 6. 實驗結果

### 6.1 前處理延遲（本專案）

在 2230 張影像上（關閉雙邊濾波）：

| 指標 | 結果 |
|------|------|
| P99 前處理延遲（CLAHE + TM） | **≈ 2.39 ms** |
| 超過 30 ms 的尖峰次數 | **0 / 2230** |
| 典型 CLAHE | ≈ 1.0～2.0 ms |
| 典型 Tone Mapping | ≈ 0.11 ms |
| 均攤 H2D / D2H | ≈ 0.88 / 0.22 ms |

相對 30 fps（33.3 ms/frame）或 60 fps（16.7 ms/frame）的預算，前處理遠低於一幀時間，可與後端推論串接而不成為瓶頸。

延遲曲線見：

- `preprocessing_latency_chart.png`：逐張前處理時間散佈
- `preprocessing_latency_breakdown.png`：H2D / CLAHE / TM / D2H 堆疊

### 6.2 YOLO26 偵測結果

#### Baseline：w/o enhance & fine-tune

| 模型 | mAP50 | mAP50-95 | Inf. Time (ms) |
|------|-------|----------|----------------|
| n | ~0.001 | ~0.0005 | 2.5 |
| s | ~0.001 | ~0.0005 | 3.7 |
| m | ~0.001 | ~0.0005 | 5.5 |
| l | ~0.001 | ~0.0005 | 6.7 |
| xl | ~0.001 | ~0.0005 | 10.9 |

（mAP50 約在 0.0008～0.002；mAP50-95 約在 0.0004～0.0007，實質上幾乎無法偵測夜間目標。）

#### Experimental：w enhance & fine-tune

| 模型 | mAP50 | mAP50-95 | Inf. Time (ms) |
|------|-------|----------|----------------|
| n | 0.672 | 0.441 | 1.7 |
| s | （介於 n～xl） | （介於 n～xl） | 2.1 |
| m | （介於 n～xl） | （介於 n～xl） | 3.2 |
| l | （介於 n～xl） | （介於 n～xl） | 3.8 |
| xl | 0.757 | 0.533 | 6.4 |

整體區間：

- **mAP50：0.672～0.757**
- **mAP50-95：0.441～0.533**
- **推論時間全面低於 baseline**（例如 xl：10.9 → 6.4 ms）

### 6.3 結果解讀

1. **精度**：夜間原始輸入對未適配的 YOLO26 幾乎無效；16-bit CLAHE 增強後的 LDR，配合 fine-tune，使 mAP 從接近 0 提升到可用水準（>0.67 mAP50）。
2. **速度**：前處理本身在毫秒級；偵測推論時間未因「多一段增強」而變慢，甚至更快——可能來自輸入對模型更友善（對比提升、動態範圍更接近訓練分佈），使內部計算量或後處理路徑更穩定。
3. **與初始目標的落差**：目前最強結果是 **enhance + fine-tune** 的組合，尚未在報告中單獨證明「僅前處理、完全不訓練」即可達到同等 mAP。此點應在討論與未來工作中明確標示。

---

## 7. 討論

### 7.1 為何傳統方法也能 real-time

CLAHE 屬傳統演算法，運算量固定、可預測。瓶頸往往不在「算子本身」，而在：

- 動態配置造成 allocator 碎片
- 過度 synchronize 造成降頻
- CPU I/O 插入 GPU 管線造成 bubble

透過預配置與 chunk 級非同步，可把傳統方法的延遲曲線壓到穩定毫秒級。

### 7.2 16-bit Domain 增強的價值

夜間暗部資訊在 uint16 中仍有解析度；先 CLAHE 再壓成 LDR，比「先壓成 LDR 再 CLAHE」更能拉出可用對比，這也解釋了後端 mAP 的劇烈提升。

### 7.3 雙邊濾波的定位

實驗中加入 Kornia bilateral blur 後，若參數偏保守，視覺差異不明顯；若加大 kernel / `sigma_color`，VRAM 暫存（unfold）與延遲會快速上升。建議將其視為可選後處理，而非主 pipeline 必備項。

### 7.4 研究目標與成果的誠實對齊

| 主張 | 目前證據強度 |
|------|----------------|
| 前處理可 real-time、穩定 | **強**（2230 張、P99≈2.39 ms、0 spikes） |
| 增強影像 + fine-tune 大幅提升 YOLO26 mAP | **強**（表格對照） |
| 僅前處理、模型免訓練即可克服夜間問題 | **尚未被本報告主表格完整證明**（需補充 pure enhance / zero-shot 數字） |

---

## 8. 結論

本實驗完成一套以 **16-bit CLAHE + 線性 Tone Mapping** 為核心的夜間前處理模組，並以 GPU workspace 預配置與非同步管線達成穩定 real-time。搭配 YOLO26 fine-tune 後，夜間偵測 mAP50 由接近 0 提升至約 **0.67～0.76**，同時維持毫秒級前處理與可接受的端到端延遲。

後續若要完整回應初始研究目標，建議補齊：

1. **消融實驗**：raw → enhance-only（no fine-tune）→ enhance + fine-tune  
2. **雙邊濾波消融**：對 mAP 與延遲的影響  
3. **與其他增強方法比較**：例如 8-bit CLAHE、Retinex、學習式 low-light enhancement  

---

## 9. 附錄

### A. 專案模組

| 檔案 | 職責 |
|------|------|
| `main.py` | 主流程、計時、開關、結果寫出 |
| `loader.py` | RAW 批次載入、H2D、白平衡 |
| `clahe.py` | GPU CLAHE workspace 與 in-place 演算法 |
| `tm.py` | Tone mapping（linear / Reinhard / log） |
| `visualize.py` | 延遲散佈圖與階段堆疊圖 |

### B. 主要超參數

```text
H, W            = 800, 1200
IMG_COUNT       = 2230
BATCH_SIZE      = 200
WARMUP_COUNT    = 20
CLAHE grid      = (8, 8)
clip_limit      = 4.0
USE_BILATERAL   = False
```

### C. 軟體依賴（節錄）

PyTorch（CUDA）、Kornia、OpenCV、NumPy、Pandas、Matplotlib / Seaborn；以 `uv` 管理環境。

### D. 圖表檔案

- `preprocessing_latency_chart.png`
- `preprocessing_latency_breakdown.png`
- YOLO26 對照表（來自偵測專案結果）
