# night-iris

夜間 LDR 前處理。物件模型先在原圖上標出要保護的交通像素，亮暗模型也在原圖上分亮／暗，暗部再扣掉物件像素。CLAHE 打整張矩形，融合時只有暗部改用 CLAHE 結果，物件與亮部維持原圖。下游偵測在 `yolo-lab`。

## 資料流

```
原圖 ─┬─ yolo-sem-object → object_mask（保護類）
      └─ yolo-sem-shadow → 暗類像素，再排除 object_mask → dark_mask
原圖 → 整張 CLAHE
融合：dark_mask 為真的像素用 CLAHE，其餘用原圖 → LDR
```

物件遮罩來自 class map 是否屬於 `[sem_object].object_classes`（person / rider / car / truck / bus / train / motorcycle / bicycle / traffic light / traffic sign）。路、植被、建築不保護。物件沒有被挖空或填白，只是融合時不採用 CLAHE。`source_1` 只出現在除錯圖：把物件像素調暗以便觀看，不會送進亮暗模型。

物件權重 `model/yolo26l-sem-object.pt` 與 `model/yolo26l-sem.pt` 是同一份 Ultralytics YOLO-sem（Cityscapes 類別）。亮暗權重 `model/yolo26l-sem-shadow-2.pt` 是用 SBU-shadow 訓練的，這兩個角色的權重都已放在 `model/`。物件模型目前會有抓錯的情況，這版先固定這份權重。

亮暗模型的 `imgsz` 在 `[sem_shadow]`，預設 640。物件模型的 `imgsz` 在 `[sem_object]`，預設 800。兩邊分開設定，因為改其中一邊會改變那一支模型的輸入尺度。

## 準備資料

在 `night-iris/data/images/` 放入 LDR 圖片（jpg／png）。

要重訓亮暗模型時，把原始 SBU 放在 `data/SBU-shadow/`（含 `SBU-Train`、`SBU-Test`），再執行：

```bash
uv run --package dataset-transform --directory dataset-transform python trans_script/sbu-shadow.py
uv run --package yolo-lab --directory yolo-lab python main.py --config configs/semantic_sbu_shadow_train.toml
```

訓完把 `yolo-lab/runs/sbu_shadow_sem_train/weights/best.pt` 設回 `[sem_shadow].model`。現在的 `yolo26l-sem-shadow-2.pt` 就是這樣得到的權重。

## 執行

在 repo 根目錄：

```bash
uv sync --all-packages
uv run --package night-iris --directory night-iris python main.py --config configs/default.toml
```

每張圖都寫最終 LDR 到 `result/ldr/<原檔名>`。階段除錯圖只對抽樣寫入 `result/debug/<stem>/`。`[viz].sample` 預設 8，用 `sample_seed` 從全部圖片抽出；`0` 表示每張都 dump。

| 檔案 | 內容 |
|------|------|
| `01_seg.png`、`01_seg_crops/` | 物件語意 class map（`*` 為保護類） |
| `01_source_1.png` | 除錯用：物件像素被調暗的原圖 |
| `02_sem.png` | 原圖上的亮／暗，物件區已排除 |
| `03_clahe_full.png` | 整張 CLAHE（寫回前） |
| `03_enhance_regions/` | 暗部 crop：原圖與融合後並排 |
| `04_output.png` | 融合 LDR |
| `result/timings.png` | 延遲彙總。非暖機張數 ≤ 12 才畫逐張堆疊，否則畫散點 |

`algo_ms` 加總 BGR→RGB、兩次 YOLO inference、兩次 mask 組合、CLAHE kernel、暗部寫回。讀圖、階段圖、YOLO 的 preprocess／postprocess、以及 GPU 拷貝不在這列，圖表最後一列另行標出。CLAHE 計時在資料已在 GPU 上之後才開始。前 `timing.warmup` 張不計入 P50／P99。

偵測的訓練與驗證設定在 `yolo-lab/configs/detect_bdd_*.toml`。
