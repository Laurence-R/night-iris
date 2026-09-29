# dataset-transform

dataset-transform 主要的工作內容是把原生資料集 (如 BDD100K, Cityscapes, SBU-shadow 等) 轉成 YOLO 格式，讓 YOLO 在 `yolo-lab` 中可以訓練 Night-Iris 的 **sem-object** 與 **sem-shadow** 模型以及後端 detection model。

## SBU-shadow → YOLO-sem

原始 mask 是 0/255。YOLO-sem 會把 255 當 ignore，所以這裡轉成類別索引 0/1（`bright` / `dark`）。

預設來源是 `night-iris/data/SBU-shadow/`，裡面要有 `SBU-Train` 與 `SBU-Test`，各含 `ShadowImages` 和 `ShadowMasks`。預設產出是 `yolo-lab/datasets/SBU-shadow-yolo/`。

```bash
uv run --package dataset-transform --directory dataset-transform python trans_script/sbu-shadow.py
```

可用 `--src`、`--dst` 覆寫路徑。`data.yaml` 的 `path` 寫成相對於 `yolo-lab/` 的路徑，所以接下來要在 `yolo-lab` 目錄執行訓練。轉完再跑 `yolo-lab` 的 `configs/semantic_sbu_shadow_train.toml`。

## ACDC → YOLO

預設只轉夜間。語意 mask 用 Cityscapes trainIds，無效像素寫成 255。偵測框來自官方 JSON，只有 person、rider、car、truck、bus、train、motorcycle、bicycle 這 8 類。

影像預設讀 `~/Downloads/rgb_anon_trainvaltest/rgb_anon`，標註預設讀同目錄的 `gt_trainval.zip` 與 `gt_detection_trainval.zip`。產出是 `yolo-lab/datasets/acdc-night-sem/` 和 `acdc-night-detect/`，不進 git。

```bash
uv run --package dataset-transform --directory dataset-transform python trans_script/acdc.py
```

