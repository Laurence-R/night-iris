# dataset-transform

把 SBU-shadow 轉成 YOLO-sem，讓 `yolo-lab` 可以訓練 Night-Iris 的亮暗模型。訓練與驗證仍在 `yolo-lab`。

## SBU-shadow → YOLO-sem

原始 mask 是 0/255。YOLO-sem 會把 255 當 ignore，所以這裡轉成類別索引 0/1（`bright` / `dark`）。

預設來源是 `night-iris/data/SBU-shadow/`，裡面要有 `SBU-Train` 與 `SBU-Test`，各含 `ShadowImages` 和 `ShadowMasks`。預設產出是 `yolo-lab/datasets/SBU-shadow-yolo/`。這份原始影像不進 git。

```bash
uv run --package dataset-transform --directory dataset-transform python trans_script/sbu-shadow.py
```

可用 `--src`、`--dst` 覆寫路徑。`data.yaml` 的 `path` 寫成相對於 `yolo-lab/` 的路徑，所以接下來要在 `yolo-lab` 目錄執行訓練。轉完再跑 `yolo-lab` 的 `configs/semantic_sbu_shadow_train.toml`。
