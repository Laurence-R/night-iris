# yolo-lab

YOLO 實驗場。資料已是 YOLO 格式時，在這裡訓練、驗證、匯出 TensorRT。格式轉換在 [`dataset-transform`](../dataset-transform/)。

偵測設定共用同一組欄位：`name`、`mode`、`model`、`data`、`device`，以及 `[train]`、`[val]`、`[predict]`、`[export]`。預訓練的偵測設定另有 `model_template`，把 `{size}` 換成 `n/s/m/l/x`。微調驗證沒有 template，因為訓練只寫出一個 `runs/<name>/weights/best.pt`。

## 本機才有的檔案

| 內容 | 路徑 |
|------|------|
| COCO 預訓練權重 | `yolo-coco/yolo26{n,s,m,l,x}.pt`（自行下載，不進 git） |
| 原夜間道路資料 | `datasets/bdd10k-night/70-15-15/` |
| Night-Iris 處理後的資料 | `datasets/bdd10k-night-iris/70-15-15/` |
| SBU 轉成的 YOLO-sem | `datasets/SBU-shadow-yolo/` |
| 偵測微調權重 | `runs/<config.name>/weights/best.pt` |

`runs/`、`experiments/`、COCO 權重、`.engine`、`.onnx` 不進版控。現有的夜間 baseline `best.pt` 超過 GitHub 單檔 100MB，所以不能放進這個 repo。Night-Iris 推論用的語意權重在 [`night-iris/model/`](../night-iris/model/)，那裡才是會提交的模型。

`main.py` 把 Ultralytics 的 `project` 設成這個套件底下的 `runs`（絕對路徑）。若全域 `runs_dir` 指到別的磁碟，先改成你 clone 下來的路徑，例如：

```bash
uv run --package yolo-lab yolo settings runs_dir="<這個 repo 的絕對路徑>/yolo-lab/runs"
```

## Detect 實驗（BDD 夜間，70/15/15）

預訓練 val 會把 `bike`→`bicycle`、`motor`→`motorcycle`。`rider` 和 `traffic sign` 在 COCO 沒有對應類別，那些框會被略過。微調權重和資料集都是這 10 類，所以不再 remap。

原圖與 Night-Iris 圖的訓練 `name` 不同，避免兩次訓練寫進同一個 `runs` 目錄。Night-Iris 那次的 predict 來源是處理後的 valid 圖，不是原圖。

| Config | 模式 | 資料 | 權重 |
|--------|------|------|------|
| [`configs/detect_bdd_orig_val_pretrained.toml`](configs/detect_bdd_orig_val_pretrained.toml) | val | `bdd10k-night/70-15-15` | `yolo-coco/` |
| [`configs/detect_bdd_night_iris_val_pretrained.toml`](configs/detect_bdd_night_iris_val_pretrained.toml) | val | `bdd10k-night-iris/70-15-15` | `yolo-coco/` |
| [`configs/detect_bdd_orig_train.toml`](configs/detect_bdd_orig_train.toml) | train | `bdd10k-night/70-15-15` | 從 `yolo-coco/` 微調，`name = bdd10k-night-baseline` |
| [`configs/detect_bdd_night_iris_train.toml`](configs/detect_bdd_night_iris_train.toml) | train | `bdd10k-night-iris/70-15-15` | 從 `yolo-coco/` 微調，`name = bdd10k-night-iris-baseline` |
| [`configs/detect_bdd_orig_val_finetuned.toml`](configs/detect_bdd_orig_val_finetuned.toml) | val | `bdd10k-night/70-15-15` | `runs/bdd10k-night-baseline/weights/best.pt` |
| [`configs/detect_bdd_night_iris_val_finetuned.toml`](configs/detect_bdd_night_iris_val_finetuned.toml) | val | `bdd10k-night-iris/70-15-15` | `runs/bdd10k-night-iris-baseline/weights/best.pt` |

## Semantic 實驗（SBU-shadow 亮／暗）

Night-Iris 的 `[sem_shadow]` 權重在這裡訓。先用 [`dataset-transform/trans_script/sbu-shadow.py`](../dataset-transform/trans_script/sbu-shadow.py) 產出 `datasets/SBU-shadow-yolo/data.yaml`，再跑：

```bash
uv run --package dataset-transform --directory dataset-transform python trans_script/sbu-shadow.py
uv run --package yolo-lab --directory yolo-lab python main.py --config configs/semantic_sbu_shadow_train.toml
```

起點是 `night-iris/model/yolo26l-sem.pt`。訓完把 `runs/sbu_shadow_sem_train/weights/best.pt` 設到 `night-iris/configs/default.toml` 的 `[sem_shadow].model`。目前推論用的 `yolo26l-sem-shadow-2.pt` 就是這條流程的結果。

| Config | 模式 | 說明 |
|--------|------|------|
| [`configs/semantic_sbu_shadow_train.toml`](configs/semantic_sbu_shadow_train.toml) | train | `task = "semantic"`，SBU 亮／暗，200 epoch |

## 執行

```bash
uv run --package yolo-lab --directory yolo-lab python main.py --config configs/detect_bdd_orig_val_pretrained.toml --model-size x --model yolo-coco/yolo26x.pt
uv run --package yolo-lab --directory yolo-lab python main.py --config configs/detect_bdd_orig_train.toml --model-size x --model yolo-coco/yolo26x.pt
```

- `--model-size`：`n` / `s` / `m` / `l` / `x` / `all`（有 `model_template` 時才會展開）
- `--model`：直接指定權重，不再套 template

## 結果記錄

`val` 寫入的目錄名是 config 的 `name`：

```
experiments/<name>/
  metrics.json
  config.snapshot.toml
```

同 config 再跑不同 size 會覆寫同目錄；看 snapshot 裡的 `model` 才知道用了哪個權重。

### 類別 ID 對齊（預訓練 val）

Roboflow 匯出的標註是連續的 `0..nc-1`，COCO 預訓練權重仍輸出原始 COCO ID。`val` 時若 `model.names` 與 `data.yaml` 的同索引名稱不一致，`main.py` 會依類別名稱把標註 remap 到模型的 ID，再算 mAP。微調權重若已與資料集對齊則略過。對照表寫在 `metrics.json`，暫存資料在 `experiments/_class_remap/`。
