# SolarSentry: MindSpore version

Same model, data split, training recipe and report logic as the PyTorch prototype, rewritten in MindSpore 2.8.

| File | What it is |
| :---- | :---- |
| `solarsentry_model.py` | The CNN (`Net`) in MindSpore, plus flip test-time augmentation |
| `train.py` | Training script. Runs locally or as a ModelArts training job |
| `solarsentry_inspect.py` | Inspection tool: folder of images in, prioritised repair CSV out |
| `convert_pt_to_ms.py` | One-off: turns your trained `model.pt` into `model.ckpt` + `solarsentry.mindir` |
| `requirements.txt` | Packages needed |
| `setup_and_run.bat` | One-click setup, conversion, test and training on Windows |
| `model/` | Weights converted from the PyTorch model (`model.ckpt`, `solarsentry.mindir`) |
| `out/` | Model retrained in MindSpore, with its test metrics |

## 1. Install (use Python 3.11)

Tested with MindSpore 2.8.0 on Python 3.11. `pip install mindspore` fails on Python 3.13, so use 3.11:

```sh
python3.11 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Use your existing trained model right away (optional)

From inside `solarsentry_ms/` (needs `pip install torch` for this step only):

```sh
python convert_pt_to_ms.py --pt ../model/model.pt --out model
cp ../model/normalization.json model/
python solarsentry_inspect.py ../sample_images --out inspection_report.csv
```

You should get the same output as the PyTorch prototype (18 confirmed faults, 2 to check, on the 28 sample images).

## 3. Retrain in MindSpore

Unzip the Raptor Maps dataset to `data/InfraredSolarModules`, then:

```sh
python train.py --data data/InfraredSolarModules --out out
```

Outputs in `out/`: `model.ckpt`, `normalization.json`, `metrics.json`, `solarsentry.mindir`.
Copy `model.ckpt` and `normalization.json` into `model/` to use them with `solarsentry_inspect.py`.

Speed: about 2.5 minutes per epoch on a 2-core CPU (about 75 minutes for 30 epochs), slower than PyTorch on CPU.
On a ModelArts GPU or Ascend instance it is much faster: add `--device GPU` or `--device Ascend`.

## 4. Train on ModelArts

1. Upload this folder (the code) and the dataset folder to an OBS bucket.
2. ModelArts > Training Jobs > Create: custom algorithm, MindSpore image, boot file `train.py`.
3. Input `data_url` = dataset OBS folder; output `train_url` = an OBS results folder. The script accepts these names.
4. Hyperparameter `device` = `Ascend` if you choose an Ascend flavour.

## 5. Atlas (Ascend edge)

`solarsentry.mindir` is the deployable model. On the Atlas, convert it to an offline model with CANN's ATC tool,
or run it directly with MindSpore Lite. Input shape is `1,1,40,24` (batch, channel, height, width), float32,
already normalised with `mu`/`sd` from `normalization.json`.

## Differences from the PyTorch version

- PyTorch defaults that differ in MindSpore are set explicitly (conv padding and bias, BatchNorm momentum), so the
  network is mathematically identical. Converted weights give the same probabilities (max difference about 1e-7).
- `OneCycleLR` is reproduced as a per-step learning-rate list. PyTorch's OneCycle also cycles Adam's beta1; the
  MindSpore version keeps beta1 fixed, so retrained metrics may differ slightly from the PyTorch run.
