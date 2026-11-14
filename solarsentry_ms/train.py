"""SolarSentry (MindSpore): train the 12-class thermal fault classifier for PV modules.
Dataset: Raptor Maps InfraredSolarModules (20,000 IR images, 24x40 px, MIT licence).

Local:      python train.py --data data/InfraredSolarModules --out out
ModelArts:  same script; ModelArts passes --data_url / --train_url (OBS paths
            mounted locally), which are accepted as aliases for --data / --out.
Ascend:     add --device Ascend
"""
import argparse, json, math, os, random, time
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import mindspore as ms
import mindspore.nn as nn
from solarsentry_model import Net, CLASSES, predict_tta

ap = argparse.ArgumentParser()
ap.add_argument("--data", "--data_url", default="data/InfraredSolarModules", help="unzipped Raptor Maps dataset")
ap.add_argument("--out", "--train_url", default="out")
ap.add_argument("--device", default="CPU", help="CPU, GPU or Ascend")
ap.add_argument("--epochs", type=int, default=30)
a, _ = ap.parse_known_args()  # ignore any extra arguments ModelArts adds

ms.set_context(mode=ms.PYNATIVE_MODE)
ms.set_device(a.device)
ms.set_seed(0); np.random.seed(0); random.seed(0)
ROOT, OUT = a.data, a.out; os.makedirs(OUT, exist_ok=True)

# ---- data: identical split and normalisation to the PyTorch prototype ----
meta = json.load(open(f"{ROOT}/module_metadata.json"))
cid = {c: i for i, c in enumerate(CLASSES)}
keys = sorted(meta, key=int)
X = np.stack([np.array(Image.open(f"{ROOT}/{meta[k]['image_filepath']}").convert("L").resize((24, 40)),
                       dtype=np.float32) / 255. for k in keys])
y = np.array([cid[meta[k]["anomaly_class"]] for k in keys], dtype=np.int32)
idx = np.arange(len(y))
tr, te = train_test_split(idx, test_size=0.15, stratify=y, random_state=0)
tr, va = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=0)
mu, sd = float(X[tr].mean()), float(X[tr].std())
Xn = ((X - mu) / sd)[:, None].astype(np.float32)  # (N, 1, 40, 24)
np.save(f"{OUT}/split_test.npy", te)
json.dump({"mu": mu, "sd": sd}, open(f"{OUT}/normalization.json", "w"))


def aug(xb):
    if random.random() < .5: xb = xb[:, :, :, ::-1]
    if random.random() < .5: xb = xb[:, :, ::-1, :]
    return np.ascontiguousarray(xb + 0.03 * np.random.randn(*xb.shape).astype(np.float32))


# ---- learning-rate schedule: same shape as torch OneCycleLR(max_lr=4e-3) ----
EP, BS = a.epochs, 128
steps_per_ep = (len(tr) + BS - 1) // BS
total = EP * steps_per_ep


def one_cycle(max_lr=4e-3, pct=0.3, div=25., final_div=1e4):
    lo, end = max_lr / div, max_lr / div / final_div
    up = pct * total - 1
    cos = lambda s, e, p: e + (s - e) / 2 * (1 + math.cos(math.pi * p))
    return [cos(lo, max_lr, i / up) if i <= up else cos(max_lr, end, (i - up) / (total - 1 - up)) for i in range(total)]


# ---- model, class-weighted loss, optimiser ----
net = Net()
cnt = np.bincount(y[tr], minlength=12); w = (cnt.sum() / cnt) ** 0.5; w = (w / w.mean()).astype(np.float32)
loss_fn = nn.CrossEntropyLoss(weight=ms.Tensor(w), label_smoothing=0.05)
opt = nn.AdamWeightDecay(net.trainable_params(), learning_rate=ms.Tensor(np.array(one_cycle(), np.float32)),
                         weight_decay=1e-4, eps=1e-8)
grad_fn = ms.value_and_grad(lambda xb, yb: loss_fn(net(xb), yb), None, opt.parameters)


def train_step(xb, yb):
    loss, grads = grad_fn(xb, yb)
    opt(grads)
    return loss


def predict(ids):
    net.set_train(False)
    return np.concatenate([predict_tta(net, ms.Tensor(Xn[ids[i:i + 512]])).asnumpy() for i in range(0, len(ids), 512)])


best, t0 = 0, time.time()
for ep in range(EP):
    net.set_train(True); perm = np.random.permutation(tr)
    for i in range(0, len(perm), BS):
        b = perm[i:i + BS]
        train_step(ms.Tensor(aug(Xn[b])), ms.Tensor(y[b]))
    pv = predict(va); f = f1_score(y[va], pv.argmax(1), average="macro"); acc = accuracy_score(y[va], pv.argmax(1))
    if f > best: best = f; ms.save_checkpoint(net, f"{OUT}/model.ckpt")
    print(f"ep{ep:02d} val acc {acc:.3f} macroF1 {f:.3f} ({time.time() - t0:.0f}s)", flush=True)

# ---- evaluate best checkpoint on the held-out test set ----
ms.load_param_into_net(net, ms.load_checkpoint(f"{OUT}/model.ckpt"))
pt = predict(te); yp = pt.argmax(1)
np.save(f"{OUT}/test_probs.npy", pt)
bt, bp = y[te] != 0, yp != 0
tp = (bt & bp).sum(); fn = (bt & ~bp).sum(); fp = (~bt & bp).sum()
res = dict(framework=f"MindSpore {ms.__version__}", device=a.device,
           n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)), epochs=EP,
           params=int(sum(p.size for p in net.trainable_params())),
           acc12=float(accuracy_score(y[te], yp)), macroF1=float(f1_score(y[te], yp, average="macro")),
           weightedF1=float(f1_score(y[te], yp, average="weighted")),
           detect_recall=float(tp / (tp + fn)), detect_precision=float(tp / (tp + fp)),
           detect_acc=float((bt == bp).mean()),
           report=classification_report(y[te], yp, target_names=CLASSES, output_dict=True),
           cm=confusion_matrix(y[te], yp).tolist(), train_seconds=time.time() - t0, mu=mu, sd=sd)
json.dump(res, open(f"{OUT}/metrics.json", "w"), indent=1)
print(classification_report(y[te], yp, target_names=CLASSES, digits=3))
print({k: v for k, v in res.items() if k not in ("report", "cm")})

# ---- export MindIR for ModelArts deployment and Atlas (ATC -> .om) ----
net.set_train(False)
ms.export(net, ms.Tensor(np.zeros((1, 1, 40, 24), np.float32)), file_name=f"{OUT}/solarsentry", file_format="MINDIR")
print(f"Saved {OUT}/model.ckpt and {OUT}/solarsentry.mindir")
