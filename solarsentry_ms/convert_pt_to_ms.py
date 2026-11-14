"""Convert the trained PyTorch weights (model.pt) into a MindSpore checkpoint (model.ckpt)
and a MindIR file (solarsentry.mindir) for ModelArts / Atlas.

Lets you use the already-trained model in MindSpore right away, before retraining.
Needs both torch and mindspore installed (only for this one-off step).

    python convert_pt_to_ms.py --pt ../model/model.pt --out model
"""
import argparse, os
import numpy as np
import torch
import mindspore as ms
from solarsentry_model import Net

# PyTorch BatchNorm names -> MindSpore names; conv/linear weight and bias keep their names
BN = {"weight": "gamma", "bias": "beta", "running_mean": "moving_mean", "running_var": "moving_variance"}

ap = argparse.ArgumentParser()
ap.add_argument("--pt", default="../model/model.pt"); ap.add_argument("--out", default="model")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

sd = torch.load(a.pt, map_location="cpu")
net = Net()
ms_names = set(net.parameters_dict())
params = []
for k, v in sd.items():
    if k.endswith("num_batches_tracked"): continue
    mod, leaf = k.rsplit(".", 1)
    name = f"{mod}.{BN[leaf]}" if f"{mod}.{BN.get(leaf, '')}" in ms_names else k
    assert name in ms_names, f"no MindSpore parameter for {k}"
    params.append({"name": name, "data": ms.Tensor(v.numpy())})
assert len(params) == len(ms_names), "parameter count mismatch"

ckpt = os.path.join(a.out, "model.ckpt")
ms.save_checkpoint(params, ckpt)
ms.load_param_into_net(net, ms.load_checkpoint(ckpt)); net.set_train(False)
ms.export(net, ms.Tensor(np.zeros((1, 1, 40, 24), np.float32)), file_name=os.path.join(a.out, "solarsentry"), file_format="MINDIR")
print(f"Converted {len(params)} tensors -> {ckpt} and {a.out}/solarsentry.mindir")
