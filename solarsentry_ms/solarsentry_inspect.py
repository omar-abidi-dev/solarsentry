"""SolarSentry (MindSpore) - inspect a folder of thermal solar-module images.

Usage:
    python solarsentry_inspect.py <image_folder> [--out report.csv] [--device CPU|GPU|Ascend]

Each image should be one module cropped from a drone thermal image (any size;
it is resized to 24 x 40 px). Outputs a prioritised repair list as CSV.
"""
import argparse, csv, json, os, sys
import numpy as np
from PIL import Image
import mindspore as ms
from solarsentry_model import Net, CLASSES, predict_tta

HERE = os.path.dirname(os.path.abspath(__file__))
# Planning assumptions (to be calibrated with inverter data during pilots)
LOSS = {"No-Anomaly": 0, "Cell": .03, "Cell-Multi": .10, "Cracking": .05, "Hot-Spot": .03, "Hot-Spot-Multi": .10,
        "Shadowing": .15, "Diode": .33, "Diode-Multi": .66, "Vegetation": .25, "Soiling": .08, "Offline-Module": 1.0}
PRIO = {"Offline-Module": "P1 urgent", "Diode-Multi": "P1 urgent", "Diode": "P1 urgent",
        "Cell-Multi": "P2 schedule", "Hot-Spot-Multi": "P2 schedule", "Cracking": "P2 schedule",
        "Vegetation": "P2 clean/trim", "Soiling": "P2 clean/trim",
        "Cell": "P3 monitor", "Hot-Spot": "P3 monitor", "Shadowing": "P3 monitor"}
WP, YIELD, TARIFF = 550, 1300, 1444.70  # Wp per module, kWh/kWp/yr (Java), Rp per kWh
THR_FLAG, THR_REVIEW = 0.9, 0.6  # fault-score thresholds chosen on the validation set


def load_model():
    norm = json.load(open(os.path.join(HERE, "model", "normalization.json")))
    net = Net()
    ms.load_param_into_net(net, ms.load_checkpoint(os.path.join(HERE, "model", "model.ckpt")))
    net.set_train(False)
    return net, norm["mu"], norm["sd"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("folder"); ap.add_argument("--out", default="inspection_report.csv")
    ap.add_argument("--device", default="CPU", help="CPU, GPU or Ascend")
    a = ap.parse_args()
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device(a.device)
    files = sorted(f for f in os.listdir(a.folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files: sys.exit("No images found.")
    net, mu, sd = load_model()
    X = np.stack([np.array(Image.open(os.path.join(a.folder, f)).convert("L").resize((24, 40)), np.float32) / 255. for f in files])
    x = ms.Tensor(((X - mu) / sd)[:, None].astype(np.float32))
    P = predict_tta(net, x).asnumpy()
    rows, n_review = [], 0
    for f, p in zip(files, P):
        score = 1 - p[0]
        if score <= THR_REVIEW: continue
        fault = CLASSES[1 + int(p[1:].argmax())]
        if score <= THR_FLAG:
            rows.append(dict(module=f, status="check next visit", fault=fault, fault_score=round(float(score), 3),
                             kwh_lost_per_year="", rp_lost_per_year="", priority="P3 verify")); n_review += 1; continue
        kwh = WP / 1000 * YIELD * LOSS[fault]
        rows.append(dict(module=f, status="confirmed fault", fault=fault, fault_score=round(float(score), 3),
                         kwh_lost_per_year=round(kwh), rp_lost_per_year=round(kwh * TARIFF), priority=PRIO[fault]))
    rows.sort(key=lambda r: (r["status"] != "confirmed fault", r["priority"][:2], -(r["kwh_lost_per_year"] or 0)))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["module"]); w.writeheader(); w.writerows(rows)
    conf = [r for r in rows if r["status"] == "confirmed fault"]
    print(f"Inspected {len(files)} modules: {len(conf)} confirmed faults, {n_review} to check on next visit.")
    print(f"Estimated loss from confirmed faults: {sum(r['kwh_lost_per_year'] for r in conf):,} kWh/yr "
          f"(Rp {sum(r['rp_lost_per_year'] for r in conf):,}/yr). Report: {a.out}")


if __name__ == "__main__":
    main()
