"""SolarSentry prototype - inspect a folder of thermal solar-module images.

Usage:
    python solarsentry_inspect.py <image_folder> [--out report.csv]

Each image should be one module cropped from a drone thermal image (any size;
it is resized to 24 x 40 px). Outputs a prioritised repair list as CSV.
"""
import argparse, csv, json, os, sys
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["No-Anomaly", "Cell", "Cell-Multi", "Cracking", "Hot-Spot", "Hot-Spot-Multi", "Shadowing",
           "Diode", "Diode-Multi", "Vegetation", "Soiling", "Offline-Module"]
# Planning assumptions (to be calibrated with inverter data during pilots)
LOSS = {"No-Anomaly": 0, "Cell": .03, "Cell-Multi": .10, "Cracking": .05, "Hot-Spot": .03, "Hot-Spot-Multi": .10,
        "Shadowing": .15, "Diode": .33, "Diode-Multi": .66, "Vegetation": .25, "Soiling": .08, "Offline-Module": 1.0}
PRIO = {"Offline-Module": "P1 urgent", "Diode-Multi": "P1 urgent", "Diode": "P1 urgent",
        "Cell-Multi": "P2 schedule", "Hot-Spot-Multi": "P2 schedule", "Cracking": "P2 schedule",
        "Vegetation": "P2 clean/trim", "Soiling": "P2 clean/trim",
        "Cell": "P3 monitor", "Hot-Spot": "P3 monitor", "Shadowing": "P3 monitor"}
WP, YIELD, TARIFF = 550, 1300, 1444.70        # Wp per module, kWh/kWp/yr (Java), Rp per kWh
THR_FLAG, THR_REVIEW = 0.9, 0.6               # fault-score thresholds chosen on the validation set


class Net(nn.Module):
    def __init__(s, n=12):
        super().__init__()
        def blk(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(),
                                            nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())
        s.b1, s.b2, s.b3 = blk(1, 32), blk(32, 64), blk(64, 128)
        s.drop = nn.Dropout(0.3); s.fc = nn.Linear(128, n)
    def forward(s, x):
        x = F.max_pool2d(s.b1(x), 2); x = F.max_pool2d(s.b2(x), 2); x = s.b3(x)
        return s.fc(s.drop(F.adaptive_avg_pool2d(x, 1).flatten(1)))


def load_model():
    norm = json.load(open(os.path.join(HERE, "model", "normalization.json")))
    net = Net(); net.load_state_dict(torch.load(os.path.join(HERE, "model", "model.pt"), map_location="cpu"))
    net.eval(); return net, norm["mu"], norm["sd"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("folder"); ap.add_argument("--out", default="inspection_report.csv")
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.folder) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files: sys.exit("No images found.")
    net, mu, sd = load_model()
    X = np.stack([np.array(Image.open(os.path.join(a.folder, f)).convert("L").resize((24, 40)), np.float32) / 255. for f in files])
    x = torch.tensor((X - mu) / sd)[:, None]
    with torch.no_grad():
        P = ((F.softmax(net(x), 1) + F.softmax(net(x.flip(3)), 1)) / 2).numpy()
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
