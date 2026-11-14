"""Figures + simulated site inspection for the SolarSentry profile."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

OUT = "out"; ROOT = "data/InfraredSolarModules"
CLASSES = ["No-Anomaly","Cell","Cell-Multi","Cracking","Hot-Spot","Hot-Spot-Multi","Shadowing",
           "Diode","Diode-Multi","Vegetation","Soiling","Offline-Module"]
m = json.load(open(f"{OUT}/metrics.json")); meta = json.load(open(f"{ROOT}/module_metadata.json"))
keys = sorted(meta, key=int); cid = {c:i for i,c in enumerate(CLASSES)}
y = np.array([cid[meta[k]["anomaly_class"]] for k in keys])
te = np.load(f"{OUT}/split_test.npy"); P = np.load(f"{OUT}/test_probs.npy"); yp = P.argmax(1)
NAVY, AMBER, INK = "#14213D", "#F2A900", "#333333"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":"#999"})

# 1. confusion matrix (row-normalised)
cm = np.array(m["cm"], float); cmn = cm/cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(7.2,6.2))
ax.imshow(cmn, cmap="YlOrBr", vmin=0, vmax=1)
for i in range(12):
    for j in range(12):
        if cmn[i,j] >= .005:
            ax.text(j,i,f"{cmn[i,j]*100:.0f}",ha="center",va="center",fontsize=7.5,
                    color="white" if cmn[i,j]>.6 else INK)
ax.set_xticks(range(12)); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
ax.set_yticks(range(12)); ax.set_yticklabels(CLASSES)
ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
ax.set_title("Confusion matrix on held-out test set (% of each true class)", fontsize=10, color=NAVY)
plt.tight_layout(); plt.savefig(f"{OUT}/cm.png", dpi=200); plt.close()

# 2. per-class F1 bar
f1 = [m["report"][c]["f1-score"] for c in CLASSES]; sup = [m["report"][c]["support"] for c in CLASSES]
order = np.argsort(f1)
fig, ax = plt.subplots(figsize=(7.2,3.6))
ax.barh([f"{CLASSES[i]}  (n={int(sup[i])})" for i in order], [f1[i] for i in order], color=NAVY)
for k,i in enumerate(order): ax.text(f1[i]+.01, k, f"{f1[i]:.2f}", va="center", fontsize=8)
ax.set_xlim(0,1.08); ax.set_xlabel("F1-score (test set)")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout(); plt.savefig(f"{OUT}/f1.png", dpi=200); plt.close()

# 3. sample predictions grid (one per class, correct where possible)
rng = np.random.default_rng(3)
fig, axes = plt.subplots(2,6, figsize=(7.4,3.9))
for c, ax in zip(range(12), axes.flat):
    cand = te[(y[te]==c)]
    pos = np.where(np.isin(te, cand))[0]
    good = [p for p in pos if yp[p]==c and P[p].max()>.6] or list(pos)
    p = rng.choice(good); img = np.array(Image.open(f"{ROOT}/{meta[keys[te[p]]]['image_filepath']}").convert("L"))
    ax.imshow(img, cmap="inferno", interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
    ok = yp[p]==y[te[p]]
    ax.set_title(f"{CLASSES[yp[p]]}\n{P[p].max()*100:.0f}% conf.", fontsize=7.2, color=NAVY if ok else "#B00020")
    ax.set_xlabel(f"true: {CLASSES[y[te[p]]]}", fontsize=6.5, color="#666")
plt.suptitle("Model output on unseen thermal module images (title = AI prediction)", fontsize=9.5, color=NAVY)
plt.tight_layout(); plt.savefig(f"{OUT}/samples.png", dpi=220); plt.close()

# 4. Simulated rooftop inspection: 1,000 modules, ~6% anomalous, drawn from test images
LOSS = {"No-Anomaly":0,"Cell":.03,"Cell-Multi":.10,"Cracking":.05,"Hot-Spot":.03,"Hot-Spot-Multi":.10,
        "Shadowing":.15,"Diode":.33,"Diode-Multi":.66,"Vegetation":.25,"Soiling":.08,"Offline-Module":1.0}
PRIO = {"Offline-Module":"P1 urgent","Diode-Multi":"P1 urgent","Diode":"P1 urgent","Cell-Multi":"P2 schedule",
        "Hot-Spot-Multi":"P2 schedule","Cracking":"P2 schedule","Cell":"P3 monitor","Hot-Spot":"P3 monitor",
        "Vegetation":"P2 clean/trim","Shadowing":"P3 monitor","Soiling":"P2 clean/trim","No-Anomaly":"-"}
WP, YIELD, TARIFF = 550, 1300, 1444.70   # Wp/module, kWh/kWp/yr (Java), Rp/kWh (PLN business tariff)
normal = np.where(y[te]==0)[0]; anom = np.where(y[te]!=0)[0]
site = np.concatenate([rng.choice(normal, 940, replace=False), rng.choice(anom, 60, replace=False)])
rng.shuffle(site)
rows = []; review = []
THR_FLAG, THR_REVIEW = 0.9, 0.6   # chosen on the validation set (FPR ~1% / ~10%)
for n,p in enumerate(site):
    score = 1-P[p][0]
    if score <= THR_REVIEW: continue
    pc = CLASSES[1+int(P[p][1:].argmax())]
    if score <= THR_FLAG:
        review.append(dict(module=f"R{n//40+1:02d}-M{n%40+1:02d}", pred=pc, true=CLASSES[y[te[p]]])); continue
    kwh = WP/1000*YIELD*LOSS[pc]
    rows.append(dict(module=f"R{n//40+1:02d}-M{n%40+1:02d}", pred=pc, true=CLASSES[y[te[p]]],
                     conf=float(score), kwh=kwh, idr=kwh*TARIFF, prio=PRIO[pc]))
rows.sort(key=lambda r:(r["prio"][:2], -r["kwh"]))
true_anom = int((y[te[site]]!=0).sum()); found = sum(1 for r in rows if r["true"]!="No-Anomaly")
rev_true = sum(1 for r in review if r["true"]!="No-Anomaly")
summ = dict(review=len(review), review_true=rev_true, modules=len(site), capacity_kwp=len(site)*WP/1000, true_anom=true_anom, flagged=len(rows),
            correctly_flagged=found, false_alarms=len(rows)-found,
            kwh=sum(r["kwh"] for r in rows), idr=sum(r["idr"] for r in rows),
            by_class={c:sum(1 for r in rows if r["pred"]==c) for c in CLASSES[1:]})
json.dump(dict(summary=summ, rows=rows), open(f"{OUT}/site.json","w"), indent=1)
print(summ)
