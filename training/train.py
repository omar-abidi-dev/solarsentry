"""SolarSentry prototype: 12-class thermal fault classifier for PV modules.
Dataset: Raptor Maps InfraredSolarModules (20,000 IR images, 24x40 px, MIT licence).
"""
import json, os, time, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

torch.manual_seed(0); np.random.seed(0); random.seed(0)
ROOT = "data/InfraredSolarModules"   # unzip the Raptor Maps dataset here
OUT = "out"; os.makedirs(OUT, exist_ok=True)
meta = json.load(open(f"{ROOT}/module_metadata.json"))
CLASSES = ["No-Anomaly","Cell","Cell-Multi","Cracking","Hot-Spot","Hot-Spot-Multi","Shadowing",
           "Diode","Diode-Multi","Vegetation","Soiling","Offline-Module"]
cid = {c:i for i,c in enumerate(CLASSES)}
keys = sorted(meta, key=int)
X = np.stack([np.array(Image.open(f"{ROOT}/{meta[k]['image_filepath']}").convert("L").resize((24,40)),
              dtype=np.float32)/255. for k in keys])
y = np.array([cid[meta[k]["anomaly_class"]] for k in keys])
idx = np.arange(len(y))
tr, te = train_test_split(idx, test_size=0.15, stratify=y, random_state=0)
tr, va = train_test_split(tr, test_size=0.1, stratify=y[tr], random_state=0)
mu, sd = X[tr].mean(), X[tr].std()
Xn = (X-mu)/sd
np.save(f"{OUT}/split_test.npy", te)

class Net(nn.Module):
    def __init__(s, n=12):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(),
                                          nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU())
        s.b1,s.b2,s.b3 = blk(1,32),blk(32,64),blk(64,128)
        s.drop = nn.Dropout(0.3); s.fc = nn.Linear(128,n)
    def forward(s,x):
        x = F.max_pool2d(s.b1(x),2); x = F.max_pool2d(s.b2(x),2); x = s.b3(x)
        return s.fc(s.drop(F.adaptive_avg_pool2d(x,1).flatten(1)))

def aug(xb):
    if random.random()<.5: xb = xb.flip(3)
    if random.random()<.5: xb = xb.flip(2)
    return xb + 0.03*torch.randn_like(xb)

net = Net(); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
cnt = np.bincount(y[tr], minlength=12); w = torch.tensor((cnt.sum()/cnt)**0.5, dtype=torch.float32); w/=w.mean()
EP = 30; sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=EP*((len(tr)+127)//128))
Xt = torch.tensor(Xn[:,None]); yt = torch.tensor(y)

def predict(ids):
    net.eval(); out=[]
    with torch.no_grad():
        for i in range(0,len(ids),512):
            xb = Xt[ids[i:i+512]]
            p = F.softmax(net(xb),1)+F.softmax(net(xb.flip(3)),1)  # flip TTA
            out.append(p/2)
    return torch.cat(out).numpy()

best, t0 = 0, time.time()
for ep in range(EP):
    net.train(); perm = np.random.permutation(tr)
    for i in range(0,len(perm),128):
        b = perm[i:i+128]; xb = aug(Xt[b])
        loss = F.cross_entropy(net(xb), yt[b], weight=w, label_smoothing=0.05)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    pv = predict(va); f = f1_score(y[va], pv.argmax(1), average="macro"); a = accuracy_score(y[va], pv.argmax(1))
    if f>best: best=f; torch.save(net.state_dict(), f"{OUT}/model.pt")
    print(f"ep{ep:02d} val acc {a:.3f} macroF1 {f:.3f} ({time.time()-t0:.0f}s)", flush=True)

net.load_state_dict(torch.load(f"{OUT}/model.pt"))
pt = predict(te); yp = pt.argmax(1)
np.save(f"{OUT}/test_probs.npy", pt)
binary_true, binary_pred = y[te]!=0, yp!=0
tp = (binary_true&binary_pred).sum(); fn=(binary_true&~binary_pred).sum(); fp=(~binary_true&binary_pred).sum()
res = dict(n_train=int(len(tr)), n_val=int(len(va)), n_test=int(len(te)), epochs=EP,
    params=int(sum(p.numel() for p in net.parameters())),
    acc12=float(accuracy_score(y[te],yp)), macroF1=float(f1_score(y[te],yp,average="macro")),
    weightedF1=float(f1_score(y[te],yp,average="weighted")),
    detect_recall=float(tp/(tp+fn)), detect_precision=float(tp/(tp+fp)),
    detect_acc=float((binary_true==binary_pred).mean()),
    report=classification_report(y[te],yp,target_names=CLASSES,output_dict=True),
    cm=confusion_matrix(y[te],yp).tolist(), train_seconds=time.time()-t0, mu=float(mu), sd=float(sd))
json.dump(res, open(f"{OUT}/metrics.json","w"), indent=1)
print(classification_report(y[te],yp,target_names=CLASSES, digits=3))
print({k:v for k,v in res.items() if k not in("report","cm")})
