# Train a GTSRB CNN (PyTorch, CPU) — 43 classes, 48x48 input
# Data: train/<class>/*.png ; eval: test/ + Test.csv (ROI crops)
import csv, os, sys, time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(8)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE_DIR, "dataset_GTSRB")
OUT_DIR = BASE_DIR
SIZE = 48
BATCH = 128
EPOCHS = 30
VAL_FRAC = 0.05

def imread_unicode(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)

# ---------------- data loading ----------------
def load_train():
    X, y = [], []
    print('Loading training images...', flush=True)
    for cls in range(43):
        d = os.path.join(SRC, 'train', str(cls))
        for f in os.listdir(d):
            if not f.endswith('.png'):
                continue
            img = imread_unicode(os.path.join(d, f))
            if img is None:
                continue
            img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
            X.append(img)
            y.append(cls)
    X = np.stack(X)                       # (N,48,48,3) uint8
    y = np.array(y, np.int64)
    print(f'Loaded {len(y)} training images', flush=True)
    return X, y

def load_test():
    rows = []
    with open(os.path.join(SRC, 'Test.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append((r['Path'], int(r['Roi.X1']), int(r['Roi.Y1']),
                         int(r['Roi.X2']), int(r['Roi.Y2']), int(r['ClassId'])))
    X, y = [], []
    print('Loading test images...', flush=True)
    for path, x1, y1, x2, y2, cls in rows:
        img = imread_unicode(os.path.join(SRC, *path.split('/')))
        if img is None:
            continue
        crop = img[y1:y2, x1:x2]
        crop = cv2.resize(crop, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        X.append(crop)
        y.append(cls)
    return np.stack(X), np.array(y, np.int64)

# ---------------- augmentation (batched, on-the-fly) ----------------
def augment(xb):
    # xb: (B,3,48,48) float in [0,1]
    B = xb.shape[0]
    ang = (torch.rand(B, device=xb.device) * 24 - 12) * np.pi / 180
    sc = 1.0 + (torch.rand(B, device=xb.device) * 0.2 - 0.1)
    tx = torch.rand(B, device=xb.device) * 0.16 - 0.08
    ty = torch.rand(B, device=xb.device) * 0.16 - 0.08
    zero = torch.zeros_like(ang)
    theta = torch.stack([
        torch.stack([torch.cos(ang) * sc, -torch.sin(ang) * sc, tx], 1),
        torch.stack([torch.sin(ang) * sc,  torch.cos(ang) * sc, ty], 1)], 1)  # (B,2,3)
    grid = F.affine_grid(theta, xb.shape, align_corners=False)
    xb = F.grid_sample(xb, grid, align_corners=False, padding_mode='zeros')
    c = 1.0 + (torch.rand(B, 1, 1, 1) * 0.5 - 0.25)     # contrast
    b = (torch.rand(B, 1, 1, 1) * 0.3 - 0.15)           # brightness
    xb = (xb * c + b).clamp(0, 1)
    return xb

# ---------------- model (classic GTSRB architecture) ----------------
class SignCNN(nn.Module):
    def __init__(self, n=43):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 12 * 12, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, n),
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)

# ---------------- main ----------------
def main():
    t0 = time.time()
    weight_path = os.path.join(OUT_DIR, 'gtsrb_cnn_best.pt')
    predictions_path = os.path.join(OUT_DIR, 'test_preds.npy')
    # A training run always starts from a new model; no previous checkpoint is reused.
    for path in (weight_path, predictions_path):
        if os.path.exists(path):
            os.remove(path)
    X, y = load_train()

    # stratified val split
    rng = np.random.RandomState(0)
    val_idx, tr_idx = [], []
    for cls in range(43):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        k = max(1, int(len(idx) * VAL_FRAC))
        val_idx += idx[:k].tolist()
        tr_idx += idx[k:].tolist()
    tr_idx, val_idx = np.array(tr_idx), np.array(val_idx)
    print(f'train {len(tr_idx)} / val {len(val_idx)}', flush=True)

    X_t = torch.from_numpy(X)                       # uint8 NHWC
    y_t = torch.from_numpy(y)

    model = SignCNN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[20, 26], gamma=0.2)
    best_val = 0.0

    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(tr_idx))
        tot_loss, tot_correct = 0.0, 0
        for i in range(0, len(perm), BATCH):
            idx = tr_idx[perm[i:i + BATCH]]
            xb = X_t[idx].float().div_(255).permute(0, 3, 1, 2)
            xb = augment(xb)
            yb = y_t[idx]
            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            tot_correct += (out.argmax(1) == yb).sum().item()
        sched.step()

        model.eval()
        with torch.no_grad():
            vc = 0
            for i in range(0, len(val_idx), BATCH):
                idx = val_idx[i:i + BATCH]
                xb = X_t[idx].float().div_(255).permute(0, 3, 1, 2)
                out = model(xb)
                vc += (out.argmax(1) == y_t[idx]).sum().item()
        val_acc = vc / len(val_idx)
        tr_acc = tot_correct / len(tr_idx)
        marker = ''
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), os.path.join(OUT_DIR, 'gtsrb_cnn_best.pt'))
            marker = '  <- saved'
        print(f'epoch {ep:2d}/{EPOCHS}  loss {tot_loss/len(tr_idx):.4f}  '
              f'train acc {tr_acc:.4f}  val acc {val_acc:.4f}{marker}  '
              f'[{time.time()-t0:.0f}s]', flush=True)

    # ---------------- final test evaluation ----------------
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model.eval()
    Xt, yt = load_test()
    Xt_t = torch.from_numpy(Xt)
    yt_t = torch.from_numpy(yt)
    preds = []
    with torch.no_grad():
        for i in range(0, len(Xt_t), BATCH):
            xb = Xt_t[i:i + BATCH].float().div_(255).permute(0, 3, 1, 2)
            preds.append(model(xb).argmax(1))
    preds = torch.cat(preds).numpy()
    acc = (preds == yt).mean()
    print(f'\nFINAL TEST ACCURACY: {acc:.4f}  ({(preds == yt).sum()}/{len(yt)})', flush=True)
    per_class = []
    for cls in range(43):
        m = yt == cls
        if m.sum():
            per_class.append((cls, (preds[m] == cls).mean(), int(m.sum())))
    print('Per-class accuracy (class, acc, n):', flush=True)
    for cls, a, n in per_class:
        print(f'  {cls:2d}: {a:.3f}  (n={n})', flush=True)
    np.save(predictions_path, preds)
    print(f'Total time: {(time.time()-t0)/60:.1f} min', flush=True)

if __name__ == '__main__':
    main()
