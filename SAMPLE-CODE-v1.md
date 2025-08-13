# Colab-ready 2D criticality experiment
# =====================================
# Compares baseline vs spectral-regularized MLP on two-moons dataset.
# Tracks accuracy, dead neuron rate, perturbation sensitivity, fractal dimension.

!pip install scikit-learn matplotlib

import math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import ticker

# -----------------
# Helper functions
# -----------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataset(n_samples=2000, noise=0.15, test_size=0.25, seed=0):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X.astype(np.float32), y.astype(np.int64), test_size=test_size, random_state=seed
    )
    return (torch.tensor(X_train), torch.tensor(y_train)), (torch.tensor(X_test), torch.tensor(y_test))

def estimate_top_singular_value(weight, n_iters=10):
    W = weight.detach()
    b = torch.randn(W.shape[1], device=W.device)
    b /= b.norm() + 1e-9
    for _ in range(n_iters):
        v = W @ b
        if v.norm() == 0:
            return 0.0
        b = W.t() @ v
        b /= b.norm() + 1e-9
    return v.norm().item() / (b.norm().item() + 1e-12)

def dead_neuron_rate_from_preacts(preacts, threshold=1e-5):
    rates = []
    for z in preacts:
        mean_abs = z.abs().mean(dim=0)
        dead = (mean_abs < threshold).float().mean().item()
        rates.append(dead)
    return float(np.mean(rates))

def perturbation_sensitivity(model, x_tensor, eps=1e-3, n_dirs=10):
    x = x_tensor.to(device)
    with torch.no_grad():
        base = model(x)
        sens_sum = 0
        for _ in range(n_dirs):
            delta = torch.randn_like(x)
            delta /= (delta.norm(dim=1, keepdim=True) + 1e-12)
            delta *= eps
            pert = x + delta
            diff = (model(pert) - base).norm(dim=1)
            sens_sum += diff.mean().item() / eps
    return sens_sum / n_dirs

def box_counting_fractal_dim(boundary_img, box_sizes):
    H, W = boundary_img.shape
    counts = []
    for s in box_sizes:
        nH = int(np.ceil(H / s))
        nW = int(np.ceil(W / s))
        count = 0
        for i in range(nH):
            for j in range(nW):
                ys = slice(i*s, min((i+1)*s, H))
                xs = slice(j*s, min((j+1)*s, W))
                block = boundary_img[ys, xs]
                if block.any():
                    count += 1
        counts.append(count if count>0 else 1)
    xs = np.log(1.0 / (np.array(box_sizes) / max(H, W)))
    ys = np.log(np.array(counts))
    slope, _ = np.polyfit(xs, ys, 1)
    return slope

# -----------------
# Model
# -----------------
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dims=[64,64], out_dim=1):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)
        self.linears = [m for m in self.net if isinstance(m, nn.Linear)]
    def forward(self, x, return_preacts=False):
        preacts = []
        h = x
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                z = layer(h)
                preacts.append(z)
                h = z
            else:
                h = layer(h)
        if return_preacts:
            return h, preacts
        return h

# -----------------
# Experiment runner
# -----------------
def run_experiment(seed=0, spectral_reg=False, spectral_lambda=0.0, target_sigma=1.0, epochs=100):
    set_seed(seed)
    (X_train, y_train), (X_test, y_test) = make_dataset(seed=seed)
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)
    model = MLP().to(device)
    opt = optim.Adam(model.parameters(), lr=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    grid_x, grid_y = np.meshgrid(np.linspace(-1.5,2.5,200), np.linspace(-1.0,1.5,200))
    grid_tensor = torch.tensor(np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1), dtype=torch.float32).to(device)
    hist = {k: [] for k in ["train_acc","test_acc","dead_rate","sens","fd","sigma"]}
    for ep in range(epochs):
        model.train()
        logits, preacts = model(X_train, return_preacts=True)
        loss = criterion(logits, y_train.float().unsqueeze(1))
        if spectral_reg:
            reg = 0.0
            for W in model.linears[:-1]:
                sigma = estimate_top_singular_value(W.weight)
                reg += (sigma - target_sigma)**2
            loss += spectral_lambda * reg
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Metrics
        with torch.no_grad():
            train_acc = ((torch.sigmoid(model(X_train)) >= 0.5).long().squeeze(1) == y_train).float().mean().item()
            test_acc = ((torch.sigmoid(model(X_test)) >= 0.5).long().squeeze(1) == y_test).float().mean().item()
            dead = dead_neuron_rate_from_preacts(preacts[:-1])
            sens = perturbation_sensitivity(model, X_train[torch.randperm(len(X_train))[:128]])
            probs = torch.sigmoid(model(grid_tensor)).cpu().numpy().reshape(200,200)
            gx, gy = np.gradient(probs)
            grad_mag = np.sqrt(gx**2 + gy**2)
            mask = (grad_mag > np.percentile(grad_mag, 99)).astype(np.uint8)
            fd = box_counting_fractal_dim(mask, [1,2,4,8,16,32,64])
            sigmas = [estimate_top_singular_value(W.weight) for W in model.linears[:-1]]
            avg_sigma = np.mean(sigmas)
        for k,v in zip(hist.keys(), [train_acc,test_acc,dead,sens,fd,avg_sigma]):
            hist[k].append(v)
    return hist, probs

# -----------------
# Multi-seed run
# -----------------
seeds = [0,1,2,3,4]
hist_b, grids_b = [], []
hist_s, grids_s = [], []
for s in seeds:
    h, g = run_experiment(s, spectral_reg=False)
    hist_b.append(h); grids_b.append(g)
    h, g = run_experiment(s, spectral_reg=True, spectral_lambda=0.5, target_sigma=1.0)
    hist_s.append(h); grids_s.append(g)

# -----------------
# Plot helpers
# -----------------
def agg(histories, key):
    arr = np.array([h[key] for h in histories])
    return arr.mean(0), arr.std(0)

def plot_with_band(ax, mean, std, label):
    ax.plot(mean, label=label)
    ax.fill_between(np.arange(len(mean)), mean-std, mean+std, alpha=0.2)

# Metric trajectories
fig, axs = plt.subplots(3,2, figsize=(10,10))
metrics = [("train_acc","Train Acc"),("test_acc","Test Acc"),("dead_rate","Dead neuron rate"),
           ("sens","Perturbation sens"),("fd","Fractal dim"),("sigma","Avg sigma")]
for ax,(m,label) in zip(axs.flatten(), metrics):
    mb, sb = agg(hist_b, m)
    ms, ss = agg(hist_s, m)
    plot_with_band(ax, mb, sb, "Baseline")
    plot_with_band(ax, ms, ss, "Spectral-reg")
    ax.set_title(label); ax.legend()
plt.tight_layout()
plt.show()

# Decision boundaries (seed 0)
fig, ax = plt.subplots(1,2, figsize=(8,4))
ax[0].imshow(grids_b[0], extent=(-1.5,2.5,-1.0,1.5), origin='lower')
ax[0].set_title("Baseline (seed 0)")
ax[1].imshow(grids_s[0], extent=(-1.5,2.5,-1.0,1.5), origin='lower')
ax[1].set_title("Spectral-reg (seed 0)")
plt.show()