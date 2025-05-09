import numpy
import torch
import torch.nn.functional as F
from torch.nn import Tanh
import matplotlib as plt

def cmp(s,dt,t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt,t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f"{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s | maxdiff: {maxdiff}}")


def build_dataset(words: str, block_size: int, corpus: str) -> tuple[torch.Tensor, torch.Tensor]:

    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + ".":
            ix = corpus[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X,Y



def plot_histograms(layers, title, attr='out', is_grad=False):
    plt.figure(figsize=(20, 4))
    legends = []
    for i, layer in enumerate(layers[:-1]):  # skip output layer
        if isinstance(layer, Tanh):
            t = getattr(layer, attr)
            if is_grad:
                t = t.grad
                print(f'Layer {i} ({layer.__class__.__name__}): mean {t.mean():+.6f}, std {t.std():.2e}')
            else:
                saturated = (t.abs() > 0.97).float().mean() * 100
                print(f'Layer {i} ({layer.__class__.__name__}): mean {t.mean():+.2f}, std {t.std():.2f}, saturated: {saturated:.2f}%')
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'Layer {i} ({layer.__class__.__name__})')
    plt.legend(legends)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_update_ratios(parameters, ud):
    plt.figure(figsize=(20, 4))
    legends = []
    for i, p in enumerate(parameters):
        if p.ndim == 2:
            plt.plot([ud[j][i] for j in range(len(ud))])
            legends.append(f'Param {i}')
    plt.axhline(y=1e-3, color='k', linestyle='--', label='Target ratio (~1e-3)')
    plt.legend(legends)
    plt.title('Update Ratios Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_weight_gradients(parameters):
    plt.figure(figsize=(20, 4))
    legends = []
    for i, p in enumerate(parameters):
        if p.ndim == 2:
            t = p.grad
            ratio = t.std() / p.std()
            print(f'Weight {tuple(p.shape)} | mean {t.mean():+.6f} | std {t.std():.2e} | grad:data ratio {ratio:.2e}')
            hy, hx = torch.histogram(t, density=True)
            plt.plot(hx[:-1].detach(), hy.detach())
            legends.append(f'{i} {tuple(p.shape)}')
    plt.legend(legends)
    plt.title('Weights Gradient Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
