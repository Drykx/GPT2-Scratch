import numpy
import torch
import torch.nn.functional as F

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