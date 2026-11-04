"""Small configurable models for the reviewer revision; no test data in fitting."""
import copy
import time
import numpy as np
import torch
from torch import nn
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor

torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)


class Representation:
    def __init__(self, imputation='sentinel', mask=False, scaled=False):
        self.imputation, self.mask, self.scaled = imputation, mask, scaled

    def fit(self, X):
        count = np.isfinite(X).sum(axis=0)
        self.fill = np.divide(np.nansum(X, axis=0), count,
                              out=np.full(X.shape[1], -140.), where=count > 0)
        if self.imputation == 'sentinel': self.fill[:] = -140.
        z = self.raw(X)
        self.mean = z.mean(axis=0) if self.scaled else np.zeros(z.shape[1])
        self.scale = z.std(axis=0) if self.scaled else np.ones(z.shape[1])
        self.scale[self.scale < 1e-12] = 1.
        return self

    def raw(self, X):
        observed = np.isfinite(X)
        z = np.where(observed, X, self.fill)
        return np.column_stack([z, observed.astype(float)]) if self.mask else z

    def transform(self, X): return (self.raw(X) - self.mean) / self.scale


class Network(nn.Module):
    def __init__(self, inputs, outputs, kind, width, window):
        super().__init__()
        self.kind, self.window = kind, window
        if kind == 'cnn':
            self.layers = nn.Sequential(nn.Conv1d(2, width, 3), nn.ReLU(),
                nn.Conv1d(width, width, 3), nn.ReLU(), nn.Flatten(),
                nn.Linear(width * (window - 4), 32), nn.ReLU(), nn.Linear(32, outputs))
        else:
            self.layers = nn.Sequential(nn.Linear(inputs, width), nn.ReLU(),
                nn.Linear(width, width // 2), nn.ReLU(), nn.Linear(width // 2, outputs))

    def forward(self, x):
        if self.kind == 'cnn': x = x.reshape(-1, 2, self.window).flip(-1)
        return self.layers(x)


class Fitted:
    def __init__(self, representation, model, neural=False, y_mean=None, y_scale=None):
        self.representation, self.model, self.neural = representation, model, neural
        self.y_mean, self.y_scale = y_mean, y_scale

    def predict(self, X):
        z = self.representation.transform(X)
        if not self.neural: return np.asarray(self.model.predict(z)).reshape(len(X), -1)
        self.model.eval()
        with torch.no_grad():
            a = torch.as_tensor(z, dtype=torch.float32)
            pred = torch.cat([self.model(b) for b in a.split(4096)]).numpy()
        return pred * self.y_scale + self.y_mean


def configurations(family):
    if family == 'knn': return [dict(k=k) for k in [3, 9, 21, 45]]
    if family == 'ridge': return [dict(alpha=a) for a in [.01, .1, 1., 10., 100., 1000.]]
    if family == 'rf': return [dict(leaf=l, max_features=f) for l in [2, 5, 15] for f in [.7, 1.]]
    if family == 'boosting': return [dict(leaves=l, l2=r) for l in [7, 15, 31] for r in [0., 1.]]
    if family == 'mlp': return [dict(width=w, lr=.001, decay=d) for w in [32, 64] for d in [.0001, .01]]
    if family == 'cnn': return [dict(width=w, lr=.001, decay=d) for w in [8, 16] for d in [.0001, .01]]
    raise ValueError(family)


def fit(Xtr, ytr, Xva, yva, *, family, config, seed, score, window=24, representation=None):
    start = time.perf_counter()
    spec = representation or dict(imputation='sentinel', mask=False, scaled=family in ['ridge', 'mlp', 'cnn'])
    rep = Representation(**spec).fit(Xtr)
    x = rep.transform(Xtr)
    ytr, yva = np.asarray(ytr).reshape(len(ytr), -1), np.asarray(yva).reshape(len(yva), -1)
    if family not in ['mlp', 'cnn']:
        if family == 'knn': m = KNeighborsRegressor(n_neighbors=config['k'], weights='distance', n_jobs=2)
        elif family == 'ridge': m = Ridge(alpha=config['alpha'])
        elif family == 'rf': m = RandomForestRegressor(n_estimators=100, min_samples_leaf=config['leaf'], max_features=config['max_features'], n_jobs=2, random_state=seed)
        elif family == 'boosting': m = HistGradientBoostingRegressor(max_iter=150, max_leaf_nodes=config['leaves'], l2_regularization=config['l2'], learning_rate=.05, early_stopping=False, random_state=seed)
        else: raise ValueError(family)
        m.fit(x, ytr[:, 0] if ytr.shape[1] == 1 else ytr)
        fitted = Fitted(rep, m)
        return fitted, [], dict(train_seconds=time.perf_counter()-start, validation_score=score(yva, fitted.predict(Xva)))
    torch.manual_seed(seed)
    ym, ys = ytr.mean(axis=0), ytr.std(axis=0)
    ys[ys < 1e-12] = 1.
    xx = torch.as_tensor(x, dtype=torch.float32)
    yy = torch.as_tensor((ytr-ym)/ys, dtype=torch.float32)
    m = Network(x.shape[1], ytr.shape[1], family, config['width'], window)
    fitted = Fitted(rep, m, True, ym, ys)
    opt = torch.optim.Adam(m.parameters(), lr=config['lr'], weight_decay=config['decay'])
    lossfn = nn.MSELoss(); best = np.inf; best_epoch = 0; history = []
    for epoch in range(1, 81):
        m.train(); total = 0.
        for ix in torch.randperm(len(xx)).split(256):
            opt.zero_grad(); loss = lossfn(m(xx[ix]), yy[ix]); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.); opt.step()
            total += float(loss.detach()) * len(ix)
        val = score(yva, fitted.predict(Xva))
        train = score(ytr, fitted.predict(Xtr))
        if not np.isfinite(val): raise ValueError('Non-finite validation score')
        history.append(dict(epoch=epoch, train_loss_scaled=total/len(xx), train_score=train, validation_score=val))
        if val < best-1e-6:
            best, best_epoch, state = val, epoch, copy.deepcopy(m.state_dict())
        if epoch-best_epoch >= 12: break
    m.load_state_dict(state)
    return fitted, history, dict(train_seconds=time.perf_counter()-start, validation_score=best,
        best_epoch=best_epoch, epochs_run=len(history), parameters=sum(p.numel() for p in m.parameters()))
