import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from typing import Tuple
from sklearn.base import ClassifierMixin, BaseEstimator
from collections import Counter
from matplotlib import pyplot as plt


random_params = dict(
    random_state=42,
    scoring=make_scorer(accuracy_score),
    cv=3,
    n_jobs=-1,
    return_train_score=True,
    refit=False,
)

grid_params = dict(
    scoring=make_scorer(accuracy_score),
    cv=3,
    n_jobs=-1,
    return_train_score=True,
    refit=False,
)


def random_tuner(model=None, X=None, y=None, params=None, opt_params={}):
    opt = RandomizedSearchCV(model, params, **opt_params)
    opt.fit(X, y)
    model.set_params(**opt.best_params_)
    return opt


def grid_tuner(model=None, X=None, y=None, params=None, opt_params={}):
    opt = GridSearchCV(model, params, **opt_params)
    opt.fit(X, y)
    model.set_params(**opt.best_params_)
    return opt


def prefixify(d: dict, p: str):
    """Create dictionary with same values but with each key prefixed by `{p}_`."""
    return {f'{p}_{k}': v for (k, v) in d.items()}


def smallest_labelizer(metrics: dict) -> Tuple[str, float]:
    """Given dict of metrics result, returns (key, metrics[key]) whose value is the smallest."""
    items = list(metrics.items())
    # print(items)
    label, smaller = items[0]
    for model_name, val in items[1:]:
        if val < smaller:
            label, smaller = model_name, val
    return label, smaller


def smallest_labelizer_draw(metrics: dict, *args, **kwargs) -> Tuple[str, float]:
    """Given dict of metrics result, returns (key, metrics[key]) whose value is maximal."""
    metric_values = list(metrics.values())
    metric_keys = list(metrics.keys())
    # print(items)
    smaller = metric_values[0]
    draws = [0]
    for idx, val in enumerate(metric_values[1:], start=1):
        if val < smaller:
            smaller = val
            draws = [idx]
        elif val == smaller:
            draws.append(idx)

    if len(draws) > 1:  # Empate. Ver qual é a classe + predita e vai
        return 'draw', smaller
    else:
        return metric_keys[draws[0]], smaller


def biggest_labelizer(metrics: dict, *args, **kwargs) -> Tuple[str, float]:
    """Given dict of metrics result, returns (key, metrics[key]) whose value is the smallest."""
    items = list(metrics.items())
    # print(items)
    label, bigger = items[0]
    for model_name, val in items[1:]:
        if val > bigger:
            label, bigger = model_name, val
    return label, bigger


def biggest_labelizer_draw(metrics: dict, *args, **kwargs) -> Tuple[str, float]:
    """Given dict of metrics result, returns (key, metrics[key]) whose value is maximal."""
    metric_values = list(metrics.values())
    metric_keys = list(metrics.keys())
    # print(items)
    big = metric_values[0]
    draws = [0]
    for idx, val in enumerate(metric_values[1:], start=1):
        if val > big:
            big = val
            draws = [idx]
        elif val == big:
            draws.append(idx)

    if len(draws) > 1:  # Empate. Ver qual é a classe + predita e vai
        return 'draw', big
    else:
        return metric_keys[draws[0]], big


def biggest_labelizer_majority(metrics: dict, dist: dict) -> Tuple[str, float]:
    """Given dict of metrics result, returns (key, metrics[key]) whose value is maximal."""
    metric_values = list(metrics.values())
    metric_keys = list(metrics.keys())
    # print(items)
    big = metric_values[0]
    draws = [0]
    for idx, val in enumerate(metric_values[1:], start=1):
        if val > big:
            big = val
            draws = [idx]
        elif val == big:
            draws.append(idx)

    if len(draws) > 1:  # Empate. Ver qual é a classe + predita e vai
        keys_dist = list(metrics.keys())
        qtds = [dist[metric_keys[k]] for k in draws]
        # print("QTDS: ", qtds)
        ma = max(qtds)
        # Se não á uma único label com quantidade maximal
        # de labels, escolher aleatoriamente entre as
        # classes disponíveis
        if ma in qtds[qtds.index(ma)+1:]:
            # print("EMPATE MAXIMO")
            return keys_dist[np.random.choice(draws)], big
        else:
            # print("APENAS UM MAXIMO")
            # Escolhe modelo que vem primeiro na lista de modelos.
            return keys_dist[draws[qtds.index(ma)]], big
    else:
        # print("easy peasy")
        return metric_keys[draws[0]], big


def biggest_labelizer_balance(metrics: dict, *args, **kwargs) -> Tuple[str, float]:
    """Given dict of metrics result, returns (key, metrics[key]) whose value is maximal."""
    metric_values = list(metrics.values())
    metric_keys = list(metrics.keys())
    big = metric_values[0]
    draws = [0]
    for idx, val in enumerate(metric_values[1:], start=1):
        if val > big:
            big = val
            draws = [idx]
        elif val == big:
            draws.append(idx)

    if len(draws) > 1:  # Empate: usa qualquer um como rótulo.
        keys_dist = list(metrics.keys())
        return keys_dist[np.random.choice(draws)], big
    else:
        return metric_keys[draws[0]], big


def biggest_labelizer_arbitrary(metrics: dict, choice: str, *args, **kwargs) -> Tuple[str, float]:
  """Given dict of metrics result, returns (key, metrics[key]) whose value is maximal."""
  metric_values = list(metrics.values())
  metric_keys = list(metrics.keys())
  # print(items)
  big = metric_values[0]
  draws = [0]
  for idx, val in enumerate(metric_values[1:], start=1):
    if val > big:
      big = val
      draws = [idx]
    elif val == big:
      draws.append(idx)
  
  if len(draws) > 1 and choice in (metric_keys[idx] for idx in draws):
      return choice, big
  return metric_keys[draws[0]], big

def biggest_labelizer_majority_arbitrary(metrics: dict, dist: dict, choice: str, *args, **kwargs) -> Tuple[str, float]:
  """Given dict of metrics result, returns (key, metrics[key]) whose value is maximal."""
  metric_values = list(metrics.values())
  metric_keys = list(metrics.keys())
  # print(items)
  big = metric_values[0]
  draws = [0]
  for idx, val in enumerate(metric_values[1:], start=1):
    if val > big:
      big = val
      draws = [idx]
    elif val == big:
      draws.append(idx)
  
  if len(draws) > 1:    
    keys_dist = list(metrics.keys())
    qtds = [dist[metric_keys[k]] for k in draws]
    # print("QTDS: ", qtds)
    ma = max(qtds)
    # Se não á uma único label com quantidade maximal
    # de labels, ver se a classe `choice`
    # está entre as opções. Se sim, retorna ela. Senão, retorna uma aleatória
    if ma in qtds[qtds.index(ma)+1:]:
        if choice in (metric_keys[idx] for idx in draws):
          return choice, big
        # print("EMPATE MAXIMO")
        return keys_dist[np.random.choice(draws)], big
    else:
        return keys_dist[draws[qtds.index(ma)]], big  
  return metric_keys[draws[0]], big


def su_extractor(X, y, ext):
    y = np.array(y)
    ext.fit(X, y)
    return dict(zip(*ext.extract()))


def unsu_extractor(X, ext):
    ext.fit(X)
    return dict(zip(*ext.extract()))


class sklearn_lgb_clf(BaseEstimator, ClassifierMixin):
    def __init__(self, fit_params: dict, classes: list,):
        self.fit_params = fit_params.copy()
        self.classes = classes.copy()
        self.lgb = None
        self.label_encoder = LabelEncoder().fit(classes)

    def fit(self, X, y, incremental = False, **fit_params):
        self.fit_params.update(fit_params)
        self.lgb = lgb.train(
            self.fit_params,
            init_model=self.lgb if incremental else None,
            train_set=lgb.Dataset(X, self.label_encoder.transform(y)),
        )
        return self

    def predict_proba(self, X):
        return self.lgb.predict(X)

    def predict(self, X):
        pred = self.lgb.predict(X)
        try:
            return pred.argmax(axis=1)
        except:
            return pred[0]


def IGNORE_WARNING():
    print("DISABLING WARNING: bad practice; use at your own risk.")

    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn
    warnings.filterwarnings('ignore')


def plot_pie(lis, **kwargs,):
    ct = Counter(lis)
    values, labels = lzip(*sorted(ct.items()))
    kwargs['autopct'] = '%.1f'
    aux = plt
    if kwargs.get('ax'):
        aux = kwargs.pop('ax')
    return aux.pie(labels,  labels=values, **kwargs)


def lzip(*x):
    return list(zip(*x))

# Inspired by
# https://towardsdatascience.com/fast-and-robust-sliding-window-vectorization-with-numpy-3ad950ed62f5
def TimeSeriesCVWindows(n, train, test, gap=0, step=1):    
    slide = np.arange(n).reshape(-1, 1) * step
    tr = np.arange(train)
    ntr = np.repeat(np.expand_dims(tr, 0), n, axis=0) + slide
    
    te = np.arange(test) + train
    nte = np.repeat(np.expand_dims(te, 0), n, axis=0) + slide + gap
    return ntr, nte
