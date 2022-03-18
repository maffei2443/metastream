import numpy as np
import random
import functools as F
np.random.seed(42)
random.seed(42)

from lightgbm import LGBMClassifier

from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.metrics import precision_score, recall_score
from imblearn.metrics import geometric_mean_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import Bunch
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import Nystroem
from skmultiflow.trees import HoeffdingTreeClassifier

# from skmultiflow.trees import HoeffdingAdaptTreeClassifier
from xgboost import XGBClassifier
# https://github.com/pavelkomarov/projection-pursuit
from skpp import ProjectionPursuitRegressor

import src.aux as A

def dumb_wrapper(fun, *args, **kwargs):
  return fun()


CLF = Bunch(
  Rf=F.partial(RandomForestClassifier, random_state=42, n_jobs=-1,),
  Dt=F.partial(DecisionTreeClassifier, random_state=42,),
  LogReg=LogisticRegression,
  Svm=SVC,
  SvmLinear=LinearSVC,
  Sgd=F.partial(SGDClassifier, n_jobs=-1),
  Knn=F.partial(KNeighborsClassifier, n_jobs=-1),
  Vfdt=HoeffdingTreeClassifier,
  # VfdtAdpt=HoeffdingAdaptTreeClassifier,
  Ppr=ProjectionPursuitRegressor,
  Nb=naive_bayes.MultinomialNB,
  NbGauss=naive_bayes.GaussianNB,
  NysSvm=F.partial(dumb_wrapper, fun=lambda: Pipeline([
    ('nys', Nystroem(random_state=42, n_jobs=-1)), 
    ('svm', LinearSVC(dual=False, ))
  ])),
  Xgb=XGBClassifier,
  Lgb=LGBMClassifier,
  LgbCustomSkWrapper=A.sklearn_lgb_clf,
)

CLF_ = list(CLF.keys())

def kappa_wrapper(y_true, y_pred, **kwargs):
  if np.array_equal(y_true, y_pred):
    return 1.0
  return cohen_kappa_score(y_true, y_pred, **kwargs)

METRICS_CLF = Bunch(
  acc=accuracy_score,
  acc_bal=balanced_accuracy_score,
  f1=f1_score,
  f1_silent=F.partial(f1_score, zero_division=0),
  geometric_mean=geometric_mean_score,
  precision=precision_score,
  recall=recall_score,
  kappa_custom=kappa_wrapper,
  kappa=cohen_kappa_score,
)

METRICS_CLF_ = list(METRICS_CLF.keys())

