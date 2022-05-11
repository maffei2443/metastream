import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, TransformerMixin
from typing import Union
# from importlib import reload as rl


class OnlinePerformanceEstimation(RegressorMixin, TransformerMixin):
    def __init__(self, base_models, loss_function, verbose=False):
        self.base_models = base_models
        self.loss_function = loss_function
        self.verbose = verbose
        # self._window_x = None  # Apenas para depuração
        self._window_y = None
        self._is_fit = False
        self._predicts = None

    def pre_fit(self, X: Union[np.ndarray, pd.DataFrame], y: Union[pd.Series, np.ndarray]):
      """Convenience function so that retrain the base models is easier.

      This function must not be called unless `base_models` are not fitted or
      it is intended to update they.

      Args:
          X (Union[np.ndarray, pd.DataFrame]): train data
          y (Union[pd.Series, np.ndarray]): train label
      """

      for tup in self.base_models:
        tup.model.fit(X, y, tup.fit_params)


    def fit(self, X, y, **kwargs):
      """Following scikit-learn fit-predict pattern.

      `y` parameter is gonna be used by `predict`; however
      the most common is to pass `X`, `y` to `predict`.
      Args:
          X (np.ndarray): [description]
          y (np.ndarray): [description]

      Returns:
          [type]: [description]
      """
      # self._window_x = X
      self._window_y = y.copy()
      self._is_fit = True
      self._y_preds = [m.model.predict(X) for m in self.base_models]
      
      self._scores = [self.loss_function(y_true=self._window_y, y_pred=y_pred,) 
          for y_pred in self._y_preds
      ]
      
      return self

    def update(self, X, y):
      sx, sy = len(X), len(y)
      assert sx == sy
      # self._window_x = [*self._window_x, *X ][sx:]
      self._window_y = [*self._window_y, *y ][sy:]
      self._y_preds = [[*old_pred, *m.model.predict(X)][sx:]
                     for (old_pred, m) in zip(self._y_preds, self.base_models)]      
      self._scores = []
      for y_pred in self._y_preds:        
        try:
          loss = self.loss_function(
              y_true=self._window_y, 
              y_pred=y_pred,
            )
          self._scores.append(loss)
        except Exception as e:
          print(1, e)
          return self._window_y, y_pred
      

    def predict(self, X=None):
      if X is not None and self.verbose:
        print("[WARNING] X is not used")
      return self._scores

