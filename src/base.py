#!/usr/bin/env python
# coding: utf-8

# In[1]:
import time

import functools as F
import pandas as pd
import numpy as np
import warnings
from collections import Counter
from tqdm import tqdm
from sklearn.model_selection import RandomizedSearchCV


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.metrics import mean_squared_error
from typing import Tuple
from sklearn.utils import Bunch

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pymfe.mfe import MFE
from sklearn.base import ClassifierMixin, BaseEstimator

from skmultiflow.data import DataStream
import src.aux as A
import abc

class MStream(BaseEstimator, ClassifierMixin):
    def __init__(self, meta_model, base_models, base_tuners, is_incremental,
        train_extractor, horizon_extractor=lambda _: {}, test_extractor = lambda _: {},
        meta_retrain_interval=10, labelizer=A.smallest_labelizer, scorer=mean_squared_error, 
        incremental_trainer=None):
        super().__init__()
        
        self.base_models = base_models
        self.base_tuners = base_tuners
        self.is_incremental = is_incremental
        self.incremental_trainer = incremental_trainer
        self.meta_model = meta_model
        self.meta_retrain_interval = meta_retrain_interval # A cada quantos "updates" metamodelo será retreinado
        self.train_extractor = train_extractor
        self.horizon_extractor = horizon_extractor
        self.test_extractor = test_extractor
        self.labelizer = labelizer
        self.scorer = scorer
        self.meta_x = []
        self.meta_y = []

        self.next_x_train, self.next_x_horizon = [], []
        self.next_y_train, self.next_y_horizon = [], []
        # Used to make cache ou other specific tricks for optimization
        self.cached_metafeatures = {}
        self.current_stream = None
        
        self._update_counter = 0
        self._base_evals = []
        self._meta_window = 0
        self._processed = 0
        self._counter_labels = Counter()
        self._processed_online = 0


    @property
    def processed_online(self):
      return self._processed_online


    def _tune(self, X, Y, ):
      for tup, tuner in zip(self.base_models, self.base_tuners):
        tuner(model=tup.model, X=X, y=Y)


    def _meta_conditional_retrain(self, verbose=True):
      if self._update_counter % self.meta_retrain_interval == 0:
        if verbose:
          print("-------- Time to retrain --------! ", 
                self._update_counter, self.meta_retrain_interval, end='\n\n')
        if self.is_incremental:
          self.meta_model = self.incremental_trainer(
              model=self.meta_model,
              x=pd.DataFrame(self.meta_x[-self._meta_window:]),
              y=self.meta_y[-self._meta_window:],
          )
        else:
          self.meta_model.fit(
              pd.DataFrame(self.meta_x[-self._meta_window:]), 
              self.meta_y[-self._meta_window:],
          )          


    def reset(self):
      self.meta_x = []
      self.meta_y = []

      self.next_x_train, self.next_x_horizon = [], []
      self.next_y_train, self.next_y_horizon = [], []
      # Used to make cache ou other specific tricks for optimization
      self.cached_metafeatures = {}
      self.current_stream = None
      
      self._update_counter = 0
      self._base_evals = []
      self._meta_window = 0
      self._processed = 0
      self._counter_labels = Counter()
      self._processed_online = 0
          
    def fit(self, X, Y, meta_window=40, train=300, horizon=0, test=10, 
          metabase_initial_size=40, skip_tune=False, retrain_interval=None, verbose=False,
        ):
        self.reset()
        self._meta_window = meta_window
        self.meta_x, self.meta_y, self._base_evals = (
            [], [], [],
        )

        # 0 - assume-se que X e y são numpy arrays
        # 1 - Tune
        if not skip_tune:
          # datasize = train * metabase_initial_size // test
          datasize = metabase_initial_size * test + train
          if verbose: print("DATASIZE (opt):", datasize)
          self._tune(X[:datasize], Y[:datasize],)
          print("TUNE OK")
        else:
          print("SKIPPED TUNE")      
        
        # 2 - Generate initial metabase
        print("Gerando metabase inicial...")
        stream = DataStream(X, Y,  allow_nan=True, )
        self.meta_x, self.meta_y, self._base_evals, self._processed = self._gen_metabase(
              stream, metabase_initial_size, train=train, horizon=horizon, 
              test=test, verbose=verbose, 
              retrain_interval=retrain_interval,
        )
        self.current_stream = stream
        self.cached_metafeatures = A.prefixify(
          self.train_extractor(self.next_x_train, self.next_y_train), 'tr'
        )
        self.cached_metafeatures.update(A.prefixify(
          self.horizon_extractor(self.next_x_horizon), 'hor'
        ))

        # # Treinar modelos nos últimos dados de treinos disponíveis só
        # deve ser feito na etapa de atualizar a metabase
        
        print(f'finalizou após processar {self._processed} amostras')
        # Salva metacaracteristicas supervisionadas, às quais serão adicionadas medidas NÃO-supervisionadas
        # para que as predições sejam feitas.
        
        self.meta_model.fit(
          pd.DataFrame(self.meta_x[-meta_window:]), 
          pd.Series(self.meta_y[-meta_window:])
        )
        return self


    @abc.abstractmethod
    def _gen_metabase(self, stream: DataStream, metabase_initial_size, train, horizon, test=10,
        verbose=1, retrain_interval=None):
      """Função para geração de metabase para o MetaStream.

      Args:
          X ([type]): [description]
          Y ([type]): [description]
          metabase_initial_size ([type]): [description]
          train ([type]): [description]
          sel (int, optional): [description]. Defaults to 10.
          verbose (int, optional): [description]. Defaults to 1.
          retrain_interval (int, optional): [A cada `retrain_interval` amostras de teste, retreina os modelos-base. Note-se
            que, embora não presente no MetaStream original, pode fazer sentido. Via de regra porém, convém configurar
            `retrain_interval` = `test`].
            Defaults to 10.

      Returns:
          [type]: [description]
      """
      raise NotImplementedError(f"`_gen_metabase` not implemented by {type(self)}")


    @abc.abstractmethod
    def predict(self, X, X_horizon=[], verbose=False, to_cache=False):
        """Prediz qual melhor algoritmo atuar nos dados imediatamente vindouros.
        
        X: matriz com atributos a serem enviados para o extrator não-supervisionado
        """
        raise NotImplementedError(f"`predict` not implemented by {type(self)}")


    @abc.abstractmethod
    def update_stream(self, X, y, selection_window, verbose=False, use_cache=True):
      """Update internal object state.
      
      This method should be called right after "predict" when "metastream"
      and after several "predicts" when "micrometastream".

      Actually, this method exists to make "predict" idempotent 
      
      by removing its collateral effects. However, semantically it only makes sense
      to call `predict` followed by `update` stream BEFORE calling `predict` again.
      This method basically updates the metabase and possibly [/optionally]
      other internal attributes like evaluation history.
      """
      raise NotImplementedError(f"`update_stream` not implemented by {type(self)}")

