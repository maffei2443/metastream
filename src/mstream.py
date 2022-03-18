#!/usr/bin/env python
# coding: utf-8

import functools as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from skmultiflow.data import DataStream
from collections import Counter

import src.aux as A
import src.aux as A
from src import base
from importlib import reload

reload(A)
reload(base)


class MetaStream(base.MStream):
    def __init__(self, meta_model, base_models, base_tuners,
                 train_extractor, horizon_extractor=lambda _: {}, test_extractor=lambda _: {},
                 meta_retrain_interval=10, labelizer=A.smallest_labelizer, scorer=mean_squared_error, is_incremental=False, incremental_trainer=None):
        super().__init__(meta_model, base_models, base_tuners, is_incremental,
                         train_extractor, horizon_extractor, test_extractor,
                         meta_retrain_interval, labelizer, scorer, incremental_trainer,
                         )
        self.fit_dbg = []
        self.update_dbg = []
        self.base_models_retrain = 0
        self._processed_online = 0
        self._cache_predict_multi = []

    def _gen_meta_label(self, stream: DataStream, metabase_initial_size, train, horizon, test=10,
                        verbose=True, retrain_interval=None, **kwargs):
        """Função para rotulação de "metabase" para o MetaStream.

        Args:
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
        if retrain_interval is None:
            retrain_interval = test

        # Para simplificar o código, retrain_interval mod sel = 0 e, claro, retrain_interval > sel
        assert not retrain_interval % test and retrain_interval >= test

        retrain_interval_original = retrain_interval
        retrain_interval = 0  # Para treinar modelo na primeira vez

        meta_y = []
        eval_history = []
        offset = 0
        it = tqdm(range(metabase_initial_size)) if verbose else range(
            metabase_initial_size)

        x_train, y_train = (i.tolist() for i in stream.next_sample(train))
        x_hor, y_hor = (i.tolist() for i in stream.next_sample(horizon))
        for _ in it:
            x_test, y_test = (i.tolist() for i in stream.next_sample(test))
            offset += test
            if retrain_interval <= 0:
                retrain_interval = retrain_interval_original
                for tup in self.base_models:
                    tup.model.fit(x_train, y_train)

            retrain_interval -= test
            evals = {}
            for tup in self.base_models:
                y_pred = tup.model.predict(x_test)
                evals[tup.name] = self.scorer(y_true=y_test, y_pred=y_pred)

            eval_history.append(evals)
            # 2.1 - Rotula metaexemplo
            y_best_label, y_best_val = self.labelizer(evals, dist=self._counter_labels)
            self._counter_labels[y_best_label] += 1
            meta_y.append(y_best_label)

            x = [*x_train, *x_hor, *x_test][test:]
            y = [*y_train, *y_hor, *y_test][test:]

            x_train, x_hor = x[:train], x[train:+train+horizon]
            y_train, y_hor = y[:train], y[train:+train+horizon]

        return meta_y, eval_history

    def _gen_metabase(self, stream: DataStream, metabase_initial_size, train, horizon, test=10,
                      verbose=True, retrain_interval=None, labelize=True, **kwargs):
        s0 = stream.sample_idx
        # To count in the end how many samples were processed (used somehow)

        if retrain_interval is None:
            retrain_interval = test

        # Para simplificar o código, retrain_interval mod sel = 0 e, claro, retrain_interval > sel
        assert not retrain_interval % test and retrain_interval >= test

        retrain_interval_original = retrain_interval
        retrain_interval = 0  # Para treinar modelo na primeira vez

        meta_x = []
        meta_y = []
        eval_history = []

        it = tqdm(range(metabase_initial_size)) if verbose \
            else range(metabase_initial_size)
        dbg = []

        x_train, y_train = (i.tolist() for i in stream.next_sample(train))
        x_hor, y_hor = (i.tolist() for i in stream.next_sample(horizon))
        for _ in it:
            if not stream.has_more_samples():
              print("WARNING: consumed all stream")
              break
            x_test, y_test = (i.tolist() for i in stream.next_sample(test))
            dbg.append((x_train, x_hor, x_test, y_train, y_hor, y_test))

            meta_features: dict = A.prefixify(
                self.train_extractor(X=x_train, y=y_train), 'tr')
            meta_features.update(A.prefixify(
                self.horizon_extractor(x_hor), 'hor'))
            meta_features.update(A.prefixify(
                self.test_extractor(x_test), 'tes'))
            meta_x.append(meta_features.copy())

            if labelize:
                if retrain_interval <= 0:
                    retrain_interval = retrain_interval_original
                    for tup in self.base_models:
                        tup.model.fit(x_train, y_train)
                    self.base_models_retrain += 1

                retrain_interval -= test

                evals = {}
                for tup in self.base_models:
                    y_pred = tup.model.predict(x_test)
                    evals[tup.name] = self.scorer(y_true=y_test, y_pred=y_pred)

                eval_history.append(evals)
                # 2.1 - Rotula metaexemplo
                y_best_label, y_best_val = self.labelizer(evals, dist=self._counter_labels)
                self._counter_labels[y_best_label] += 1
                meta_y.append(y_best_label)

            x = [*x_train, *x_hor, *x_test][test:]
            y = [*y_train, *y_hor, *y_test][test:]

            x_train, x_hor = x[:train], x[train:]
            y_train, y_hor = y[:train], y[train:]

        # Ultimos exemplos-base serao posteriormente usados para induzir modelos-base
        # para rotular corretamente o metaexemplo que será gerado.
        if not labelize:
            return meta_x, np.cumsum([train, horizon, test]), stream.sample_idx - s0, dbg

        self.fit_dbg = dbg
        self.next_x_train, self.next_y_train = x_train, y_train
        self.next_x_horizon, self.next_y_horizon = x_hor, y_hor
        return meta_x, meta_y, eval_history, stream.sample_idx - s0


    def predict(self, X, sel=None, to_cache=True, verbose=False,):
      if verbose: print('Single' if not sel else f"Multi with sel={sel}")
      if not sel:
        sel = len(X)
      if to_cache:
        self._cache_predict_multi.clear()

      X_chunks = [X[i:i+sel] for i in range(0, len(X), sel)]
      if verbose: print(f'n_chunks: {len(X_chunks)}')
      mtf_lis = []
      for x in X_chunks:
        meta_features: dict = self.cached_metafeatures.copy()
        xtest_fts = A.prefixify(self.test_extractor(x), 'tes')
        # Nao carece guardar features de treino e horizon 
        # pois são compartilhadas
        if to_cache:
          self._cache_predict_multi.append(xtest_fts)

        meta_features.update(xtest_fts)        
        if verbose:  print('meta_features: ', meta_features)
        mtf_lis.append(meta_features)
      return self.meta_model.predict(pd.DataFrame(mtf_lis))


    def _compute_mtf_and_eval(self, X, y, use_cache=True, ):
        meta_features = self.cached_metafeatures.copy()
        if not use_cache:
          meta_features.update(A.prefixify(self.test_extractor(X), 'tes'))
        else:
          # CONSOME cache
          meta_features.update(self._cache_predict_multi.pop(0))
          
        evals = {}
        for tup in self.base_models:
            y_pred = tup.model.predict(X)
            evals[tup.name] = self.scorer(y_true=y, y_pred=y_pred, )
        return meta_features, evals
      

    def update_stream(self, X, y, sel=None, use_cache=True, verbose=False, base_retrain=False):
        """Update internal object state.

        This method should be called right after "predict" when "metastream"
        and after several "predicts" when "micrometastream".

        Actually, this method exists to make "predict" idempotent 

        by removing its collateral effects. However, semantically it only makes sense
        to call `predict` followed by `update` stream BEFORE calling `predict` again.
        This method basically updates the metabase and possibly [/optionally]
        other internal attributes like evaluation history.
        """
        if verbose: print('Multi update' if sel else 'Single update')

        if not sel:
          sel = len(X)
        lis_x = [X[i:i+sel] for i in range(0, len(X), sel)]
        lis_y = [y[i:i+sel] for i in range(0, len(y), sel)]
        # Carrega features [supervisionadas/do horizon] extraídas no último predict
        if base_retrain:
          if verbose:  print(f"[update_stream] Treinamento base")
          for tup in self.base_models:
            tup.model.fit(self.next_x_train, self.next_y_train)
          self.base_models_retrain += 1

        mtf_and_evals_lis = [
          self._compute_mtf_and_eval(xx, yy, use_cache=use_cache, )
          for (xx, yy) in zip(lis_x, lis_y)
        ]

        # Contador de atualização dos modelos-base
        meta_features_lis, evals_lis  = list(zip(*mtf_and_evals_lis))
        self._base_evals.extend(evals_lis)
        if verbose:        print('evals_lis:', *evals_lis, sep='\n  ')

        labeled_x = [*self.next_x_train, *self.next_x_horizon, *X]
        labeled_y = [*self.next_y_train, *self.next_y_horizon, *y]

        thresh_train_hor = len(X) + len(self.next_x_train)
        new_train_slice = slice(len(X), thresh_train_hor)
        self.next_x_train = labeled_x[new_train_slice]
        self.next_y_train = labeled_y[new_train_slice]

        new_hor_slice = slice(
            thresh_train_hor, thresh_train_hor + len(self.next_x_horizon))
        self.next_x_horizon = labeled_x[new_hor_slice]
        self.next_y_horizon = labeled_y[new_hor_slice]

        best_label_lis, best_eval_lis = list(zip(*
            [self.labelizer(evals, dist=self._counter_labels) for evals in evals_lis]
        ))
        # y_best_label, y_best_val = self.labelizer(evals_lis, dist=self._counter_labels)
        self._counter_labels += Counter(best_label_lis)

        # Update internal state
        self.meta_x.extend(meta_features_lis)
        self.meta_y.extend(best_label_lis)

        # Novo metaexemplo criado
        self._processed_online += len(X)
        self._update_counter += len(lis_x)
        self._meta_conditional_retrain(verbose)

        self.cached_metafeatures = A.prefixify(
            self.train_extractor(self.next_x_train, self.next_y_train), 'tr',
        )
        self.cached_metafeatures.update(A.prefixify(
            self.horizon_extractor(self.next_x_horizon), 'hor'
        ))

