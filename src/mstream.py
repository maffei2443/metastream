import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from skmultiflow.data import DataStream
from collections import Counter
from typing import List

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

    def _handle_predictions(self, list_predictions: List[dict], list_ytrue: List[List]):
      df_lis = [pd.DataFrame(scores) for scores in list_predictions]
      lis_scores_dict = [
        df.apply(lambda y_pred: self.scorer(y_true=y_true, y_pred=y_pred)).to_dict()
        for (df, y_true) in zip(df_lis, list_ytrue)
      ]
      lis_labels_performance = [
        self.labelizer(dici, dist=self._counter_labels)
        for dici in lis_scores_dict
      ]
      return lis_labels_performance, lis_scores_dict


    def _gen_metabase(self, stream: DataStream, metabase_initial_size, train, horizon, test=10,
                      gamma=0, verbose=True, retrain_interval=None, labelize=True, **kwargs):
        if not gamma:
          gamma = test
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
            x_test, y_test = stream.next_sample(test)
            x_test = x_test.tolist()
            
            # x_test, y_test = (i.tolist() for i in stream.next_sample(test))
            dbg.append((x_train, x_hor, x_test, y_train, y_hor, y_test))

            mtf_lis = []
            test_chunks = [x_test[i:i+gamma] for i in range(0, len(x_test), gamma)]

            base_mtf: dict = A.prefixify(
                self.train_extractor(X=x_train, y=y_train), 'tr')
            base_mtf.update(A.prefixify(
                self.horizon_extractor(x_hor), 'hor'))
            for chunk in test_chunks:
              meta_features = base_mtf.copy()
              meta_features.update((
                A.prefixify(self.test_extractor(chunk), 'tes')
              ))
              mtf_lis.append(meta_features)
            meta_x.extend(mtf_lis)

            if labelize:
                if retrain_interval <= 0:
                    retrain_interval = retrain_interval_original
                    for tup in self.base_models:
                        tup.model.fit(x_train, y_train)
                    self.base_models_retrain += 1

                retrain_interval -= test

                slices_idx, _ = A.SlidingWindow(n=int (np.ceil(test/gamma)), train=gamma, 
                                  test=0, gap=0, step=gamma)
                slices_idx = slices_idx.tolist()
                # Not exact division requires cutting extra indices
                if test % gamma:
                  slices_idx[-1] = slices_idx[-1][:test%gamma]
                predictions = [tup.model.predict(x_test) for tup in self.base_models ]
                predictions = [
                  {
                    tup.name: pred[slc]
                    for (tup, pred) in zip(self.base_models, predictions) }
                  for slc in slices_idx
                ]
                correct_lis = [y_test[slc] for slc in slices_idx]                
                eval_best_lis, eval_lis = self._handle_predictions(predictions, correct_lis, )                          
                eval_history.extend(eval_lis)
                y_best_lis, y_score_lis = A.lzip(*eval_best_lis)
                meta_y.extend(y_best_lis)

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


    def predict(self, X, gamma=None, to_cache=True, verbose=False,):
      if verbose: print('Single' if not gamma else f"Multi with gamma={gamma}")
      if not gamma:
        gamma = len(X)
      if to_cache:
        self._cache_predict_multi.clear()

      X_chunks = [X[i:i+gamma] for i in range(0, len(X), gamma)]
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
      

    def update_stream(self, X, y, gamma=None, use_cache=True, verbose=False, ):
        """Update internal object state.

        This method should be called right after "predict" when "metastream"
        and after several "predicts" when "micrometastream".

        Actually, this method exists to make "predict" idempotent 

        by removing its collateral effects. However, semantically it only makes sense
        to call `predict` followed by `update` stream BEFORE calling `predict` again.
        This method basically updates the metabase and possibly [/optionally]
        other internal attributes like evaluation history.
        """
        if verbose: print('Multi update' if gamma else 'Single update')

        if not gamma:
          gamma = len(X)
        lis_x = [X[i:i+gamma] for i in range(0, len(X), gamma)]
        lis_y = [y[i:i+gamma] for i in range(0, len(y), gamma)]
        # Carrega features [supervisionadas/do horizon] extraídas no último predict

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

