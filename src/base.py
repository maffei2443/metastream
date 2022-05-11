import pandas as pd
from collections import Counter

from sklearn.metrics import mean_squared_error
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
          
    def fit(self, X, Y, meta_window=40, train=300, horizon=0, test=10, gamma=0,
          metabase_initial_size=40, skip_tune=False, retrain_interval=None, verbose=False,
        ):
        self.reset()
        self._meta_window = meta_window
        self.meta_x, self.meta_y, self._base_evals = (
            [], [], [],
        )

        # 0 - X and y assumed to be numpy arrays
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
        if verbose: print("Genrating initial metabase...")
        stream = DataStream(X, Y,  allow_nan=True, )
        self.meta_x, self.meta_y, self._base_evals, self._processed = self._gen_metabase(
              stream, metabase_initial_size, train=train, horizon=horizon, 
              test=test, gamma=gamma, verbose=verbose, 
              retrain_interval=retrain_interval,
        )
        self.current_stream = stream
        self.cached_metafeatures = A.prefixify(
          self.train_extractor(self.next_x_train, self.next_y_train), 'tr'
        )
        self.cached_metafeatures.update(A.prefixify(
          self.horizon_extractor(self.next_x_horizon), 'hor'
        ))
        if verbose: print('Training models on most updated data...')
        for tup in self.base_models:
          tup.model.fit(self.next_x_train, self.next_y_train)

        print(f'Finished processing {self._processed} samples')

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
        """Predicts best best algorithm X.

        X: attribute matrix to be used to unsupervised feature extraction
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

