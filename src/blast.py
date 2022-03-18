from importlib import reload as rl
import numpy as np
from sklearn.base import ClassifierMixin, TransformerMixin

import src.ope as OPE
rl(OPE)

class BLAST( ClassifierMixin, TransformerMixin):
    def __init__(self, base_models, loss_function, ):
        self.base_models = base_models
        self.loss_fn = loss_function
        self.ope = OPE.OnlinePerformanceEstimation(
            base_models=base_models,
            loss_function=loss_function,
        )


    # @A.RuntimeWrapper
    def fit(self, X, y, ):
        self.ope.fit(X, y)
        return self
    
    def update(self, X, y, ):
        try:
          res = self.ope.update(X, y, )
          if res:
            raise Exception("Exception underground") 
          return self
        except Exception as e:
          print(e)
          return res

    # @A.RuntimeWrapper
    def predict(self):
        return np.argmin(self.ope.predict())

    def __repr__(self) -> str:
        return f"BLAST(base_models = {self.base_models}, loss_function={self.loss_fn})"
