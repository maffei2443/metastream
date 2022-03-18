# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))

import numpy as np
import pandas as pd
import scipy.stats
from pathlib import Path
np.random.seed(42)

# Métricas e preprocessamento
from sklearn.metrics import zero_one_loss
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Gerais
import functools as F
import itertools as I
from importlib import reload as rl
from sklearn.utils import Bunch
from tqdm import tqdm
import copy

# Locais
import src.models as M
import src.blast as B
import src.aux as A

def lc(*x):
    return list(I.chain(*x))

def Generator(lis):
    for m in lis:
        yield m
# https://stackoverflow.com/questions/4971269/
# how-to-pick-a-new-color-for-each-plotted-line-within-a-figure-in-matplotlib
from matplotlib.cm import get_cmap
n = "Accent"
cmap = get_cmap(n)  # type: matplotlib.colors.ListedColormap
colors = cmap.colors  # type: list


# In[2]:


import argparse
import os
import shutil
import tempfile
import joblib
import mlflow
import functools as F
from importlib import reload as rl

import pandas as pd
import numpy as np
import scipy.stats


from pathlib import Path
from sklearn.metrics import accuracy_score

from sklearn.utils import Bunch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

import src.models as M
import src.aux as A
np.random.seed(42)


PATH = Path(tempfile.mkdtemp())
os.makedirs(PATH/'png')
os.makedirs(PATH/'csv')
os.makedirs(PATH/'joblib')

par = argparse.ArgumentParser()
par.add_argument('--base', type=str, help='Database to use', default='elec2')
par.add_argument('--nrows', type=int, help='How many samples will be used at most', default=30_000)
par.add_argument('--train', type=int, help='Size of train set', default=300)
par.add_argument('--test', type=int, help='Size of test window', default=10)
par.add_argument('--alpha', type=int, help='BLAST alpha param', default=10)
par.add_argument('--omega_base', type=int, help='How many samples til retrain base', default=10)
# par.add_argument('--metric', help='Metric to use on base models')
par.add_argument('--metabase_initial_size', type=int, help='Size of initial metabase', default=410)
par.add_argument('--online_size', type=int, help='How many metaexamples to test (online phase)', default=100)
par.add_argument('--supress_warning', type=int, help='Whether to supress warnings', default=1)
par.add_argument('--tune', type=int, help='Whether or not to fine tune base models', default=1)
par.add_argument('--default_draw_offline', type=str, 
    help='Which base model to be used as label when there is a draw', default='NysSvm')


args, rest = par.parse_known_args()
params = Bunch(**args.__dict__)
print(*params.items(), sep='\n')
if args.supress_warning:
  A.IGNORE_WARNING()
  del args.supress_warning


BASE=Path('csv')
EXPERIMENT_NAME = f'{args.base}_blast'
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if not exp:
    print(f"Criando experimento {EXPERIMENT_NAME} pela primeira vez")
    experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME)
else:
    experiment_id = exp.experiment_id
run = mlflow.start_run(experiment_id=experiment_id)
mlflow.log_params(args.__dict__)
mlflow.set_tag('strategy', 'blast')
print("Running experiment:", EXPERIMENT_NAME)
print(f'  {run.info.run_id}')


MODELS=[
    'NysSvm', 
    'Rf',
]
TRAIN=args.train
TEST=args.test
METASTREAM_INITIAL_METABASE = args.metabase_initial_size
METRIC='acc'
META_METRICS=['acc', 'kappa_custom', 'geometric_mean']
META_METRICS_FUN = [M.METRICS_CLF[met] for met in META_METRICS]
HP_GRID_LIS = [
    {"svm__C": [1,10,100],
     "nys__kernel": ['poly', 'rbf', 'sigmoid']
     },
    {
        "max_depth": [3, 5, None],
        "n_estimators": [100, 200, 300],
        "min_samples_split": scipy.stats.randint(2,11),
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]
    },
]
mlflow.set_tag('models', MODELS)
# In[40]:

print("reading...")
df = pd.read_csv( BASE / f'{args.base}.csv', nrows=args.nrows)
print('read')
X, y = df.iloc[:, 1:-1].fillna(0), df.iloc[:, -1]
lbe = LabelEncoder()
yt = lbe.fit_transform(y)


models = [
    Bunch(name=n, model=M.CLF[n]()) 
    for n in MODELS
]
opt_params = A.random_params.copy()
opt_params['cv'] = TEST
base_tuners = [
    F.partial(A.random_tuner, params=hp_grid,
            opt_params=opt_params)
    for hp_grid in HP_GRID_LIS
]


OPT_SIZE = TRAIN + METASTREAM_INITIAL_METABASE * TEST
if args.tune:
  for m, tuner in zip(models, base_tuners):
      tuner(model=m.model, X=X[:OPT_SIZE], y=y[:OPT_SIZE])
else:
  print("no tunes")


def eval_blast(blast: B.BLAST, X, y, alpha, omega, xidx, yidx, verbose=False):
    if verbose:
        print(f"ATUALIZA MODELOS-BASE A CADA INTERVALO DE {omega} PREDIÇÕES-base")
        print(f"BLAST A CADA {alpha} amostras")
    lt_alpha = 0
    lt_omega = 0

    preds_his = []
    scores_his = []
    blast_his = []
    x_buff = []
    y_buff = []
    for idx, (train_idx, test_idx) in tqdm(enumerate(zip(xidx, yidx))):
        xtest, ytest = X.iloc[test_idx], y[test_idx]
        # Para atualizar o BLAST no momento desejado
        x_buff.extend(xtest.values)
        y_buff.extend(ytest)

        lt_omega += len(ytest) # Mais TEST processados
        lt_alpha += len(ytest) # Mais alpha predicoes (1 predicao por dado)

        blast_pred = blast.predict()
        blast_his.append(blast.base_models[blast_pred].name)

        preds = {
            m.name: m.model.predict(xtest)
            for m in models
        }
        scores = {
            m: accuracy_score(ytest, v)
            for (m,v) in preds.items()
        }
        preds['true'] = ytest
        preds_his.append(preds)
        scores_his.append(scores)

        #     Retreinar modelos e o BLAST
        if lt_alpha >= alpha or lt_omega >= omega:
            #  Atualização do BLAST (OPE
            if lt_alpha >= alpha:
                if verbose: print("BLAST update", idx)
                lt_alpha = 0
                # Após predizer para `alpha` amostras, atualize-se o BLAST
                res = blast.update(x_buff, y_buff)
                x_buff, y_buff = [], []
                if res != blast:
                    print("Exception occured")
                    return res

            # Retreinar modelos base aqui para não acontecer de treinar e
            # testar nos mesmos dados em nível base no momento de atualizar o BLAST
            if lt_omega >= omega:
                if verbose: print("  BASE models update", idx)
                xtrain, ytrain = X.iloc[train_idx], y[train_idx]
                for m in models:
                    m.model.fit(xtrain, ytrain,)
                lt_omega = 0

    return preds_his, scores_his, blast_his


rl(B)
blast = B.BLAST(
    base_models=models, 
    loss_function=zero_one_loss, 
)

xidx, yidx = A.TimeSeriesCVWindows(
    args.metabase_initial_size + args.online_size, TRAIN, TEST, gap=0, step=TEST
)
off_xidx, on_xidx = np.split(xidx, [args.metabase_initial_size])
off_yidx, on_yidx = np.split(yidx, [args.metabase_initial_size])


# Primeiro teste para ter de onde começar: treina em 300,
# tamanho da metajanela de treinamento, e testa no próximos
# 300 também.
train_idx = off_xidx[0]
for m in models:
    m.model.fit(X.iloc[train_idx], y[train_idx])
blast.fit(X.iloc[train_idx], y[train_idx]);

ph_off, sh_off, bh_off = eval_blast(
    blast, X, y, alpha=args.alpha, omega=args.omega_base, 
    xidx=off_xidx, yidx=off_yidx, verbose=False
)
ph_on, sh_on, bh_on = eval_blast(
    blast, X, y, alpha=args.alpha, omega=args.omega_base, 
    xidx=on_xidx, yidx=on_yidx, verbose=False
)


df_offline_scores = pd.DataFrame(sh_off)
meta_predicts = bh_off.copy()
df_offline_scores.to_csv(PATH/'csv'/'df_offline_scores.csv',)
joblib.dump(meta_predicts, PATH/'joblib'/'meta_predicts_offline.joblib');


df_online_scores = pd.DataFrame(sh_on)
meta_predicts = bh_on.copy()
df_online_scores.to_csv(PATH/'csv'/'df_online_scores.csv', )
joblib.dump(meta_predicts, PATH/'joblib'/'meta_predicts.joblib');


mlflow.log_artifacts(PATH)
mlflow.end_run()
shutil.rmtree(PATH)
