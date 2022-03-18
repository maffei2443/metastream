# from IPython.core.display import display, HTML
# display(HTML("<style>.container { width:100% !important; }</style>"))

_ = None
import argparse
import json as J
import os
import shutil
import tempfile
import joblib
import mlflow
import functools as F
from importlib import reload as rl
import copy


import pandas as pd
import numpy as np
import scipy.stats

import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter as C
from sklearn.metrics import accuracy_score
from pylab import ma, cm

from sklearn.utils import Bunch
from sklearn.preprocessing import LabelEncoder
import lightgbm
from tqdm import tqdm
from pymfe.mfe import MFE

import src.models as M
import src.mstream as MS
import src.aux as A
np.random.seed(42)


# https://stackoverflow.com/questions/4971269/<br><br>



from matplotlib.cm import get_cmap
n = "Accent"
cmap = get_cmap(n)  # type: matplotlib.colors.ListedColormap
colors = cmap.colors  # type: list


PATH = Path(tempfile.mkdtemp())
os.makedirs(PATH/'png')
os.makedirs(PATH/'csv')
os.makedirs(PATH/'joblib')



par = argparse.ArgumentParser()
par.add_argument('--base', type=str, help='Database to use', default='elec2')
par.add_argument('--nrows', type=int, help='How many samples will be used at most', default=30_000)
par.add_argument('--train', type=int, help='Size of train set', default=300)
par.add_argument('--horizon', type=int, help='Size of horizon set', default=0)
par.add_argument('--test', type=int, help='Size of test window', default=10)
# par.add_argument('--metric', help='Metric to use on base models')
par.add_argument('--metabase_initial_size', type=int, help='Size of initial metabase', default=410)
par.add_argument('--online_size', type=int, help='How many metaexamples to test (online phase)', default=100)
par.add_argument('--offline_size', type=int, help='How many metaexamples to test (online phase)', default=100)
par.add_argument('--meta_retrain_interval', type=int, help='How many new metaexample til retrain', default=1)
par.add_argument('--base_retrain_interval', type=int, help='How many new base examples til retrain', default=10)
par.add_argument('--meta_train_window', type=int, help='How many metaexamples to train on', default=300)
par.add_argument('--gamma', type=int, 
                 help='Batch size. Zero means to predict one algorithm to whole window', default=0)
par.add_argument('--is_incremental', type=int, help='To use or not the incremental metamodel', default=0)
par.add_argument('--reverse_models', type=int, help='To use or not reverse models order', default=0)
par.add_argument('--supress_warning', type=int, help='Whether to supress warnings', default=1)
par.add_argument('--choice', type=str, help='Which model will have preference when Tie happens', default='NysSvm')
par.add_argument('--tune', type=int, help='Whether or not to fine tune base models', default=1)





args, rest = par.parse_known_args()
params = Bunch(**args.__dict__)
print(*params.items(), sep='\n')
if args.supress_warning:
  A.IGNORE_WARNING()
  del args.supress_warning



# args.online_size = 2000
# args.meta_retrain_interval = 1
# args.is_incremental = 0
labelizer = F.partial(A.biggest_labelizer_arbitrary, choice=args.choice)
joblib.dump(labelizer, PATH/'joblib'/'labelizer.joblib')



BASE=Path('csv')
# mapa = {ex.name: ex.experiment_id for ex in mlflow.list_experiments()}
EXPERIMENT_NAME = f'{args.base}_meta'
exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

if not exp:
    print(f"Criando experimento {EXPERIMENT_NAME} pela primeira vez")
    experiment_id = mlflow.create_experiment(name=EXPERIMENT_NAME)
else:
    experiment_id = exp.experiment_id
run = mlflow.start_run(experiment_id=experiment_id)
mlflow.log_params(args.__dict__)



META_MODEL='LgbCustomSkWrapper'
MODELS=[
    'NysSvm', 
    'Rf',
]
mlflow.set_tag('meta_model', META_MODEL)
METRIC='acc'
META_METRICS=['acc', 'kappa_custom', 'geometric_mean']
META_RETRAIN_INTERVAL=args.meta_retrain_interval
MT_TRAIN_FEATURES = [
  "best_node","elite_nn","linear_discr",
            "naive_bayes","one_nn","random_node","worst_node",
            "can_cor","cor", "cov","g_mean",
            "gravity","h_mean","iq_range","kurtosis",
            "lh_trace",
            "mad",
            "max","mean",
            "median",
            "min",
            "nr_cor_attr","nr_disc","nr_norm","nr_outliers",
            "p_trace","range","roy_root","sd","sd_ratio",
            "skewness","sparsity","t_mean","var","w_lambda"
            ]
MT_HOR_FEATURES = []
MT_TEST_FEATURES = []
HP_GRID_LIS = [
    {"svm__C": [1,10,100],
     "nys__kernel": ['poly', 'rbf', 'sigmoid']
     }, 
    {   "max_depth": [3, 5, None],
        "n_estimators": [100, 200, 300],
        "min_samples_split": scipy.stats.randint(2, 11),
        "bootstrap": [True, False],
        "criterion": ["gini", "entropy"]
    }
]
HP_META_MODEL = {
    'boosting_type': 'dart',
    'learning_rate': 0.01,
    'tree_learner': 'feature',
    'metric': 'multi_error,multi_logloss',
    'objective': 'multiclassova',
    'num_class': len(MODELS),
    'is_unbalance': True,
    'verbose': -1,
    'seed': 42,
}



if args.reverse_models:
  print("reversing..")
  MODELS = MODELS[::-1]
  HP_GRID_LIS = HP_GRID_LIS[::-1]



mlflow.set_tag('models', MODELS)
mlflow.set_tag('strategy', 'incremental' if args.is_incremental else 'nao-incremental')
mlflow.set_tag('meta-retreinamento', args.meta_retrain_interval)



joblib.dump(HP_META_MODEL, PATH/'joblib'/'hp_meta.joblib')
mlflow.log_params(A.prefixify(HP_META_MODEL, 'metaHp'))





df = pd.read_csv( BASE / f'{args.base}.csv', nrows=args.nrows)
X, y = df.iloc[:, 1:-1].fillna(0), df.iloc[:, -1]
lbe = LabelEncoder()
yt = lbe.fit_transform(y)



# runinfo_lis = mlflow.list_run_infos(EXPERIMENT_ID)
# df_lis = []
# for rinfo in runinfo_lis:
#     try:
#         df_lis.append(pd.read_csv(
#             'mlruns/{}/{}/artifacts/metabase.csv'.format(
#                 EXPERIMENT_ID, rinfo.run_id),
#             index_col=False)
#         )
#     except:
#         pass
# df_cache = pd.concat(df_lis, axis=1)



# class CacheMtF:
#     def extractor(self, df_cache, prefix='tr'):
#         test_cols = [i for i in df_cache.columns 
#                      if i.startswith(prefix)]
#         df = df_cache[test_cols]
#         df = df.rename(lambda x: '_'.join(x.split('_')[1:]), axis=1)
#         for mtf in df.apply(lambda x: x.to_dict(), axis=1):
#             yield mtf    
#     def __init__(self, df_cache, prefix):
#         self.generator = self.extractor(df_cache, prefix)
#     def __call__(self, *args, **kwargs):
#         return next(self.generator)
# train_extractor = CacheMtF(df_cache, 'tr')
# test_extractor = CacheMtF(df_cache, 'tes')


rl(M)
rl(A)


train_extractor = F.partial(A.su_extractor, ext=MFE(
    features=MT_TRAIN_FEATURES,
    random_state=42,
))
horizon_extractor = lambda x: {}
test_extractor = lambda x: {}
meta_model = M.CLF[META_MODEL](
    fit_params=HP_META_MODEL,
    classes=[m for m in MODELS],
)
models = [
    Bunch(name=n, model=M.CLF[n]()) 
    for n in MODELS
]
opt_params = A.random_params.copy()
opt_params['cv'] = args.test



def fun(model, x, y, retrain_window = META_RETRAIN_INTERVAL):
    x = x[-retrain_window:]
    y = y[-retrain_window:]
    model.fit(x, y, incremental=True)
    return model
incremental_trainer = fun if args.is_incremental else None


if args.tune:
  print("SOME TUNING...")
  optmize_data = args.metabase_initial_size * args.test + args.train
  for m, hp in zip(models, HP_GRID_LIS):
      A.random_tuner(
          model=m.model, 
          params=hp, 
          opt_params=opt_params,
          X=X[:optmize_data], y=yt[:optmize_data],
      )
else:
  print("NO TUNING AT ALL")  



for m in models:
  mlflow.sklearn.log_model(m.model, m.name)
# 
# - Nota: faz sentido rodar uma vez com tudo e, depois, só carregar isso (ocupa espaço, poupa tempo)



METABASE_INITIAL_SIZE=args.metabase_initial_size
init_params = dict(
    meta_model=meta_model,
    base_models=models,
    base_tuners=[],
    train_extractor=train_extractor,
    horizon_extractor=horizon_extractor,
    test_extractor=test_extractor,
    labelizer=labelizer,
    scorer=A.accuracy_score,
    meta_retrain_interval=META_RETRAIN_INTERVAL,
    is_incremental=args.is_incremental,
    incremental_trainer=incremental_trainer,  # POHA PARA DE SER BIZONHO
)
fit_params = dict(
    X=X,
    Y=yt,
    meta_window=args.meta_train_window,
    train=args.train,
    horizon=args.horizon,
    test=args.test,
    metabase_initial_size=METABASE_INITIAL_SIZE,
)



rl(MS)
FT_HISTORY = []
ms = MS.MetaStream(**init_params)
ms.fit(**fit_params, verbose=True, skip_tune=True);
FT_HISTORY.append(meta_model.lgb.feature_importance())
# In[31]:

# Backup para poder recomeçar a fase online sem gerar a metabase novamente

meta_x = ms.meta_x.copy()
meta_y = ms.meta_y.copy()
nxtr, nytr = ms.next_x_train.copy(), ms.next_y_train.copy()
nxhr, nyhr = ms.next_y_horizon.copy(), ms.next_y_horizon.copy()
cached_metafeatures = ms.cached_metafeatures.copy()
base_evals = ms._base_evals.copy()
stream = copy.deepcopy(ms.current_stream)
counter_labels = copy.deepcopy(ms._counter_labels)



# # PAra testar com implementação do 
# rl(MS)
# FT_HISTORY = []
# ms2 = MS.MetaStream(**init_params)
# ms2.fit(**fit_params, verbose=True, skip_tune=True);
# # FT_HISTORY.append(meta_model.lgb.feature_importance())



rl(M)
rl(A)
mmetrics_fun = [M.METRICS_CLF[met] for met in META_METRICS]
off_meta_eval = []
off_preds = []
off_targets = []
print("FASE OFFLINE")
mm = M.CLF[META_MODEL](
    fit_params=HP_META_MODEL,
    classes=[m for m in MODELS],
)
train_idx_lis, test_idx_lis = A.TimeSeriesCVWindows(
  n=args.offline_size, train=args.train, test=args.test
)
df_meta_x = pd.DataFrame(ms.meta_x)
fnames = df_meta_x.columns
meta_x_off = df_meta_x.values
meta_y_off =pd.Series(ms.meta_y).values



for (train_idx, test_idx) in tqdm(zip(train_idx_lis, test_idx_lis)):
    xtrain, ytrain = meta_x_off[train_idx], meta_y_off[train_idx]
    xtest, ytest = meta_x_off[test_idx], meta_y_off[test_idx]
    
    mm.fit(pd.DataFrame(xtrain, columns=fnames), ytrain)
    predictions = mm.predict(xtest)
    off_preds.append(predictions)
    off_targets.append(ytest)
    off_meta_eval.append(
        [m(y_true=ytest, 
           y_pred=mm.label_encoder.inverse_transform(predictions))
         for m in mmetrics_fun]
    )


del fnames, df_meta_x, meta_x_off, meta_y_off, mm
print("FIM FASE OFFLINE")



print("gamma:", args.gamma)



FT_HISTORY = []
    
lis = []
true_lis = []
online_size = args.online_size
predict_lis = []
processed = ms._processed
print("INÍCIO FASE ONLINE")
for i in tqdm(range(1, online_size+1)):
    if not ms.current_stream.has_more_samples():
        print(f"Acabaram os dados no índice {i}")
        break
    xtest, ytest = (
        i.tolist() for i in ms.current_stream.next_sample(args.test)
    )
    # Predição (nível meta)
    pred=ms.predict(xtest, sel=args.gamma)
    lis.extend(pred)
    pre_dict = {
        'true': np.array(ytest), 
    }
    
    # Predição nível base
    for m in models:
        pre_dict[m.name] = m.model.predict(
            xtest
        )
    predict_lis.append(pre_dict)
    try:        
        ms.update_stream(
            xtest, 
            ytest,
            sel=args.gamma,
            base_retrain=True,
#             verbose=True,
        )
        true_lis.append(ms.meta_y[-1])
    except Exception as e:
        print("Acabaram-se os generators")
        raise e
        break
    FT_HISTORY.append(meta_model.lgb.feature_importance())



df_fti = pd.DataFrame(FT_HISTORY, columns=ms.meta_model.lgb.feature_name())
df_fti.to_csv(PATH/'csv'/'df_fti.csv', index=False,)
# Motivo: lgbm trabalha com números, então pegamos o nome
# do modelo usando o transformação inversa
lis = ms.meta_model.label_encoder.inverse_transform(
    lis
)
joblib.dump(lis, PATH/'joblib'/'meta_predicts.joblib')
joblib.dump(ms.meta_y, PATH/'joblib'/'meta_labels.joblib')
joblib.dump(ms._base_evals, PATH/'joblib'/'base_evals.joblib')


print("FIM FASE ONLINE")
print("DAQUI PARA BAIXO SÃO APENAS DUMPS E PLOTS")


df_base_online_predict = pd.DataFrame(predict_lis)
aux = df_base_online_predict.apply(
  lambda x: [accuracy_score(i, x[0]) for i in x],
  axis=1,
)
df_online_scores = pd.DataFrame(aux.to_list(), columns=df_base_online_predict.columns)
df_online_scores.to_csv(PATH/'csv'/'df_online_scores.csv', index=False)


def log_meta_offline_metrics(off_meta_eval):
    def inner(s):
        mean, std = np.mean(s), np.std(s)
        mean, std = np.round(np.mean(s), 3), np.round(np.std(s), 3)
        res = f"{s.name.capitalize().ljust(16)}: {mean:.3} ± {std:.3}"
        print(res)
        return mean, std
    df_offline_meta_eval = pd.DataFrame(off_meta_eval, columns=META_METRICS)
    mts = df_offline_meta_eval.apply(inner)
    mts.index = ['mean', 'std']
    mts.to_csv(PATH/'csv'/'meta_offline_metrics.csv', index='index')
    # Para ler:
    # pd.read_csv('offline_metrics.csv', index_col=0)
    return mts
log_meta_offline_metrics(off_meta_eval)



def log_meta_online_metrics(y_true, y_pred):
    mp = dict()
    for mtr, mtr_name in zip(mmetrics_fun, META_METRICS):
      mp[mtr_name] = (np.round(mtr(y_true=y_true, y_pred=y_pred), 3))
    mp = pd.Series(mp)
    joblib.dump(mp, PATH/'joblib'/'meta_online_metrics.joblib', )
    # Para ler:
    # pd.read_csv('offline_metrics.csv', index_col=0)
    return mp
mp = log_meta_online_metrics(true_lis, lis)



def plot_offline_performance(colors, PATH, MODELS, off_labels, df_off_scores):
    # df_off_scores = pd.DataFrame(base_evals)
    # off_labels = pd.Series(labels)
    f, (ax1, ax2) = plt.subplots(1, 2, dpi=200, figsize=(13, 6.6)) 
    f.suptitle('Fase OFFline', c='r', fontweight='bold', fontsize='x-large')
    A.plot_pie(off_labels, colors=colors, ax=ax1);
    def mean_std(x):
        return f"{np.mean(x):.3} ± {np.std(x):.3}"
    ax1.legend(df_off_scores.apply(mean_std), loc='lower right');
    ax1.set_title('Desempenho absoluto \n(empate = modelo que vem primeiro na lista de modelos)',)
    # df_off_evals = pd.DataFrame(base_evals)
    df_label_acc = df_off_scores.apply(lambda x: A.biggest_labelizer_draw(x.to_dict()), axis=1)
    df_off_labels = pd.DataFrame(
      df_label_acc.to_list(), columns=['label', 'acc']
    )
    A.plot_pie(df_off_labels.label, colors=colors, ax=ax2, autopct='%.1f')
    ax2.set_title('Distribuição e desempenho setorizados por melhor caso');
    def mean_std2(x):
        return f"{x.mean()[0]:.3} ± {x.std()[0]:.3}"
    ax2.legend(
    [mean_std2(df_off_labels[df_off_labels.label == c]) for c in MODELS + ['draw']] [::-1]
    , loc='upper right');
    plt.savefig(PATH/'png'/'initial-metabase-performance.png');
off_labels = pd.Series(ms.meta_y[:METABASE_INITIAL_SIZE])
vc=off_labels.value_counts()
DEFAULT = vc.index[vc.argmax()]
mlflow.set_tag('default', DEFAULT)
del vc

df_off_scores = pd.DataFrame(ms._base_evals[:METABASE_INITIAL_SIZE])
plot_offline_performance(
  colors, PATH, MODELS, off_labels, df_off_scores,
)
# In[ ]:
joblib.dump(ms.meta_y, PATH/'joblib'/'all_meta_y.joblib')
joblib.dump(ms._base_evals, PATH/'joblib'/'all_base_evals.joblib')



df_off = pd.DataFrame(ms._base_evals[:METABASE_INITIAL_SIZE])
d = df_off.apply(
    lambda x: A.biggest_labelizer_draw(x.to_dict()), axis=1
)
df_off_labels = pd.DataFrame(d.to_list(), columns=['label', 'acc'])



df_off_labels.to_csv(PATH/'csv'/'df_off_labels.csv', index=False)



def plot_online_produced_labels(PATH, METABASE_INITIAL_SIZE, labels):
    f, ax = plt.subplots(1, 1, figsize=(6.4, 4.8), dpi=100)
    ax.set_title('Rótulos produzidos na fase ONLINE', c='r',  fontsize=14)
    A.plot_pie(labels[METABASE_INITIAL_SIZE:], ax=ax);
    plt.savefig(PATH/'png'/'labels-generated-online.png')
joblib.dump(ms.meta_y, PATH/'joblib'/'all_labels.joblib')
plot_online_produced_labels(PATH, METABASE_INITIAL_SIZE, labels=ms.meta_y)
# In[ ]:





def plot_draw_labels(PATH, off_lbl, on_lbl):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 9.6), dpi=200)
    ax1.set_title('Rótulos REAIS da fase OFFLINE', c='r', fontsize=14)
    A.plot_pie(off_lbl, ax=ax1);
    ax2.set_title('Rótulos REAIS da fase ONLINE', c='b', fontsize=14)
    A.plot_pie(on_lbl, ax=ax2);
    plt.savefig(PATH/'png'/'labels-with-draw.png')
true = list(map(A.biggest_labelizer_draw, ms._base_evals))
true_lbl, true_score = A.lzip(*true)
off_lbl, on_lbl = np.split(true_lbl, [METABASE_INITIAL_SIZE]) 
plot_draw_labels(PATH, off_lbl, on_lbl)



def plot_no_draw(PATH, off_nodraw, on_nodraw):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 9.6), dpi=200)
    ax1.set_title('Rótulos REAIS SEM EMPATES da fase OFFLINE', 
              c='r', fontsize=14)
    A.plot_pie(off_nodraw, ax=ax1);
    ax2.set_title('Rótulos REAIS SEM EMPATES da fase ONLINE', 
              c='b', fontsize=14)
    A.plot_pie(on_nodraw, ax=ax2);
    plt.savefig(PATH/'png'/'labels-minus-draw.png')
    return off_nodraw,on_nodraw
off_nodraw = [i for i in off_lbl if i != 'draw']
on_nodraw = [i for i in on_lbl if i != 'draw']
plot_no_draw(PATH, off_lbl, on_lbl)
del off_lbl, on_lbl, off_nodraw, on_nodraw



def plot_final_labels(PATH, labels):
    f, ax = plt.subplots(1, dpi=110)
    ax = plt.subplot()
    plt.title(f'Rótulos finais ({len(labels)} metaexemplos)');
    A.plot_pie(labels, ax=ax)
    plt.savefig(PATH/'png'/'labels-all.png')
plot_final_labels(PATH, ms.meta_y)





def plot_base_gain(colors, PATH, MODELS, lis, df_score):
    global DEFAULT
    s = df_score.drop('true', axis=1).sum()
    print('DEFAULT: ', DEFAULT)
    f, axs = plt.subplots(len(MODELS), 1, dpi=200, figsize=(12.8, len(MODELS)*4.8))
    for ax, c, m in zip(axs, colors, s.sort_values(ascending=False).index):
        ganho = [
          (df_score[model_name][idx] - def_score)
          for idx, (model_name, def_score) 
          in enumerate(zip(
            df_score[lis], df_score[m]))
        ]
        cumsum=np.cumsum(ganho)
        xx = lambda: range(cumsum.shape[0])
        ax.set_facecolor('#A1A5A1')
        ax.fill_between(xx(), cumsum, color=c);
        ax.scatter(xx(), ganho, s=10, c='black')
        mean, std = np.round(np.mean(ganho), 3), np.round(np.std(ganho), 3)
        title = f'Ganho relativo ao {m}' + (m == DEFAULT) * " (Padrão) " + f": {mean} ± {std}"
        ax.set_title(title, color='r', fontweight='bold')
    plt.savefig(PATH/'png'/'relative-gain.png')
plot_base_gain(colors, PATH, MODELS, lis, df_online_scores)



def plot_offline_fti(colors, PATH, lgb):
    f, ax = plt.subplots(1, figsize=(12.8, 9.6), dpi=200)
    lightgbm.plot_importance(lgb, ax=ax, color=colors);
    plt.savefig(PATH/'png'/'final-fti.png')
plot_offline_fti(colors, PATH, ms.meta_model.lgb)



rl(A)
def plot_online_true_vs_pred(labels, lis, online_size):
    my = labels[-online_size:]
    f, (ax1, ax2) = plt.subplots(
      nrows=1, ncols=2,
      sharex=True, sharey=True,
      dpi=200, figsize=(12, 4.8),
    )
    f.suptitle('Fase online', c='r', fontweight='bold', fontsize=15);
    A.plot_pie(my, textprops={'fontsize': 14}, ax=ax1)
    ax1.set_title("Ground truth", weight='bold');
    
    my_pred = lis
    A.plot_pie(my_pred, textprops={'fontsize': 14}, ax=ax2)
    ax2.set_title("Predicts", weight='bold');
# ## Análise da metabase



def log_online_micro_performance(df_online_scores):
    on_metr = dict()
    for lbl in df_online_scores.columns.drop('true'):
        tmp = []
        query = df_online_scores[lbl]
        # on_metr[f'mean_{lbl}'] = 
        tmp.append(np.round(np.mean(query), 3))
        tmp.append(np.round(np.std(query), 3))
        # on_metr[f'std_{lbl}'] = np.round(query.std(), 3)         
        on_metr[lbl] = tmp
    df = pd.DataFrame(on_metr)
    df.index = ['mean', 'std']
    df.to_csv(PATH/'csv'/'online_micro_performance.csv', )
    return df
log_online_micro_performance(df_online_scores)



def plot_incoming_labels(PATH, MODELS, labels):
    mY800 = labels[:800]
    plt.figure(figsize=(15, 9.6))
    ax = plt.subplot(1,1,1, )
    for i, m in enumerate(MODELS):
        mlis = [idx for (idx, val) in enumerate(mY800) if val == m]
        ax.plot(mlis, len(mlis)*[i], marker='.', linewidth=0,)
    plt.title('Distribuição dos meta-rótulos ordenado por ordem de chegada')
    plt.legend(loc='upper right', labels=MODELS);
    plt.savefig(PATH/'png'/'incoming-labels.png')
plot_incoming_labels(PATH, MODELS, ms.meta_y)



df_base_online_predict = pd.DataFrame(predict_lis)
def log_base_macro_performance(META_METRICS, mmetrics_fun, df_base_online_predict):
  df = dict()
  columns = df_base_online_predict.columns.drop('true')
  
  print("Macro (base)")
  y_true = df_base_online_predict.true.explode(ignore_index=True).astype(int)
  for c in columns:
    tmp = []
    y_pred = df_base_online_predict[c].explode(ignore_index=True).astype(int)
    print(' ', c)
    for mtr, mtr_name in zip(mmetrics_fun, META_METRICS):
        val = np.round(mtr(y_true=y_true.values, y_pred=y_pred.values), 3)
        tmp.append(val)
        print(f'    {mtr_name}: {val}')
    df[c] = tmp
  df = pd.DataFrame(df)
  df.index = META_METRICS
  df.to_csv(PATH/'csv'/'base_macro_performance.csv', index=True)
  return df
log_base_macro_performance(
    META_METRICS, mmetrics_fun, df_base_online_predict
)



def log_base_micro(test, META_METRICS, mmetrics_fun, df_base_online_predict, df_online_scores, y_true):
    print("Micro (base)")
    y_true_chunks = np.split(y_true.values, list(y_true.index)[test::test])
    dici = {}
    for mtr, mtr_name in zip(mmetrics_fun, META_METRICS):      
        mtr_dict = dict()
        print(' ', mtr_name)
        for c in df_online_scores.columns.drop('true'):
            y_pred = df_base_online_predict[c].explode(ignore_index=True).astype(int)
            y_chunks = np.split(y_pred.values, list(y_pred.index)[test::test])
            res_lis = [mtr(yt,yp) for (yt, yp) in zip(y_true_chunks, y_chunks)]
            mean, std = np.round(np.mean(res_lis), 3), np.round(np.std(res_lis), 3)
            mtr_dict[c] = (mean, std)
            print(f'    {c}: {mean} ± {std}')
        dici[mtr_name] = mtr_dict
    dici = {k: pd.DataFrame(v) for (k,v) in dici.items()}
    for name, v in dici.items():
        v.index = ['mean', 'std']
        v.to_csv(PATH/'csv'/f'df_online_{name}.csv')
    return dici



y_true = df_base_online_predict.true.explode(ignore_index=True).astype(int)
log_base_micro(args.test, META_METRICS, mmetrics_fun, 
    df_base_online_predict, df_online_scores, y_true
);



def log_threshold_metrics(labels, lis, df_online_scores, thresh=.1):
    def top2_diff(x):
        sx = sorted(x, reverse=True)
        return abs(sx[0] - sx[1])
    idx_mask = df_online_scores.drop('true', axis=1).apply(
        top2_diff, axis=1) > thresh
    y_true = pd.Series(labels[-len(lis):])
    y_pred = pd.Series(lis)
    y_true, y_pred = y_true[idx_mask], y_pred[idx_mask]
    dici = A.prefixify({
      mname: np.round(mfun(y_true, y_pred), 3)
      for (mname, mfun)
      in zip(META_METRICS, mmetrics_fun)
    }, 'on_meta')
    return dici



def plot_online_meta_threshold(PATH, ms, lis, df_online_scores, log_threshold_metrics):
    ll = []
    xticks = np.linspace(0, 1, 11)
    for t in xticks:
        ll.append(log_threshold_metrics(ms.meta_y, lis, df_online_scores, t))
    dd = pd.DataFrame(ll).rename(
    lambda x: ''.join(x.split('_')[2:]) , axis=1)    
    dd.fillna(0).plot(marker='o', markersize=6, linewidth=1.2, )
  # https://stackoverflow.com/questions/30490740/move-legend-outside-figure-in-seaborn-tsplot
  # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);
    plt.xlabel('Threshold')
    return dd
df_online_meta = plot_online_meta_threshold(
    PATH, ms, lis, df_online_scores, log_threshold_metrics);
df_online_meta.to_csv(PATH/'csv'/'df_online_meta_threshold.csv')
df_online_meta.head(2) #.apply(print, axis=1,);
del df_online_meta


def plot_topn_mtf(PATH, df_fthis, n=3):
    topn=df_fthis.sum().sort_values(ascending=False)[:n].index
    plt.figure(figsize=(20, 12), dpi=250,)
    sns.lineplot(data=df_fthis[topn], linewidth=2, dashes=False,);
    plt.title(
      f'Importância, ao longo do tempo, das {n} MtF com maior importância acumulada',
      fontsize=30,
    )
    plt.legend(fontsize=20);
    plt.savefig(PATH/'png'/f'top{n}-fti.png')
plot_topn_mtf(PATH, df_fti, 5)


def Generatefy(x):
    for i in x:
        yield i
online_preds = Generatefy(lis)
online_scores = df_online_scores.apply(lambda x: x[next(online_preds)], axis=1).values
default_scores = df_online_scores[DEFAULT].values



def plot_heatmap(PATH, online_scores, default_scores):
    global DEFAULT
    fig, ax1 = plt.subplots(1, figsize=(8, 8), dpi=70)
    fig.subplots_adjust(bottom=0.25)
    heatmap, xedges, yedges = np.histogram2d(default_scores, online_scores, bins=11)
    mask = np.diag(np.ones(11))
    masked_data = ma.masked_array(heatmap.T, mask)
    cm.Spectral.set_bad(alpha=0)
    im1 = ax1.pcolormesh(masked_data, cmap=cm.Spectral)
    ax1.set_ylabel('Recommended score')
    ax1.set_title(bool(args.is_incremental)*"Não " + 'Incremental')
    ax1.set_xlabel(f'Default score ({DEFAULT})')
    ticks = ["{:.1f}".format(x) for x in np.linspace(0,1,11)]
    ticksl = ["{:.1f}".format(x) if int(x*10)%2==0 else "" for x in np.linspace(0,1,11)]
    ax1.set_xticks(np.arange(len(ticks))+.5, minor=False)
    ax1.set_yticks(np.arange(len(ticks))+.5, minor=False)
    ax1.set_xticklabels(ticksl, rotation=90, fontdict={'fontsize':12})
    ax1.set_yticklabels(ticksl, fontdict={'fontsize':12})
    ax1.figure.colorbar(im1, ax=ax1);
    plt.savefig(PATH/'png'/'heatmap.png')
plot_heatmap(PATH, online_scores, default_scores)



metadata = pd.DataFrame(ms.meta_x)
metadata.to_csv(PATH/'csv'/'metabase.csv')
metadata.to_csv(f'metabase_{args.base}.csv', index=False)



mlflow.log_artifacts(PATH)
mlflow.end_run()
shutil.rmtree(PATH)

