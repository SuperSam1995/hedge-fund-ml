# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% pycharm={"name": "#%%\n"}
# %load_ext autoreload
# %autoreload 2

from random import randint
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import plotly.express as px

def read_csv(loc, date=True):
    df = pd.read_csv(loc)
    if date:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    return df


def dic_read(loc):
    a_file = open(loc, "rb")
    output = pickle.load(a_file)
    return output


def set_seed(seed_value=123):
    import os
    import random
    import tensorflow as tf
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)
    from keras import backend as K
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)
    K.set_session(sess)


def random_sampling(dataset, n_sample, window):
    '''
    implicitly assuming there is no calendar effect.
    :param dataset: np.ndarray
    :param n_sample:
    :param window:
    :return:
    '''
    isinstance(dataset, np.ndarray)
    step = 0
    res = []
    while step < n_sample:
        step += 1
        randidx = randint(0, dataset.shape[0] - window)
        res.append(dataset[randidx:window + randidx])
    # label as real data
    # label = np.ones(n_sample)
    # return np.array(res), label
    return np.array(res)


set_seed()

hfd = read_csv('../cleaned_data/hfd.csv')
factor_etf_data = read_csv('../cleaned_data/factor_etf_data.csv')
hfd_fullname = dic_read('../cleaned_data/hfd_fullname.pkl')
factor_etf_name = dic_read('../cleaned_data/factor_etf_name.pkl')
rf = read_csv('../cleaned_data/rf.csv')

all_data_name = {**factor_etf_name, **hfd_fullname}

# dataset = factor_etf_data.join(hfd).join(rf)
dataset = factor_etf_data.join(hfd)
data_scaler = MinMaxScaler()
data = data_scaler.fit_transform(dataset)

dataset = random_sampling(data, 2000, 48)

# %% pycharm={"name": "#%%\n"}
dataset.shape

# %% pycharm={"name": "#%%\n"}
from tensorflow import keras
#load trained model under root ./trained_generator
import glob
import os
path='D:\\Cambridge_dissertation\\GAN\\trained_generator\\old'
all_file=glob.glob(os.path.join(path,'*.h5'))

model_name=[]
generated_data = []
for filename in all_file:
    model=keras.models.load_model(filename)
    noise = np.random.normal(0, 1, (1000, 48, 35))
    generated_data.append(model.predict(noise))
    # model_name.append(filename[48:-20])
    model_name.append(filename[48+4:-20])

# %% pycharm={"name": "#%%\n"}
# add benchmark
generated_data.append(random_sampling(data,1000,48))
model_name.append('Random_Sampling')

# %% pycharm={"name": "#%%\n"}
generated_data[0].shape

# %% pycharm={"name": "#%%\n"}
isinstance(generated_data[0],np.ndarray)

# %% pycharm={"name": "#%%\n"}
generated_data[-1].shape

# %% pycharm={"name": "#%%\n"}
dataset[:1000].shape

# %% pycharm={"name": "#%%\n"}
from GAN_eval import GAN_eval

subplot_title = list(all_data_name.values())
# subplot_title.append('RF')
res=[]
ks= []
for i in range(len(model_name)):
    evaluation = GAN_eval(dataset[:1000],generated_data[i],dataset[1000:],subplot_title=subplot_title,model_name=[model_name[i]])
    res.append(evaluation.run_all())
    ks.append(evaluation.ks_test())

# %% pycharm={"name": "#%%\n"}
len(res)

# %% pycharm={"name": "#%%\n"}
gan_metrics = round(pd.concat(res,axis=1),5)

# %% pycharm={"name": "#%%\n"}
gan_metrics

# %% pycharm={"name": "#%%\n"}
px.bar(gan_metrics)

# %% pycharm={"name": "#%%\n"}
gan_metrics

# %% pycharm={"name": "#%%\n"}
#scale the data back to monthly logged return
rescale_data = []
for i in range(generated_data[0].shape[0]):
    rescale_data.append(data_scaler.inverse_transform(generated_data[0][i]))

# %% pycharm={"name": "#%%\n"}
# save data
np.save('MTSS_WGANGP_generated.npy',np.array(rescale_data))
