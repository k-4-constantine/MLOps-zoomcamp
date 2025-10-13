#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().system('python -V')


# In[8]:


import pandas as pd


# In[9]:


import pickle


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[11]:


from sklearn.feature_extraction import DictVectorizer

from sklearn.metrics import root_mean_squared_error


# In[12]:


import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")


# In[15]:


def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    return df


# In[16]:


df_train = read_dataframe('./data/green_tripdata_2021-01.parquet')
df_val = read_dataframe('./data/green_tripdata_2021-02.parquet')


# In[17]:


len(df_train), len(df_val)


# In[19]:


categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
numerical = ['trip_distance']

dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)


# In[20]:


target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values


# In[22]:


from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import xgboost as xgb


# In[27]:


train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)


# In[36]:


from pathlib import Path
models_folder = Path('models')
models_folder.mkdir(exist_ok=True)


# In[38]:


with mlflow.start_run():
    best_params = {
    'learning_rate':0.7645973055915408,
    'max_depth':33,
    'min_child_weight':3.3858551804076,
    'objective':'reg:linear',
    'reg_alpha':0.29589767192164507,
    'reg_lambda': 0.0579918889780704,
    'seed':42
    }

    mlflow.log_params(best_params)

    booster = xgb.train(
                params=best_params, 
                dtrain=train,
                num_boost_round=30,
                evals=[(valid,"validation")],
                early_stopping_rounds=50
            )

    y_pred = booster.predict(valid)
    rmse = root_mean_squared_error(y_val, y_pred)
    mlflow.log_metric("rmse", rmse)

    with open("models/preprocessor.b", "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

    mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")


# In[ ]:




