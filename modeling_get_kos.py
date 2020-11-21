import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

np.set_printoptions(precision=3, suppress=True)

tf.__version__

raw_dataset = pd.read_csv("features.csv", na_values='?'
                          , comment='\t'
                          , skipinitialspace=True)

dataset = raw_dataset.copy()

dataset.isna().sum()

dataset = dataset.dropna()


def train_cats(df):
	for n,c in df.items():
	    if is_string_dtype(c): 
	        df[n] = c.astype('category').cat.as_ordered()

train_cats(dataset)


def proc_df(df, y_fld = 'harga_nomina', skip_flds=['kost_name_rough'],	na_dict = {}, max_n_cat = None):

	y = dataset[y_fld].values
	dataset.drop(skip_flds+[y_fld], axis=1, inplace=True)

	for n,c in dataset.items(): 
	    if not is_numeric_dtype(c) and (max_n_cat is None or c.nunique()>max_n_cat):
	        dataset[n] = c.cat.codes+1
	    
	res = [pd.get_dummies(dataset, dummy_na=True), y, na_dict]

	return res

x, y, nas = proc_df(dataset, y_fld = 'harga_nomina', skip_flds=['kost_name_rough'],	na_dict = {}, max_n_cat = None)


def get_dict_encode(df_1,df_2,col='kota'):
    
    zip_iterator = zip(
        list(df_1.groupby([col]).size().reset_index(name='counts')[col]), 
        list(df_2.groupby([col]).size().reset_index(name='counts')[col])
    )
    
    return dict(zip_iterator)

dict_kota = get_dict_encode(raw_dataset,x,col='kota')
dict_type_kos = get_dict_encode(raw_dataset,x,col='type_kos')
dict_area = get_dict_encode(raw_dataset,x,col='area')





