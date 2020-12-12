import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
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

dataset.drop(['kost_name_rough'], axis=1, inplace=True)

max_n_cat = None

for n,c in dataset.items(): 
    if not is_numeric_dtype(c) and (max_n_cat is None or c.nunique()>max_n_cat):
        dataset[n] = c.cat.codes+1

print(list(dataset.groupby(["kota"]).size().reset_index(name='counts')["kota"]))
print(list(raw_dataset.groupby(["kota"]).size().reset_index(name='counts')["kota"]))

print(list(dataset.groupby(["type_kos"]).size().reset_index(name='counts')["type_kos"]))
print(list(raw_dataset.groupby(["type_kos"]).size().reset_index(name='counts')["type_kos"]))

print(list(dataset.groupby(["area"]).size().reset_index(name='counts')["area"]))
print(list(raw_dataset.groupby(["area"]).size().reset_index(name='counts')["area"]))

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset, diag_kind='kde')

train_dataset.describe().transpose()

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('harga_nomina')
test_labels = test_features.pop('harga_nomina')

train_features

train_labels

normalizer = preprocessing.Normalization()

normalizer.adapt(np.array(train_features))

print(normalizer.mean.numpy())

train_features[:5]

first = np.array(train_features[:1])

print(normalizer.mean.numpy())


with np.printoptions(precision=2, suppress=True):
  print(first)
  print(normalizer(first).numpy())

np.shape(np.array(train_features))

model = keras.Sequential([
                            layers.Input(shape=(4,)),
                            layers.Dense(64, activation='relu'),
                            layers.Dense(64, activation='relu'),
                            layers.Dense(1)
  ])

model.compile(loss='mean_absolute_error',
            optimizer=tf.keras.optimizers.Adam(0.001))

model.summary()

get_ipython().run_cell_magic('time', '', 'history = model.fit(\n    train_features,\n    train_labels,\n    validation_split=0.2,\n    verbose=0,\n    epochs=100\n)')


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 20])
    plt.xlabel('Epoch')
    plt.ylabel('Error [harga_nomina]')
    plt.legend()
    plt.grid(True)

plot_loss(history)

test_result = model.evaluate(
    test_features, 
    test_labels,
    verbose=0
)

get_ipython().run_cell_magic('time', '', 'history=model.fit(\n    train_features,\n    train_labels,\n    validation_split=0.2,\n    verbose=2,\n    epochs=1000\n)')

plot_loss(history)

test_result = model.evaluate(test_features,
                             test_labels,
                             verbose=0)


test_predictions = model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values [harga_nomina]')
plt.ylabel('Predictions [harga_nomina]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

test_predictions

test_labels

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Predictions Error [harga_nomina]')
_ = plt.ylabel('Count')

model.save('getkos_model')

reloaded = tf.keras.models.load_model('getkos_model')

test_result = reloaded.evaluate(
    test_features, test_labels, verbose=0
)

test_result

converter = tf.lite.TFLiteConverter.from_saved_model('getkos_model')
tflite_model = converter.convert()

with open('model_getkos_regression.tflite', 'wb') as f:
    f.write(tflite_model)

test_features[3:4]

model.predict(np.array(test_features[3:4]))

test_labels[3:4]

get_ipython().system('ls')


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

cat_map = {"kota":dict_kota, "type_kos":dict_type_kos, "area":dict_area}

with open('cat_map.json', 'w') as fp:
    json.dump(cat_map, fp)

sns.pairplot(x, diag_kind='kde')


train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
