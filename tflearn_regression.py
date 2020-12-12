
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

tf.__version__

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(url, names=column_names,
                          na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)

dataset = raw_dataset.copy()

dataset.isna().sum()

dataset = dataset.dropna()

dataset['Origin'] = dataset['Origin'].map({
    1:'USA',
    2:'Europe',
    3:'Japan'
})

get_ipython().run_line_magic('pinfo', 'pd.get_dummies')

dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
dataset.tail()

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# sns.pairplot
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')


train_dataset.describe().transpose()

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')


train_dataset.describe().transpose()[['mean', 'std']]

normalizer = preprocessing.Normalization()

normalizer.adapt(np.array(train_features))

print(normalizer.mean.numpy())

train_features[:5]

first = np.array(train_features[:1])

print(normalizer.mean.numpy())

with np.printoptions(precision=2, suppress=True):
  print(first)
  print(normalizer(first).numpy())

second = np.array(train_features[:2])
zero = np.array(train_features[1:2])

first = np.array(train_features[:2])

with np.printoptions(precision=2, suppress=True):
  print('First example :', first)
  print()
  print('Normalized :', normalizer(first).numpy())


# ini untuk menormalisasikan data horsepower
horsepower = np.array(train_features['Horsepower'])

horsepower_normalization = preprocessing.Normalization(input_shape=[1,])
horsepower_normalization.adapt(horsepower)


# create NN dengan output berupa 1 node
# (input ternormalisasi) -> (output)
horsepower_model = keras.Sequential([horsepower_normalization,
                                     keras.layers.Dense(units=1)])

horsepower_model.summary()


horsepower_model.predict(horsepower[:10])


type(horsepower_model)
get_ipython().run_line_magic('pinfo', 'keras.Sequential.predict')


horsepower_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)


get_ipython().run_cell_magic('time', '', "history = horsepower_model.fit(\n    train_features['Horsepower'], \n    train_labels, \n    epochs=100,\n    verbose=0, \n    validation_split=0.2\n)")


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 20])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)



plot_loss(history)


test_result = {}

test_result['horsepower_model'] = horsepower_model.evaluate(
    test_features['Horsepower'],
    test_labels, verbose=0
)


x = tf.linspace(0.0, 250, 251)
y = horsepower_model.predict(x)

get_ipython().run_line_magic('pinfo', 'horsepower_model.predict')


def plot_horsepower(x, y):
  plt.scatter(train_features['Horsepower'], train_labels, label='Data')
  plt.plot(x, y, color='k', label='Predictions')
  plt.xlabel('Horsepower')
  plt.ylabel('MPG')
  plt.legend()


plot_horsepower(x, y)


linear_model = tf.keras.Sequential([
                                    normalizer,
                                    layers.Dense(units=1)
])

linear_model.layers[1].kernel

linear_model.predict(train_features[:10])

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

get_ipython().run_cell_magic('time', '', 'new_history = linear_model.fit(\n    train_features, train_labels,\n    epochs=100,\n    verbose=0,\n    validation_split=0.2\n)')

plot_loss(new_history)

test_result['linear_mode'] = linear_model.evaluate(
    test_features, test_labels, verbose=0
)

# bentuk layer : (input_norm) -> (64 hidden layer) -> (64 hidden layer) -> (single output)
def build_and_compile_model(norm):
  model = keras.Sequential([
                            norm,
                            layers.Dense(64, activation='relu'),
                            layers.Dense(64, activation='relu'),
                            layers.Dense(1)
  ])

  model.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001))
  return model

dnn_horsepower_model = build_and_compile_model(horsepower_normalization)

dnn_horsepower_model.summary()


get_ipython().run_cell_magic('time', '', "history = dnn_horsepower_model.fit(\n    train_features['Horsepower'],\n    train_labels,\n    validation_split=0.2,\n    verbose=0,\n    epochs=100\n)")


plot_loss(history)

x = tf.linspace(0.0, 250, 251)
y = dnn_horsepower_model.predict(x)


plot_horsepower(x, y)


test_result['dnn_horsepower_model'] = dnn_horsepower_model.evaluate(
    test_features['Horsepower'], 
    test_labels,
    verbose=0
)


dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

get_ipython().run_cell_magic('time', '', 'history=dnn_model.fit(\n    train_features,\n    train_labels,\n    validation_split=0.2,\n    verbose=2,\n    epochs=1000\n)')


plot_loss(history)

test_result['dnn_model'] = dnn_model.evaluate(test_features,
                                               test_labels,
                                               verbose=0)


pd.DataFrame(test_result, index=['Mean absolute error [MPG]']).T


test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)


error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Predictions Error [MPG]')
_ = plt.ylabel('Count')


dnn_model.save('dnn_model')


reloaded = tf.keras.models.load_model('dnn_model')

test_result['reloaded'] = reloaded.evaluate(
    test_features, test_labels, verbose=0
)


pd.DataFrame(test_result, index=['Mean absolute error [MPG]']).T


get_ipython().run_line_magic('pinfo', 'normalizer.adapt')


print(help(tf.lite.TFLiteConverter))


converter = tf.lite.TFLiteConverter.from_saved_model('dnn_model')
tflite_model = converter.convert()

with open('model_dnn_regression.tflite', 'wb') as f:
  f.write(tflite_model)


get_ipython().run_line_magic('pinfo', 'dnn_model.predict')


test_features[3:4]


hey = np.array(test_features[3:4])
hey

heyhey = [[4., 113., 95., 2228., 14., 71., 0., 1., 0.]]
heyhey


dnn_model.predict(np.array(heyhey))

np.array(test_labels[:10])


get_ipython().system('ls')

