#Faça o download do dataset UCI HAR.
#Carregue os dados em Python.
#Identifique número de amostras, atributos e classes.

import pandas as pd
import os

# Caminho base para o dataset
dataset_path = '/home/emilly/Trabalho_2_Tp/Redu-o-de-Dimensionalidade-na-Base-UCI-HAR/UCI HAR Dataset'

features_path = os.path.join(dataset_path, 'features.txt')
features = pd.read_csv(features_path, sep=' ', header=None, names=['id', 'name'])
feature_names = features['name'].tolist()

# Carregar labels das atividades
activity_labels_path = os.path.join(dataset_path, 'activity_labels.txt')
activity_labels = pd.read_csv(activity_labels_path, sep=' ', header=None, names=['id', 'label'])
class_names = activity_labels['label'].tolist()

# Carregar dados de treino
X_train_path = os.path.join(dataset_path, 'train/X_train.txt')
y_train_path = os.path.join(dataset_path, 'train/y_train.txt')
X_train = pd.read_csv(X_train_path, delim_whitespace=True, header=None)
y_train = pd.read_csv(y_train_path, delim_whitespace=True, header=None)

# Carregar dados de teste
X_test_path = os.path.join(dataset_path, 'test/X_test.txt')
y_test_path = os.path.join(dataset_path, 'test/y_test.txt')
X_test = pd.read_csv(X_test_path, delim_whitespace=True, header=None)
y_test = pd.read_csv(y_test_path, delim_whitespace=True, header=None)

# Combinar dados de treino e teste
X = pd.concat([X_train, X_test], axis=0)
y = pd.concat([y_train, y_test], axis=0)

# Atribuir nomes às colunas de X
X.columns = feature_names

# Número de amostras
num_samples = X.shape[0]

# Número de atributos
num_attributes = X.shape[1]

# Número de classes
num_classes = len(class_names)

print(f"Número de amostras: {num_samples}")
print(f"Número de atributos: {num_attributes}")