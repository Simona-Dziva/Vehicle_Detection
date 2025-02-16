# import libraries

import kagglehub
import pandas as pd
import os

from PIL import Image
import numpy as np

from sklearn.model_selection import train_test_split
import random

import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import BatchNormalization,MaxPool2D,Dense,Conv2D,Flatten
from keras.callbacks import EarlyStopping,LearningRateScheduler

from sklearn.metrics import classification_report , ConfusionMatrixDisplay , confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib.pyplot as plt

# Stažení datové sady
path = kagglehub.dataset_download("brsdincer/vehicle-detection-image-set")
print("Path to dataset files:", path)

# Nastavenie cesty k datasetu
dataset_path_1 = "/root/.cache/kagglehub/datasets/brsdincer/vehicle-detection-image-set/versions/1/data/non-vehicles"
dataset_path_2 = "/root/.cache/kagglehub/datasets/brsdincer/vehicle-detection-image-set/versions/1/data/vehicles"

# Načtení dat
df=pd.DataFrame(columns=['image','label'])

# Overenie, či priečinok 1 non-vehicles existuje
if os.path.exists(dataset_path_1):
    print("Dataset nájdený!")
    print(df['label'].value_counts())
    print("Zoznam súborov:", os.listdir(dataset_path_1))
else:
    print("Zložka non-vehicles neexistuje. Skontroluj cestu k datasetu.")

# Overenie, či priečinok 2 vehicle existuje
if os.path.exists(dataset_path_2):
    print("Dataset nájdený!")
    print(df['label'].value_counts())
    print("Zoznam súborov:", os.listdir(dataset_path_2))
else:
    print("Zložka vehicles neexistuje. Skontroluj cestu k datasetu.")

# Non-vehicles:Inicializácia zoznamov na efektívnejšie ukladanie dát
images = []
labels = []

# Prechádzame cez obrázky - non vehicle
for name in os.listdir(dataset_path_1):  # Nájde všetky formáty
    filepath = os.path.join(dataset_path_1, name)  # Získanie úplnej cesty každého súboru
    image = Image.open(filepath)
    img = np.array(image)  # Priamo konvertujeme PIL image na NumPy pole

    # Skontrolujeme, či má správny rozmer (64,64,3)
    if img.shape != (64, 64, 3):
        img = img.reshape((64, 64, 3))  # Pokus o reshape, ale môže zlyhať, ak veľkosť nepasuje

    images.append(img)
    labels.append(0)  # Predpokladáme, že label pre všetky obrázky je 0

# Vytvorenie DataFrame
df_non_vehicles = pd.DataFrame({'image': images, 'label': labels})

# Vehicles: Inicializácia zoznamov na efektívnejšie ukladanie dát
images = []
labels = []

# Prechádzanie cez všetky obrázky v zložke - vehicle
for filename in os.listdir(dataset_path_2):  # Iteracia cez súbory v adresári
    filepath = os.path.join(dataset_path_2, filename)  # Získanie úplnej cesty každého súboru
    try:
        image = Image.open(filepath)
        img = np.array(image)  # Priamo konvertujeme PIL image na NumPy pole

        # Skontrolujeme, či má správny rozmer (64,64,3)
        if img.shape != (64, 64, 3):
            img = img.reshape((64, 64, 3))  # Pokus o reshape, ale môže zlyhať, ak veľkosť nesedi

        images.append(img)
        labels.append(1)  # Predpokladáme, že label pre všetky obrázky je 1
    except Exception as e:
        print(f"Error processing file {filename}: {e}")

# Vytvorenie DataFrame
df_vehicles = pd.DataFrame({'image': images, 'label': labels})

# Sloučení dat
df = pd.concat([df_vehicles, df_non_vehicles]).reset_index()

# rozdelenie dát na tréningový a testovací set
x_train, x_test, y_train, y_test = train_test_split(
    df['image'], df['label'], random_state=42, test_size=0.2, stratify=df['label'])

# Zmena rozmeru obraázkov
def change_image_dimension(data):
    data = np.reshape(data.to_list(), (len(data), 64, 64, 3))
    return data

x_train = change_image_dimension(x_train)
x_test = change_image_dimension(x_test)

# Normalizácia / škálovnie hodnôt pixelov z [0, 255] na [0, 1]
x_train = np.stack(x_train) / 255.0
x_test = np.stack(x_test) / 255.0

# Creating Arrays in NumPy (int) with an Integer Type
y_train = np.array(y_train, dtype=int)
y_test = np.array(y_test, dtype=int)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)