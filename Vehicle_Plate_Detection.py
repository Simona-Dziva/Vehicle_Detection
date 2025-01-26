import kagglehub

# Download latest version
path = kagglehub.dataset_download("brsdincer/vehicle-detection-image-set")

print("Path to dataset files:", path)

import pandas as pd

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Input, Dense, Dropout

from matplotlib import pyplot as plt