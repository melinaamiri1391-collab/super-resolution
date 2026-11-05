import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

import keras
from keras import layers, activations, losses, optimizers, callbacks

from PIL import Image
import os

from model import build_model
from load_model import load_model

low_img  = []
high_img = []

low_size  = (128,128)
high_size = (256,256)

for name in sorted(os.listdir('SR/low_res')):
    image_path = os.path.join('SR/low_res', name)

    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(low_size)
    img = np.array(img)/255
    low_img.append(img)


for name in sorted(os.listdir('SR/high_res')):
    image_path = os.path.join('SR/high_res', name)

    img = Image.open(image_path)
    img = img.convert('RGB')
    img = img.resize(high_size)
    img = np.array(img)/255
    high_img.append(img)



low_img  = np.array(low_img)
high_img = np.array(high_img)


X_train, X_test, y_train, y_test = train_test_split(low_img, high_img, test_size=0.2, random_state=1)


model = build_model(X_train=X_train)


adam = optimizers.Adam()
loss = losses.Huber()

model.compile(
    optimizer=adam,
    loss=loss,
    metrics=['mae']
)

es  = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5)

history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=16,
    callbacks=[es, rlr],
    verbose=0
)
