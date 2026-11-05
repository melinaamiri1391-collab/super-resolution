import keras
from keras import layers


def build_model(X_train):
    input = layers.Input(shape=X_train.shape[1:])
    c1    = layers.Conv2D(64, 5, padding='same', activation='relu')(input)
    sdr1  = layers.SpatialDropout2D(0.2)(c1)
    c2    = layers.Conv2D(128, 3, padding='same', activation='relu')(sdr1)
    c3    = layers.Conv2D(256, 3, padding='same', activation='relu')(c2)
    up1   = layers.UpSampling2D((2,2), interpolation='bilinear')(c3)
    c4    = layers.Conv2D(3, 1, padding='same')(up1)

    model = keras.Model(input, c4)
    return model