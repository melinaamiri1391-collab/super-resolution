import keras

def load_model():
    return keras.saving.load_model('super_resolution.keras')