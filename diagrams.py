print("ENTRA")
from keras.models import Sequential
from keras.layers import Permute, Conv2D, BatchNormalization, Activation, Flatten, Dense
from keras.utils import plot_model
from keras_visualizer import visualizer 
import os

# Parámetros
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4
nb_actions = 6  # SpaceInvaders-v0

# Construcción del modelo
model = Sequential()
model.add(Permute((2, 3, 1), input_shape=(WINDOW_LENGTH,) + INPUT_SHAPE))
model.add(Conv2D(32, (8, 8), strides=(4, 4)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (4, 4), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3), strides=(1, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

# Crear diagrama
try:

    plot_model(
        model,
        to_file="dqn_model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir="TB",  # "TB" para vertical, "LR" para horizontal
        expand_nested=False,
        show_layer_activations=True,
        dpi=120
    )
    visualizer(model, file_format='png', view=True)
    model.save("dqn_model.h5")
    print("✅ Diagrama generado correctamente.")
except Exception as e:
    print("❌ Error al generar el diagrama:")
    raise e  # Lanza el error completo

