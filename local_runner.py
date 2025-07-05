from __future__ import division

from PIL import Image
import numpy as np
import gym
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D, Permute, BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow.keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

env_name = 'SpaceInvaders-v0'
env = gym.make(env_name)

#np.random.seed(123)
#env.seed(123)
nb_actions = env.action_space.n

class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)
    

model = Sequential()
# Bloque 0 - Capa de entrada
model.add(Permute((2, 3, 1), input_shape=(WINDOW_LENGTH,) + INPUT_SHAPE))

# Bloque 1 - Primer bloque convolucional: Detección de objetos grandes
model.add(Convolution2D(filters=32, kernel_size=8, strides=4,
    activation="relu", data_format="channels_first", padding="same"))

# Bloque 2 - Segundo bloque convolucional: Detección de detalles finos y localizados
model.add(Convolution2D(filters=64, kernel_size=4, strides=2,
    activation="relu", data_format="channels_first", padding="same"))

# Bloque 3 - Tercer bloque convolucional: Características visuales muy específicas
model.add(Convolution2D(filters=64, kernel_size=3, strides=1,
    activation="relu", data_format="channels_first", padding="same"))

# Bloque 4 - Aplanado de características
model.add(Flatten())

# Bloque 5 - Capa densa intermedia
model.add(Dense(512, activation="relu"))

# Bloque 6 - Capa de salida
model.add(Dense(nb_actions, activation="linear"))

#1. Memory (Replay buffer)
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)

#2. Exploration policy
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                              value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

#3. Definición del agente
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy,
               memory=memory, processor=AtariProcessor(),
               nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)

dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])


final_weights = sys.argv[1]

print('Cargando pesos...')
dqn.load_weights(final_weights)
test_result = dqn.test(env, nb_episodes=10, visualize=True)
print(np.mean(test_result.history.get('episode_reward', [])))
