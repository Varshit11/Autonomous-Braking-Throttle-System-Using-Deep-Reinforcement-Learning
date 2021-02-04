import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.initializers import RandomUniform
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, concatenate, LSTM, Reshape, BatchNormalization, Lambda, Flatten
from keras.layers import LeakyReLU
# from keras.layers.advanced_activations import LeakyRelu


class Critic:
    """ Critic for the DDPG Algorithm, Q-Value function approximator
    """

    def __init__(self, inp_dim, out_dim, lr, tau):
        # Dimensions and Hyperparams
        self.env_dim = inp_dim
        self.act_dim = out_dim
        self.tau, self.lr = tau, lr
        # Build models and target models
        self.model = self.network()
        self.target_model = self.network()
        self.model.compile(Adam(self.lr), 'mse')
        self.target_model.compile(Adam(self.lr), 'mse')
        # Function to compute Q-value gradients (Actor Optimization)
        self.action_grads = K.function([self.model.input[0], self.model.input[1]], K.gradients(self.model.output, [self.model.input[1]]))

    def network(self):
        """ Assemble Critic network to predict q-values
        """
        state = Input((self.env_dim))
        action = Input((self.act_dim,))
        x = Dense(400, kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1))(state)
        # x = LeakyReLU(alpha=0.8)(x)
        # x = LeakyReLU(alpha=0.3)(x)
        # x = concatenate([Flatten()(x), action])
        x = concatenate([x, action])
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.8)(x)
        # x = BatchNormalization()(x)

        x = Dense(200, kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.8)(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU(alpha=0.5)(x)

        x = Dense(100, kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.8)(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU(alpha=0.5)(x)

        x = Dense(200, kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.8)(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU(alpha=0.5)(x)

        x = Dense(400, kernel_initializer=RandomUniform(minval=-0.1, maxval=0.1))(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.8)(x)
        # x = BatchNormalization()(x)
        # x = LeakyReLU(alpha=0.8)(x)

        out = Dense(1, activation='linear', kernel_initializer=RandomUniform(minval=-0.03, maxval=0.03))(x)
        return Model([state, action], out)

    def gradients(self, states, actions):
        """ Compute Q-value gradients w.r.t. states and policy-actions
        """
        return self.action_grads([states, actions])

    def target_predict(self, inp):
        """ Predict Q-Values using the target network
        """
        return self.target_model.predict(inp)

    def train_on_batch(self, states, actions, critic_target):
        """ Train the critic network on batch of sampled experience
        """
        return self.model.train_on_batch([states, actions], critic_target)

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        W, target_W = self.model.get_weights(), self.target_model.get_weights()
        for i in range(len(W)):
            target_W[i] = self.tau * W[i] + (1 - self.tau)* target_W[i]
        self.target_model.set_weights(target_W)

    def save(self, path):
        self.model.save_weights(path + '_critic.h5')

    def load_weights(self, path):
        self.model.load_weights(path)
