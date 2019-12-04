import os
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import backend as K
import gym
from replay_buffer import ReplayBuffer
from ouanoise import OUActionNoise
from actor import Actor
from critic import Critic
tf.disable_v2_behavior()



class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 max_size=10000, layer1_size=400,
                 layer2_size=300, batch_size=64):
        n_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.sess = tf.Session()


        self.actor  = Actor(alpha, n_actions, 'Actor', input_dims, self.sess,
                            layer1_size, layer2_size, env.action_space.high, 
                            self.batch_size, ckpt_dir='tmp/ddpg/actor')

        self.critic = Critic(beta, n_actions, 'Critic', input_dims, self.sess,
                             layer1_size, layer2_size, self.batch_size,
                             ckpt_dir='tmp/ddpg/critic')

        self.target_actor  = Actor(alpha, n_actions, 'TargetActor', input_dims,
                                   self.sess,layer1_size, layer2_size, 
                                   env.action_space.high, self.batch_size,
                                   ckpt_dir='tmp/ddpg/target_actor')

        self.target_critic = Critic(beta, n_actions, 'TargetCritic', input_dims,
                                    self.sess, layer1_size, layer2_size, 
                                    self.batch_size, ckpt_dir='tmp/ddpg/target_critic')

        self.noise = OUActionNoise(mu=np.zeros(n_actions))


        self.update_actor = [self.target_actor.params[i].assign(
                             tf.multiply(self.actor.params[i], self.tau) +
                             tf.multiply(self.target_actor.params[i], 1. -self.tau))
                             for i in range(len(self.target_actor.params))]
        
        
        self.update_critic = [self.target_critic.params[i].assign(
                              tf.multiply(self.critic.params[i], self.tau) +
                              tf.multiply(self.target_critic.params[i], 1. -self.tau))
                                                for i in range(len(self.target_critic.params))]
        
        self.sess.run(tf.global_variables_initializer())

        self.update_target_network_parameters(first=True)

    def update_target_network_parameters(self, first=False):
        for _, d in enumerate(["/device:GPU:0", "/device:GPU:1"]):
            with tf.device(d):
                if first:
                    old_tau = self.tau
                    self.tau = 1.0
                    self.target_actor.sess.run(self.update_actor)
                    self.target_critic.sess.run(self.update_critic)
                    self.tau = old_tau
                else:
                    self.target_critic.sess.run(self.update_critic)
                    self.target_actor.sess.run(self.update_actor)

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def choose_action(self, state):
        # print("State[0]: ",state[0].shape)
        # print("State[1]: ",state[1].shape)
        state1 = state[0][np.newaxis, :]
        state2 = state[1][np.newaxis, :]
        state = [state1, state2]
        for _, d in enumerate(["/device:GPU:0", "/device:GPU:1"]):
            with tf.device(d):
                mu = self.actor.predict(state)
        noise = self.noise()
        mu_prime = mu + noise

        return mu_prime[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        for _, d in enumerate(["/device:GPU:0", "/device:GPU:1"]):
            with tf.device(d):
                state, action, reward, new_state, done = \
                                            self.memory.sample_buffer(self.batch_size)
                #target q-value(new_state) with actor's bounded action forward pass
                critic_value_ = self.target_critic.predict(new_state,
                                                self.target_actor.predict(new_state))

                target = []
                for j in range(self.batch_size):
                    target.append(reward[j] + self.gamma*critic_value_[j]*done[j])

                target = np.reshape(target, (self.batch_size, 1))

                _ = self.critic.train(state, action, target) #s_i, a_i and y_i

                # a = mu(s_i)
                a_outs = self.actor.predict(state)
                # gradients of Q w.r.t actions
                grads = self.critic.get_action_gradients(state, a_outs)

                self.actor.train(state, grads[0])

                self.update_target_network_parameters(first=True)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()
    
    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()