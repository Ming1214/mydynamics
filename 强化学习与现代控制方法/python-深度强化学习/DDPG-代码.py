#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt
import tensorflow as tf

tfk = tf.keras


# ### 状态积分器

# In[2]:


def func(t, y, tau_p = 0., tau_m = 0.):
    d_x = -y[3]*np.sin(y[2])
    d_y = y[3]*np.cos(y[2])
    d_phi = y[4]
    d_eta1 = -1/6*y[3]+1/12*tau_p
    d_eta2 = -20/21*y[4]+1/21*tau_m
    return d_x, d_y, d_phi, d_eta1, d_eta2


# ### 环境

# In[3]:


class Env():
    
    def __init__(self, delta_t = 1, end_t = 200, init = [1., 1., 0.]):
        self.delta_t = delta_t
        self.end_t = end_t
        self.init_state = np.array(init+[0., 0.])
        self.target = np.array([0, 0, 0, 0, 0])
        self.reset()
    
    def step(self, tau_r, tau_l):
        self.time += self.delta_t
        sol = solve_ivp(func, [0, self.delta_t], self.state, args = (tau_r, tau_l), t_eval = [self.delta_t])
        new_state = sol.y[:, 0]
        new_state[2] %= (2*np.pi)
        if new_state[2] > np.pi:
            new_state[2] -= (2*np.pi)
        new_distance = np.sqrt(np.sum(np.square(self.target[:3]-new_state[:3])))
        # reward = self.distance - new_distance
        reward = -(new_distance**2+0.1*(self.distance-new_distance)**2)
        self.state = new_state
        self.distance = new_distance
        self.trace = np.vstack((self.trace, self.state.reshape(1, -1)))
        self.dis = np.vstack((self.dis, [self.distance]))
        return self.target-self.state, reward, 1. if self.time >= self.end_t else 0.
    
    def reset(self):
        self.time = 0.
        self.state = self.init_state[: ]
        self.trace = np.array([self.state])
        self.distance = np.sqrt(np.sum(np.square(self.target[:3]-self.state[:3])))
        self.dis = np.array([self.distance])
        return self.target - self.state
    
    def show(self):
        ts = np.linspace(0, self.end_t, num = self.dis.shape[0], endpoint = True)
        plt.figure(figsize = (10, 5))
        plt.subplot(2, 1, 1)
        plt.plot(ts, self.dis, "k-", label = "distance")
        plt.legend()
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.plot(ts, self.trace[:, 0], "r-", label = "x")
        plt.plot(ts, self.trace[:, 1], "b-", label = "y")
        plt.plot(ts, self.trace[:, 2], "g-", label = "phi")
        plt.legend()
        plt.grid()
        plt.show()


# ### AC网络

# In[4]:


def build_actor(nodes = [10], max_action = None):
    inputs = tfk.layers.Input(shape = (5,), name = "state_input")
    h = tfk.layers.Dense(nodes[0], activation = "relu")(inputs)
    for n in nodes[1:]:
        h = tfk.layers.Dense(n, activation = "relu")(h)
    if max_action:
        outputs = max_action * tfk.layers.Dense(2, activation = "tanh")(h)
    else:
        outputs = tfk.layers.Dense(2, activation = "linear")(h)
    model = tfk.Model(inputs = inputs, outputs = outputs)
    #model.compile(loss = "mse", optimizer = "adam")
    return model


# In[5]:


def build_critic(nodes = [10]):
    sinputs = tfk.layers.Input(shape = (5,), name = "state_input")
    ainputs = tfk.layers.Input(shape = (2,), name = "action_input")
    h = tf.concat([sinputs, ainputs], axis = 1)
    for n in nodes:
        h = tfk.layers.Dense(n, activation = "relu")(h)
    outputs = tfk.layers.Dense(1, activation = "linear")(h)
    model = tfk.Model(inputs = [sinputs, ainputs], outputs = outputs)
    model.compile(loss = "mse", optimizer = "adam")
    return model


# ### 经验池

# In[6]:


class Replay_Buffer():
    
    def __init__(self, max_size = 1000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0
        
    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr+1)%self.max_size
        else:
            self.storage.append(data)
    
    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size = batch_size)
        x, y, u, r, d = [], [], [], [], []
        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy = False))
            y.append(np.array(Y, copy = False))
            u.append(np.array(U, copy = False))
            r.append(np.array(R, copy = False))
            d.append(np.array(D, copy = False))
        return np.array(x, np.float32), np.array(y, np.float32), np.array(u, np.float32), np.array(r, np.float32), np.array(d, np.float32)


# ### DDPG

# In[7]:


class DDPG():
    
    def __init__(
        self, nodes_actor = [10], nodes_critic = [10], max_action = None, 
        buffer_size = 10000, batch_size = 128, 
        update_iteration = 10, gamma = 0.95, tau = 0.005
    ):
        self.actor = build_actor(nodes_actor, max_action = max_action)
        self.actor_target = build_actor(nodes_actor, max_action = max_action)
        self.actor_target.set_weights(self.actor.get_weights())
        self.actor_opt = tfk.optimizers.Adam()
        
        self.critic = build_critic(nodes_critic)
        self.critic_target = build_critic(nodes_critic)
        self.critic_target.set_weights(self.critic.get_weights())
        
        self.replay_buffer = Replay_Buffer(buffer_size)
        self.batch_size = batch_size
        self.update_iteration = update_iteration
        self.gamma = gamma
        self.tau = tau
    
    def update(self):
        for it in range(self.update_iteration):
            x, y, u, r, d = self.replay_buffer.sample(self.batch_size)
            state = tf.constant(x)
            next_state = tf.constant(y)
            action = tf.constant(u)
            reward = tf.constant(r)
            done = tf.constant(d)
            
            target_Q = self.critic_target([next_state, self.actor_target(next_state)])
            target_Q = reward + ((1-done)*self.gamma*target_Q)
            self.critic.train_on_batch([state, action], target_Q)
            
            with tf.GradientTape(watch_accessed_variables = False) as tape:
                tape.watch(self.actor.variables)
                actor_loss = -tf.reduce_mean(self.critic([state, self.actor(state)]))
            grads = tape.gradient(actor_loss, self.actor.variables)
            self.actor_opt.apply_gradients(zip(grads, self.actor.variables))
            
            self.critic_target.set_weights([
                self.tau*w_new+(1-self.tau)*w_old for (w_new, w_old) in 
                zip(self.critic.get_weights(), self.critic_target.get_weights())
            ])
            self.actor_target.set_weights([
                self.tau*w_new+(1-self.tau)*w_old for (w_new, w_old) in 
                zip(self.actor.get_weights(), self.actor_target.get_weights())
            ])


# ### 实战

# In[8]:


env = Env()
while True:
    s, r, d = env.step(*np.random.normal(size = 2))
    if d > 0.5:
        break
env.show()


# In[9]:


def train(env, ddpg, num_episodes = 2000, interupt = 100, noise = 1.):
    for epi in range(num_episodes):
        state = env.reset()
        while True:
            action = ddpg.actor(state.reshape(1, -1)).numpy().flatten()
            action += np.random.normal(0, noise/(epi+1), size = 2)
            next_state, reward, done = env.step(*action)
            ddpg.replay_buffer.push((state, next_state, action, reward, done))
            state = next_state
            if len(ddpg.replay_buffer.storage) > ddpg.replay_buffer.max_size:
                ddpg.update()
            if done > 0.5:
                break
        ddpg.update()
        if epi % interupt == 0:
            print("Episode {}".format(epi))
            # ddpg.actor.save("./ddpg/actor_{}".format(epi), save_format = "tf")
            env.show()


# In[10]:


def test(env, ddpg):
    s = env.reset()
    while True:
        a = ddpg.actor(s.reshape(1, -1)).numpy().flatten()
        s, r, d = env.step(*a)
        if d > 0.5:
            break
    env.show()


# In[11]:


env = Env(delta_t = 0.1, end_t = 500, init = [10, 10, 0])
ddpg = DDPG(nodes_actor = [16], nodes_critic = [16], max_action = None, update_iteration = 10)
train(env, ddpg, 500, 10, noise = 1)


# In[12]:


test(env, ddpg)


# In[ ]:




