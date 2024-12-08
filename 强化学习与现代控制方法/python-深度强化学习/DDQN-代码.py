#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

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
        self.target = np.array([0., 0., 0., 0., 0.])
        self.action_code = [
            [1, 1], [1, 0], [1, -1], 
            [0, 1], [0, 0], [0, -1], 
            [-1, 1], [-1, 0], [-1, -1]
        ]
        self.reset()
    
    def step(self, action):
        self.time += self.delta_t
        tau_p, tau_m = self.action_code[action]
        sol = solve_ivp(func, [0, self.delta_t], self.state, args = (tau_p, tau_m), t_eval = [self.delta_t])
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


# ### Q-net

# In[4]:


class Qnet(tfk.Model):
    
    def __init__(self, nodes = [10]):
        super().__init__()
        self.hiddens = [tfk.layers.Dense(n, activation = "relu") for n in nodes]
        self.outputs = tfk.layers.Dense(9, activation = "linear")
    
    def call(self, inputs):
        h = inputs
        for hidden in self.hiddens:
            h = hidden(h)
        return self.outputs(h)
    
    def predict(self, inputs):
        q_values = self(inputs)
        return tf.argmax(q_values, axis = -1)


# ### 经验池

# In[5]:


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
        return np.array(x, np.float32), np.array(y, np.float32), np.array(u, np.int32), np.array(r, np.float32), np.array(d, np.float32)


# ### DDQN

# In[6]:


class DDQN():
    
    def __init__(self, nodes = [10], buffer_size = 10000, batch_size = 128, 
        update_iteration = 10, gamma = 0.95, tau = 0.005
    ):  
        self.qnet = Qnet(nodes)
        self.opt = tfk.optimizers.Adam()
        self.qnet_target = Qnet(nodes)
        self.qnet_target.set_weights(self.qnet.get_weights())
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
            
            qact = self.qnet.predict(next_state)
            target_Q = reward + (1-done)*self.gamma*self.qnet_target(next_state).numpy()[range(len(reward)), qact.numpy()]
            with tf.GradientTape() as tape:
                loss = tfk.losses.mean_squared_error(
                    y_true = target_Q, 
                    y_pred = tf.reduce_sum(self.qnet(state)*tf.one_hot(action, depth = 9), axis = 1)
                )
            grads = tape.gradient(loss, self.qnet.variables)
            self.opt.apply_gradients(zip(grads, self.qnet.variables))
            
            self.qnet_target.set_weights([
                self.tau*w_new+(1-self.tau)*w_old for (w_new, w_old) in 
                zip(self.qnet.get_weights(), self.qnet_target.get_weights())
            ])


# ### 实战

# In[7]:


env = Env()
while True:
    s, r, d = env.step(np.random.randint(9))
    if d > 0.5:
        break
env.show()


# In[8]:


def test(env, ddqn):
    s = env.reset()
    while True:
        a = int(ddqn.qnet.predict(s.reshape(1, -1)))
        s, r, d = env.step(a)
        if d > 0.5:
            break
    env.show()


# In[9]:


ddqn = DDQN()
test(env, ddqn)


# In[10]:


def train(env, ddqn, num_episodes = 2000, interupt = 100, noise = 1.):
    for epi in range(num_episodes):
        state = env.reset()
        while True:
            if np.random.random() < noise/(epi+1):
                action = np.random.randint(9)
            else:
                action = int(ddqn.qnet.predict(state.reshape(1, -1)))
            next_state, reward, done = env.step(action)
            ddqn.replay_buffer.push((state, next_state, action, reward, done))
            state = next_state
            if len(ddqn.replay_buffer.storage) > ddqn.replay_buffer.max_size:
                ddqn.update()
            if done > 0.5:
                break
        ddqn.update()
        if epi % interupt == 0:
            print("Episode {}".format(epi))
            # ddqn.qnet.save("./ddqn/qnet_{}".format(epi), save_format = "tf")
            env.show()


# In[12]:


env = Env(delta_t = 0.1, end_t = 500, init = [10, 10, 0])
ddqn = DDQN(nodes = [16])
train(env, ddqn, num_episodes = 500, interupt = 10)


# In[13]:


test(env, ddqn)


# In[ ]:




