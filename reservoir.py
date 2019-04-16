#! /usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import networkx as nx
import json
import atexit
import os.path
from decimal import Decimal
from collections import OrderedDict
import datetime
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import pandas

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def f(state, t):
  x, y, z = state  # unpack the state vector
  return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # derivatives

state0 = [1.0, 1.0, 1.0]
t = np.arange(0.0, 400.0, 0.1)
#normalize data
states = odeint(f, state0, t)
states[:,0]=(states[:,0]-np.mean(states[:,0]))/np.std(states[:,0])
states[:,1]=(states[:,1]-np.mean(states[:,1]))/np.std(states[:,1])
states[:,2]=(states[:,2]-np.mean(states[:,2]))/np.std(states[:,2])

class Reservoir:
    def __init__(self):
        config_file_name = 'reservoir.config'
        global config
        with open(config_file_name) as config_file:
            config = json.load(config_file, object_pairs_hook=OrderedDict)

        # Input layer
        self.M = config["input"]["nodes"]
        self.input_len = config["input"]["length"]
        self.input_func = []
        dataset = []
        for i in range(self.M):
            self.input_func.append(eval(config["input"]["functions"][i]))
            dataset.append(self.input_func[i](np.arange(self.input_len) / self.input_len))
        self.dataset = np.array(list(zip(*dataset))).T

        # Reservoir layer
        self.start_node = config["reservoir"]["start_node"]
        self.N = self.start_node
        self.step = config["reservoir"]["step"]
        self.end_node = config["reservoir"]["end_node"]
        self.degree_func = eval(config["reservoir"]["degree_function"])
        self.D = self.degree_func(self.start_node)
        self.sigma = config["reservoir"]["sigma"]
        self.bias = config["reservoir"]["bias"]
        self.alpha = config["reservoir"]["leakage_rate"]
        self.beta = config["reservoir"]["regression_parameter"]

        # Output layer
        self.P = config["output"]["nodes"]

        # Training relevant
        self.init_len = config["training"]["init"]
        self.train_len = config["training"]["train"]
        self.test_len = config["training"]["test"]
        self.error_len = config["training"]["error"]


    def train(self):
        # collection of reservoir state vectors
        self.R = np.zeros(
            (1 + self.N + self.M, self.train_len - self.init_len))
        # collection of input signals
        self.S = np.vstack((x[self.init_len + 1: self.train_len + 1] for x in self.dataset))
        self.r = np.zeros((self.N, 1))
        np.random.seed(42)
        self.Win = np.random.uniform(-self.sigma,
                                     self.sigma, (self.N, self.M + 1))
        # TODO: the values of non-zero elements are randomly drawn from uniform dist [-1, 1]
        g = nx.erdos_renyi_graph(self.N, self.D / self.N, 42, True)
        self.A = nx.adjacency_matrix(g).todense()
        # spectral radius: rho
        self.rho = max(abs(np.linalg.eig(self.A)[0]))
        self.A *= 1.2 / self.rho
        # run the reservoir with the data and collect r
        for t in range(self.train_len):
            u = np.vstack((x[t] for x in self.dataset))
            self.r = (1 - self.alpha) * self.r + self.alpha * \
                     np.tanh(np.dot(self.A,self.r) + np.dot(self.Win, np.vstack((self.bias, u))))
            if t >= self.init_len:
                self.R[:, [t - self.init_len]
                       ] = np.vstack((self.bias, u, self.r))[:, 0]
        # train the output
        R_T = self.R.T  # Transpose
        self.Wout = np.dot(np.dot(self.S, R_T), np.linalg.inv(
            np.dot(self.R, R_T) + self.beta * np.eye(self.M + self.N + 1)))

    def _run(self):
        # run the trained ESN in alpha generative mode. no need to initialize here,
        # because r is initialized with training data and we continue from there.
        self.S = np.zeros((self.P, self.test_len))
        u = np.vstack((x[self.train_len] for x in self.dataset))
        for t in range(self.test_len):
            self.r = (1 - self.alpha) * self.r + self.alpha * \
                     np.tanh(np.dot(self.A,self.r) + np.dot(self.Win, np.vstack((self.bias, u))))
            s = np.dot(self.Wout, np.vstack((self.bias, u, self.r)))
            self.S[:, t] = np.squeeze(np.asarray(s))
            # use output as input
            u = s
        # compute error of each state
        self.RMS = []
        for i in range(self.P):
          self.RMS.append(sum(np.square(
            self.dataset[i, self.train_len+1: self.train_len+self.error_len+1] -
            self.S[i, 0: self.error_len])) / self.error_len)
        #file=open('error.txt','a')
        #file.write(str(self.alpha)+','+str(self.sigma)+','+str(self.RMS[0])+','+
        # str(self.RMS[1])+','+str(self.RMS[2]) + '\n')
        #file.close()
    def draw(self):
        plt.subplots(1, self.M)
        for i in range(self.M):
            ax = plt.subplot(1, self.M, i + 1)
            plt.text(0.5, -0.1, 'error = %.2e' % self.RMS[i], size=10, ha="center",
                     transform=ax.transAxes)
            plt.plot(self.S[i], label = 'prediction')
            plt.plot(self.dataset[i][self.train_len + 1 : self.train_len + self.test_len + 1],
                     label = 'input signal')
            plt.title('state'+str(i))
            plt.legend(loc = 'upper right')
        savefile = 'state' + 'C' + str(self.input_len) + 'sigma' + str(self.sigma) + 'alpha' +\
                   str(self.alpha) + 'RF' + str(self.beta) + '.png'
        plt.savefig(savefile)
        #plt.show()
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(self.S[0], self.S[1], self.S[2],label='predict orbit')
        ax.plot(self.dataset[0][self.train_len + 1 : self.train_len + self.test_len + 1],
                self.dataset[1][self.train_len + 1 : self.train_len + self.test_len + 1],
                self.dataset[2][self.train_len + 1 : self.train_len + self.test_len + 1],label
      	  ='actual orbit')
        ax.legend(loc='upper right')
        savefile = 'oribt'+'C'+str(self.input_len)+'sigma'+str(self.sigma)+'alpha'+\
                   str(self.alpha) +'RF'+str(self.beta)+ '.png'
        plt.savefig(savefile)
        #plt.show()
        indexes_pred=[]
        indexes_true=[]
        for i in range(2,len(self.S[2])-1):
            if self.S[2][i]>self.S[2][i-1] and self.S[2][i]>self.S[2][i+1]:
                indexes_pred.append(self.S[2][i])
        for i in range(2,len(self.dataset[2][self.train_len + 1 : self.train_len +
                                                                  self.test_len + 1])-2):
            if self.dataset[2][self.train_len + i]>self.dataset[2][self.train_len + i-1] and \
                    self.dataset[2][self.train_len + i]>self.dataset[2][self.train_len + i+1]:
                indexes_true.append(self.dataset[2][self.train_len + i])
        fig=plt.figure()
        for i in range(len(indexes_pred)-1):
            plt.plot(indexes_pred[i],indexes_pred[i+1],'ro')
        for i in range(len(indexes_true)-1):
            plt.plot(indexes_true[i],indexes_true[i+1],'bo')
        plt.xlabel('z(i)')
        plt.ylabel('z(i+1)')
        file='Lorenz map'
        plt.savefig(file)
        #plt.show()
    def run(self):
            for i in range(self.start_node, self.end_node + 1, self.step):
                self.N = i
                self.D = self.degree_func(self.N)
                self.train()
                self._run()
                config["reservoir"]["start_node"] = i
                self.draw()

if __name__ == '__main__':
    r = Reservoir()
    r.run()
