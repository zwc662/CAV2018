from mdp import mdp
import numpy as np
import scipy.optimize as optimize
from cvxopt import matrix, solvers
from scipy import sparse
import sys
import os
import ast
import time
import mdptoolbox
from mdp import mdp

solvers.options['show_progress'] = False

class apirl():
    def __init__(self, M = None, theta = None, max_iter = 30):
        self.M = M
        if theta is not None:
            self.theta = theta.copy()
        self.exp_mu = None
        if max_iter is not None:
            self.max_iter = max_iter
    
    def read_demo_file(self, paths):
        exp_mu = np.zeros(self.M.features[0].shape)
        mu_temp = exp_mu.copy()
        num_paths = 0
        s = self.M.S[-2]      
        t = 0
        t_ = 0

        avg = 0
        init_dist = np.zeros((len(self.M.S))) 

        file = open(str(paths), 'r')
        for line_str in file.readlines():
            line = line_str.split('\n')[0].split(' ')
            t = int(float(line[0]))
            if t == 0:
                avg += t_
                diff = float('inf')
                while diff > self.M.epsilon:
                    t_ += 1         
                    diff = self.M.features[s] * self.M.discount**t_
                    mu_temp += diff
                    diff = np.linalg.norm(diff, ord = 2)
                exp_mu = exp_mu + mu_temp       
                num_paths += 1

                mu_temp = self.M.features[-2].copy()
                t += 1

            s = int(float(line[1]))
            if t == 1:
                init_dist[s] += 1

            mu_temp = mu_temp + self.M.features[s] * (self.M.discount**t)
            t_ = t
        file.close()

        diff = float('inf')
        while diff > self.M.epsilon:
            t_ += 1         
            diff = self.M.features[s] * self.M.discount**t_
            mu_temp += diff
            diff = np.linalg.norm(diff, ord = 2)
        exp_mu += mu_temp

        exp_mu = exp_mu/num_paths
        #avg = avg/num_paths

        init_dist /= num_paths
        self.M.set_initial_transitions(init_dist)

        print("%d demonstrated paths in total" % num_paths)
        #print("Average step length is %d" % avg)
        print("Expert expected features are:")
        print(exp_mu)
        return exp_mu

    def random_demo(self):
        self.M.set_policy_random()
        mus = self.M.expected_features_manual()
        return mu[-2]

    def QP(self, expert, features, epsilon = None):
        if epsilon is None:
            epsilon = self.M.epsilon

	assert expert.shape[-1] == np.array(features).shape[-1]
	c = matrix(np.eye(len(expert) + 1)[-1] * -1)
	G_i = []
	h_i = []

	for k in range(len(expert)):
		G_i.append([0])	
	G_i.append([-1])
	h_i.append(0)

	for j in range(len(features)):
		for k in range(len(expert)):
			G_i[k].append( - expert[k] + features[j][k])	
		G_i[len(expert)].append(1)
		h_i.append(0)

	for k in range(len(expert)):
		G_i[k] = G_i[k] + [0.0] * (k + 1) + [-1.0] + [0.0] * (len(expert) + 1 - k - 1)
	G_i[len(expert)] = G_i[len(expert)] + [0.0] * (1 + len(expert)) + [0.0]
	h_i = h_i + [1] + (1 + len(expert)) * [0.0]

	G = matrix(G_i)
	h = matrix(h_i)

	dims = {'l': 1 + len(features), 'q': [len(expert) + 1, 1], 's': []}
	start = time.time()
	sol = solvers.conelp(c, G, h, dims)
	end = time.time()
	print("QP operation time = " + str(end - start))
	solution = np.array(sol['x'])
	if solution is not None:
		solution=solution.reshape([len(expert) + 1]).tolist()
		w = solution[0:-1]
		t = solution[-1]
	else:
		w = None
		t = None
	return w, t
    
    def iteration(self, exp_mu = None):
        if exp_mu is None:
            exp_mu = self.exp_mu.copy()
        features = list()

        print("Generating initial policy for AL")
        theta = np.ones((len(self.M.features[0])))
        theta = theta/np.linalg.norm(theta, ord = 2)
        print("Initial policy weight vector:")
        print(theta)

        mus, policy = self.M.optimal_policy(theta.copy())
        mu = mus[-2].copy()
        print("Initial policy feature vector:")
        print(mu)

        err = float('inf')
	
        itr = 0
        
        diff = np.linalg.norm(exp_mu - mu, ord = 2)
        print("Initial policy feature margin: %f" % diff)
        diff_ = 0.0

        opt = {'diff': diff, 
                'theta': theta.copy(), 
                'policy': policy.copy(), 
                'mu': mu.copy()} 
        features.append(mu.copy())
        
        print("\n>>>>>>>>APIRL iteration start, learn from:")
        print(exp_mu)
        print(">>>>>>>>>>Max iteration number: %d" % self.max_iter)
        print(">>>>>>>>>>epsilon: %f" % self.M.epsilon)
        while True:
            print("\n>>>>>>>>>Iteration %d" % itr)
            if diff <= self.M.epsilon:
                print(">>>>>>>>>>>Converge<<<<<<<<<<\
                        epsilon-close policy found" % diff)
                break
            
            if itr >= self.max_iter:      
                print("Reached maximum iteration. Return best learnt policy.")
                break
            if abs(diff - diff_) < self.M.epsilon:
                print("Reached local optimum. End iteration")
                break
            diff_ = diff

            itr += 1
            features.append(mu.copy())
            theta, err  = self.QP(exp_mu, features) 

            print("QP error: %f" % err)

            theta = theta/np.linalg.norm(theta, ord = 2)
            print("New candidate policy weight vector:")
            print(theta)

            mus, policy  = self.M.optimal_policy(theta)
            mu = mus[-2].copy()
            print("New candidate policy feature vector:")
            print(mu)

        
            diff = np.linalg.norm(mu - exp_mu, ord = 2)
            print("Feature margin: %f" % diff)

            if diff < opt['diff']:
                opt = {'diff': diff, 
                        'theta': theta.copy(), 
                        'policy': policy.copy(), 
                        'mu':mu.copy()} 
        	print("Update best learnt policy")

        if diff <= self.M.epsilon:
            print("\n<<<<<<epsilon-close policy is found. APIRL finished")
        else:
            print("\n<<<<<<Can't find espsilon-close policy. APIRL stop")

        print("Optimal policy weight vector:")
        print(opt['theta'])
        print("Optimal policy feature vector:")
        print(opt['mu'])
        print("Feature margin: %f" % opt['diff'])
        return opt

    def run(self, option = None):
        if True:
            if option is None:
                option = raw_input("Expectef features are from 1. human 2. optimal policy 3. random policy, 4. exit")
                option = int(option)

            if option == 4:
                return
            elif option == 1:
                self.exp_mu = self.read_demo_file()
            elif option == 2:
                mus, _  = self.M.optimal_policy()
                self.exp_mu = mus[-2]
            elif option == 3:
                self.exp_mu = self.random_demo()
            else:
                return
            opt = self.iteration()


        return opt
                
        
#if __name__ == "__main__":
#    AL = apprenticeship_Ming()
#    AL.run()
