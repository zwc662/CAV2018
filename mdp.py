import numpy as np
import random
import scipy.optimize as optimize
from cvxopt import matrix, solvers
from scipy import sparse
import os
import ast
import time
import mdptoolbox
import math
import mdptoolbox 


class mdp():
    def __init__(self, num_S = None, num_A = None):
        if num_S is None or (num_A is None):
            self.input()
            num_S = len(self.S)
            num_A = len(self.A)
        self.S = range(num_S)
        # Always add two extra states at the end
        # S[-1] is the single unsafe terminal and S[-2] is the sngle initial terminal
        # All unsafe states have probability 1 to reach the unsafe terminal
        # From initial terminal there is a distribution of transiting to the
        # initial states
        self.A = range(num_A)
        # List of actions
        self.T = None
        # A list of numpy transition matrices for each actions
        ## [T(_, a_0, _), T(_, a_1, _), ...]
        self.P = None
        # The transition probability of DTMC given a policy
        self.policy = None
        # Policy is a |S|x|A| matrix of distributions actions for each states
        self.targets = list()
        # A list of absorbing target state to be reached, may be empty
        self.starts = list()
        # A list of initial states
        self.unsafes = list()
        # A list of unsafe states
        self.theta = None
        self.features = None
        # Features of all states
        self.rewardss = None
        # Rewards of all states
        self.epsilon = 1e-5
        self.max_iter = 10000
        self.discount = 0.99


    def set_initial_transitions(self, starts = None, distribution = None):
        # Set uniform distribution of starting from each initial states
        if starts is None:
            starts = self.starts

        for a in range(len(self.A)):
            self.T[a][:, self.S[-2]] = self.T[a][:, self.S[-2]] * 0.0
            self.T[a][self.S[-2]] = self.T[a][self.S[-2]] * 0.0

            if distribution is None:
                for s in self.starts:
                    self.T[a][self.S[-2], s] = 1.0/len(starts)
            else:
                for s in self.S:
                    self.T[a][self.S[-2], s] = distribution[s]
        

    def set_targets_transitions(self, targets = None):
        # Set target states to be absorbing
        if targets is None:
            targets = self.targets
        for a in range(len(self.A)):
            for t in targets:
                self.T[a][t] = self.T[a][t] * 0.0
                self.T[a][t, t] = 1.0

    def set_unsafes_transitions(self, unsafes = None):
        # Set all probabilities of transitioning from unsafe states to unsafe
        # terminal to be 1
        if unsafes is None:
            unsafes = self.unsafes

        for a in range(len(self.A)):
            self.T[a][:, self.S[-1]] = self.T[a][:, self.S[-1]] * 0.0
            self.T[a][self.S[-1]] = self.T[a][self.S[-1]] * 0.0
            self.T[a][self.S[-1], self.S[-1]] = 1.0
            for u in unsafes:
                self.T[a][u] = self.T[a][u] * 0.0
                #self.T[a][u, self.S[-1]] = 1.0
                self.T[a][u, u] = 1.0

    def set_features(self, num_features = 50): 
        self.features = (np.zeros([len(self.S),num_features]).astype(np.float16))      
        centroids = list()
        for i in range(num_features):
            centroids.append(i * len(self.S)/num_features)
            self.features[:, i] = [math.exp(-abs(j - centroids[-1]) * num_features/len(self.S)) for j in self.S]   
	self.features[-2] = 0.0 * self.features[-2]
        self.features[-1] = 0.0 * self.features[-1]
        print("Feature set up")  


    def read_transitions_file(self, transitions):
        # Count the times of transitioning from one state to another
        # Calculate the probability
        # Give value to self.T
        file = open(str(transitions), 'r')
        temp = [0, 0, 0, 0]
        for line_str in file.readlines():
            line = line_str.split('\n')[0].split(' ')
            t = int(float(line[0]))
            a = int(float(line[2]))
            s = int(float(line[1]))
            s_ = int(float(line[3]))
            if t == 0:
                for aa in self.A:
                    self.T[aa][self.S[-2]][s] += 1       
                self.T[a][s, s_] += 1
                temp = [0, s, a, s_]
            else:
                if [s, a, s_] == temp[1:]:
                    if temp[0] < 0:
                        temp[0] += 1
                    else:
                        self.T[a][s, s_] += 1   
                        temp[0] = 0
                else:
                    self.T[a][s, s_] += 1
                    temp = [0, s, a, s_]
        file.close()
        #self.set_unsafes()
        #self.set_targets()

        for a in range(len(self.A)):
            for s in range(len(self.S)):
                tot = np.sum(self.T[a][s])
                if tot == 0.0:
                    self.T[a][s,s ] = 0.0
            self.T[a] = sparse.bsr_matrix(self.T[a])        
            self.T[a] = sparse.diags(1.0/self.T[a].sum(axis = 1).A.ravel()).dot(self.T[a])


    def set_transitions_random(self):
        # Count the times of transitioning from one state to another
        # Calculate the probability
        # Give value to self.T
        self.T = list()
        for a in range(len(self.A)):
            self.T.append(sparse.random(len(self.S), len(self.S), density = 0.1).todense())
            for s in range(len(self.S)):
                tot = np.sum(self.T[a][s])
                if tot == 0.0:
                    self.T[a][s,s] = 1.0
            self.T[a] = sparse.bsr_matrix(self.T[a])
            self.T[a] = sparse.diags(1.0/self.T[a].sum(axis = 1).A.ravel()).dot(self.T[a])

    def set_policy_random(self):
        self.policy = np.random.random((len(self.S), len(self.A))) 
        self.policy = self.policy/np.reshape(np.linalg.norm(self.policy, axis = 1, ord = 1), [len(self.S), 1])
        self.P = sparse.bsr_matrix(np.zeros([len(self.S), len(self.S)], dtype=float))
        '''
        for a in range(len(self.A)):
            self.P += self.T[a].dot(sparse.bsr_matrix(np.repeat(np.reshape(self.policy.T[a], [len(self.S), 1]), len(self.S), axis = 1 )))
        self.P = sparse.diags(1.0/self.P.sum(axis = 1).A.ravel()).dot(self.P)
        '''
        for a in self.A:
            if isinstance(self.T[a], np.ndarray) is False:
                self.T[a] = self.T[a].todense()
            policy = np.reshape(self.policy.T[a], (len(self.S), 1))
            self.P += np.multiply(self.T[a], policy)
        self.P = sparse.bsr_matrix(self.P)
        self.P = sparse.diags(1.0/self.P.sum(axis = 1).A.ravel()).dot(self.P)
        
        print("DTMC transition constructed")
        return self.policy

    def set_policy(self, policy = None):
        if policy is None:
            pass  
        elif isinstance(policy, sparse.csr_matrix) or isinstance(policy, sparse.csc_matrix):
            self.policy = policy.todense()
        else:
            self.policy = policy
        assert self.policy.shape == (len(self.S), len(self.A))

        for s in self.S:
            p_tot = self.policy[s].sum()
            a_max = self.policy[s].argmax()
            if p_tot < 1.0:
                self.policy[s, a_max] += 1.0 - p_tot
                 
            p_tot = self.policy[s].sum()
            if p_tot < 1.0:
                print(self.policy[s])

        self.P = sparse.bsr_matrix(np.zeros([len(self.S), len(self.S)], dtype=np.float64))
        '''
        for a in range(len(self.A)):
            self.P += self.T[a].dot(sparse.bsr_matrix(np.repeat(np.reshape(self.policy.T[a], [len(self.S), 1]), len(self.S), axis = 1 )))
        self.P = sparse.diags(1.0/self.P.sum(axis = 1).A.ravel()).dot(self.P)
        '''
        
        for a in self.A:
            if isinstance(self.T[a], np.ndarray) is False:
                self.T[a] = self.T[a].todense()
            policy = np.reshape(self.policy.T[a], (len(self.S), 1))
            self.P += np.multiply(self.T[a], policy)
        self.P = sparse.bsr_matrix(self.P)
        self.P = sparse.diags(1.0/self.P.sum(axis = 1).A.ravel()).dot(self.P)
        
        print("DTMC transition constructed")

    def output(self):
        # Output files for PRISM
        os.system('rm ./data/state_space')
        os.system('touch ./data/state_space')
        file = open('./data/state_space', 'w')
        file.write('states\n' + str(len(self.S)) + '\n')
        file.write('actions\n' + str(len(self.A)) + '\n')
        file.close()

        os.system('rm ./data/unsafe')
        os.system('touch ./data/unsafe')
        file = open('./data/unsafe', 'w')
        for i in range(len(self.unsafes)):
            file.write(str(self.unsafes[i]) + '\n')
        file.close()

        os.system('rm ./data/start')
        os.system('touch ./data/start')
        file = open('./data/start', 'w')
        for i in range(len(self.starts)):
            file.write(str(self.starts[i]) + '\n')
        file.close()

        os.system('rm ./data/mdp')
        os.system('touch ./data/mdp')
        file = open('./data/mdp', 'w')
        for s in range(len(self.S)):
            for a in range(len(self.A)):
                for s_ in range(len(self.S)):
                    file.write(str(self.S[s]) + ' ' + str(self.A[a]) + ' ' + str(self.S[s_]) + ' ' + str(self.T[self.A[a]][self.S[s], self.S[s_]]) + '\n')
        file.close()

    def input(self):
        file = open('./data/state_space', 'r')
        lines = file.readlines()
        for i in range(len(lines)):
            if lines[i].split('\n')[0] == 'states':
                self.S = range(int(lines[i + 1].split('\n')[0]))
            if lines[i].split('\n')[0] == 'actions':
                self.A = range(int(lines[i + 1].split('\n')[0]))
        file.close()

        self.unsafes = list()
        file = open('./data/unsafe', 'r')
        lines = file.readlines()
        for line in lines:
            self.unsafes.append(int(line.split('\n')[0]))
        file.close()


        self.starts = list()
        file = open('./data/start', 'r')
        lines = file.readlines()
        for line in lines:
            self.starts.append(int(line.split('\n')[0]))
        file.close()

        self.T = list()
        for a in self.A:
            self.T.append(np.zeros([len(self.S), len(self.S)]))

        file = open('./data/mdp', 'r')
        lines = file.readlines()
        for line in lines:
            trans = line.split('\n')[0].split(' ')
            self.T[int(trans[1])][int(trans[0]), int(trans[2])] = float(trans[-1])
        file.close()

    def move(self, state, action):
        prob = random.random()
        for s in self.S:
            prob -= self.T[action][state, s]
            if prob <= 0:
                return s
        return state

                


    def expected_features_manual(self, discount = None, epsilon = None, max_iter = None):
        if discount is None:
            discount = self.discount
        if epsilon is None:
            epsilon = self.epsilon
        if max_iter is None:
            max_iter = self.max_iter

	itr = 0
	mu = self.features.copy()
	diff = float('inf')
	assert self.P.shape[1] == self.features.shape[0]
	assert self.P.shape[0] == self.features.shape[0]
	while diff > epsilon:
		itr += 1	
		mu_temp = mu.copy()
		mu = self.features + discount * (self.P.dot(mu))
                diff = np.linalg.norm(mu - mu_temp, axis = 1, ord = 2).max()
	print("Expected features manually calculated")
        return mu
	
    def expected_features(self, discount = None, epsilon = None, max_iter = None):
        if discount is None:
            discount = self.discount
        if epsilon is None:
            epsilon = self.epsilon
        if max_iter is None:
            max_iter = self.max_iter
	mu = []
	for f in range(self.features.shape[-1]):
		V = self.features[:, f].reshape(len(self.S))
		VL = mdptoolbox.mdp.ValueIteration(np.array([self.P]), V, discount, epsilon, max_iter, initial_value = 0)
		VL.run()
	        mu.append(VL.V)
	mu =  np.array(mu).T
        print("Expected features calculated")
        return mu

    def expected_value_manual(self, discount = None, epsilon = None, max_iter = None):
        if discount is None:
            discount = self.discount
        if epsilon is None:
            epsilon = self.epsilon
        if max_iter is None:
            max_iter = self.max_iter

        itr = 0
        v = self.rewards
        diff = float('inf')
        while diff > epsilon:
            itr += 1
	    print("Iteration %d, difference is %f" % (itr, diff))
	    v_temp = v
	    v = self.rewards + discount * (self.P.dot(v))
            diff = (abs((v_temp - v).max()) + abs((v_temp - v).min()))/2
	print("Expected value calculated")
        return v

			

    def expected_value(self, discount = None, epsilon = None, max_iter = None):
        if discount is None:
            discount = self.discount
        if epsilon is None:
            epsilon = self.epsilon
        if max_iter is None:
            max_iter = self.max_iter

	VL = mdptoolbox.mdp.ValueIteration(np.array([self.P]), self.rewards, discount, epsilon, max_iter, initial_value = 0)
	VL.run()
	print("Expected value calculated")
	return VL.V

    def value_iteration(self, discount = None, epsilon = None, max_iter = None):
        if discount is None:
            discount = self.discount
        if epsilon is None:
            epsilon = self.epsilon
        if max_iter is None:
            max_iter = self.max_iter

	VL = mdptoolbox.mdp.ValueIteration(np.array(self.T), self.rewards, discount, epsilon, max_iter, initial_value = 0)
	VL.run()
	policy = np.zeros([len(self.S), len(self.A)]).astype(float)
 	for s in range(len(VL.policy)):
		policy[s, VL.policy[s]] = 1.0
        print("Value iteration finished, optimal policy generated")
	return policy


    def optimal_policy(self, theta = None):
        if theta is None:
            theta = self.theta
        self.rewards = np.dot(self.features, theta)
 
        self.policy = self.value_iteration()
        self.set_policy(self.policy)
        mus = self.expected_features_manual() 
        return mus, self.policy
		
	
	
    def LP_value_scipy(self, epsilon = None, discount = None):
        if discount is None:
            discount = self.discount
        if epsilon is None:
            epsilon = self.epsilon

	if not isinstance(self.P, sparse.csr_matrix):
		self.P = sparse.csr_matrix(self.P)
	start = time.time()
	c = np.ones((len(self.S)))
    	A_ub = discount * self.P.transpose() - sparse.eye(len(self.S))
	assert A_ub.shape == (len(self.S), len(self.S))
	b_ub = -1 * self.rewards
	sol = optimize.linprog(c = c, A_ub = A_ub.todense(), b_ub = b_ub, method = 'simplex')
	end = time.time()
	print('Solving one expected value via sparse LP, time = %f' % (end - start))
	return np.reshape(np.array(sol['x']), (len(self.S)))
	

    def LP_value_cvxopt(self, epsilon = None, discount = None):
    	self.P = self.P.todense()
	assert self.P.shape == (len(self.S), len(self.S))
    	start = time.time()
    	c = np.ones((len(self.S))).tolist()
    	G = (discount * self.P.T - np.eye(len(self.S))).tolist()
    	h = (-1 * self.rewards).tolist()
    	sol = solvers.lp(matrix(c), matrix(G), matrix(h))
    	end = time.time()
    	print('Solving one expected value via LP, time = ' + str(end - start))
	return np.reshape(np.array(sol['x']), (len(self.S)))


    def LP_features_cvxopt(self, epsilon = None, discount = None):
        if epsilon is None:
            epsilon = self.epsilon
        if discount is None:
            dicount = self.discount
    	self.P = self.P.todense()
	assert self.P.shape == (len(self.S), len(self.S))
    	mu = []
    	for f in range(len(self.features[0])):
    		start = time.time()
    		c = np.ones((len(self.S))).tolist()
    		G = (discount * self.P.T - np.eye(len(self.S))).tolist()
    		h = (-1 * self.features[:, f]).tolist()
		print("Start solving feature %d..." % f)
    		sol = solvers.lp(matrix(c), matrix(G), matrix(h))
    		mu.append(np.array(sol['x']).reshape((len(self.S))))
		print("Finished solving feature %d..." % f)
    		end = time.time()
    		print('Solving one expected feature via LP, time = ' + str(end - start))
    	return mu
        
