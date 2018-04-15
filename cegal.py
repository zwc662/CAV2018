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
from apirl import apirl

import subprocess, shlex
from threading import Timer
import matplotlib
import pylab
import warnings
import random
import re

solvers.options['show_progress'] = False

class cegal(apirl, object):
    def __init__(self, M = None, theta = None, max_iter = 30, safety = None, steps = None):
        if sys.version_info[0] >= 3: 
            super().__init__()
        else:
            super(cegal, self).__init__(M, theta, max_iter)
        self.safety = safety
        self.steps = steps
        
    def MOQP(self, expert, features, K, epsilon = None):
        if epsilon is None:
            epsilon = self.M.epsilon

	cexs = features['cexs']
	
	cands = features['cands']

	safes = features['safes']
	
	if True:
		#G_i_j=[[], [], [], []]
		G_i_j_k = []
		for e in range(len(expert) + 2):
			G_i_j_k.append([])
		h_i_j_k = []
		c = matrix(- ((1000 * K) * np.eye(len(expert) + 2)[-2] + 1000 * (1 - K) * np.eye(len(expert) + 2)[-1]))

		for m in range(len(cands)):
			for e in range(len(expert)):
				G_i_j_k[e].append(K * (- expert[e] + cands[m][e]))
			G_i_j_k[len(expert)].append(1)
			G_i_j_k[len(expert) + 1].append(0)
			h_i_j_k.append(0)
					#G_i_j[0].append(- (expert[0] - cands[m][0]))
					#G_i_j[1].append(- (expert[1] - cands[m][1]))
					#G_i_j[2].append(- (expert[2] - cands[m][2]))
					#G_i_j[3].append(- (expert[3] - cands[m][3]))
		for j in range(len(cexs)):
			for k in range(len(safes)):
				for e in range(len(expert)):
					G_i_j_k[e].append((1.0 - K) * (- safes[k][e] + cexs[j][e]))
				G_i_j_k[len(expert)].append(0)
				G_i_j_k[len(expert) + 1].append(1)
				h_i_j_k.append(0)
						#G_i_j[0].append(cands[k][0] - mu_Bs[j][0] - (cands[l][0] - mu_Bs[n][0]))
						#G_i_j[1].append(cands[k][1] - mu_Bs[j][1] - (cands[l][1] - mu_Bs[n][1]))
						#G_i_j[2].append(cands[k][2] - mu_Bs[j][2] - (cands[l][2] - mu_Bs[n][2]))
						#G_i_j[3].append(cands[k][3] - mu_Bs[j][3] - (cands[l][3] - mu_Bs[n][3]))
						#h_i_j.append(0)
				
		for e in range(len(expert)):
			G_i_j_k[e] = G_i_j_k[e] + [0.0] * (e + 1) + [-1.0] + [0.0] * (len(expert) + 2 - e - 1)
		G_i_j_k[len(expert)] = G_i_j_k[len(expert)] + [0.0] * (len(expert) + 3)
		G_i_j_k[len(expert) + 1] = G_i_j_k[len(expert) + 1] + [0.0] * (len(expert) + 3)
	        h_i_j_k = h_i_j_k + [1000.0] + (2 + len(expert)) * [0.0]
				#G_i_j[0]= G_i_j[0] + [0., -1., 0., 0., 0.]
				#G_i_j[1]= G_i_j[1] + [0., 0., -1., 0., 0.]
				#G_i_j[2]= G_i_j[2] + [0., 0., 0., -1., 0.]
				#G_i_j[3]= G_i_j[3] + [0., 0., 0., 0., -1.]
				#h_i_j = h_i_j + [1., 0., 0., 0., 0.]
		G = matrix(G_i_j_k)
		h = matrix(h_i_j_k)
				#G = matrix(G_i_j)
			#	h = matrix([-1 * penalty, 1., 0., 0., 0.])
				#h = matrix(h_i_j)
		dims = {'l':  len(cands) + len(cexs) * len(safes), 'q': [2 + len(expert), 1], 's': []}
		sol = solvers.conelp(c, G, h, dims)
		sol['status']
		solution = np.array(sol['x'])
		if solution is not None:
			solution=solution.reshape(len(expert) + 2)
			t=(1 - K) * solution[-1] + K * solution[-2]
			w=solution[:-2]
                        t /= np.linalg.norm(w, ord = 2)
		else:
			solution = None
			t = None
			w = None
	return w, t
    
    def initialization(self, safety = None, steps = None):
        if safety is None:
            safety = self.safety
        if steps is None:
            steps = self.steps

        features = {'cexs': [], 'cands': [], 'safes': []}
        features['cands'].append(np.zeros(self.M.features[-1].shape).astype(float))
        features['safes'].append(np.zeros(self.M.features[-1].shape).astype(float))
        expert = np.zeros(self.M.features[-1].shape).astype(float)

        theta = np.random.random((len(self.M.features[0])))
        theta = theta/np.linalg.norm(theta, ord = 2)
        mus, policy = self.M.optimal_policy(theta)
        mu = mus[-2].copy()
    
        while True:
            cex, prob = self.verify(policy, mus, safety, steps)
            if prob < safety:
                opt = {'diff': float('inf'), 
                        'theta': theta, 
                        'policy': policy, 
                        'mu':mu, 
                        'prob': prob} 
                print(theta)
                '''
                file = open('init_cartpole', 'w')
                for s in self.M.S:
                    for a in self.M.A:
                        file.write(str(policy[s, a]) + ' ')
                    file.write('\n')
                file.close()
                '''
                return opt 
            if len(features['cexs']) > 1:
                if np.linalg.norm(features['cexs'][-1] - cex, ord = 2) < self.M.epsilon:
                    print("Can't find more counterexample. Return None.")
                    return None
            
            if cex is not None:
                features['cexs'].append(cex)     
        
            theta, _  = self.MOQP(expert = expert, features = features, K = 0)
            theta = theta/np.linalg.norm(theta, ord = 2)
            mus, policy = self.M.optimal_policy(theta)
            mu = mus[-2].copy()
             
        


    def iteration(self, exp_mu = None, opt = None, safety = None, steps = None):
        if exp_mu is None:
            exp_mu = self.exp_mu

        if safety is None:
            safety = self.safety

        if steps is None:
            steps = self.steps

        features = {'cexs': [], 'cands': [], 'safes': []}

        if opt is not None:
            print("Verify provided policy")
            theta = np.array(opt['theta'])
            policy = opt['policy'].copy()
            mu = opt['mu'].copy()
            self.M.set_policy(policy)
            mus = self.M.expected_features_manual().copy()
            cex, prob = self.verify(policy, mus, safety, steps)
            opt['prob'] = prob
            if prob < safety:
                print("Provided policy is safe. Use as initial safe policy")
                features['cands'].append(mu.copy())
                features['safes'].append(mu.copy())
            else:
                print("Provided policy is unsafe. Generating initial safe policy")        
                if cex is not None:
                    features['cexs'].append(cex.copy())
                opt = None

        if opt is None:
            print("Initial safe policy is not provided. Generating initial safe policy")
            opt = self.initialization(safety = safety, steps = steps)
            if opt is None:
                print("Failed to find a safe policy")
                return None
        print("Initial safe policy is generated.")
        theta = np.array(opt['theta'])
        policy = opt['prob']
        mu = opt['mu'].copy()
        diff = np.linalg.norm(mu - exp_mu, ord = 2)
        opt['diff'] = diff

        features['cands'].append(mu.copy())
        features['safes'].append(mu.copy())

        
	INF = 0.0
	SUP = 1.0
	K = 1.0

        print("Run apprenticeship learning to start iteration.")
        opt_ = super(cegal, self).iteration(exp_mu) 
        theta = np.array(opt_['theta'])
        policy = opt_['policy'].copy()
        mu = opt_['mu'].copy()
        self.M.set_policy(policy)
        mus = self.M.expected_features_manual().copy()
        cex, prob = self.verify(policy, mus, safety, steps)
        opt_['prob'] = prob
        if prob < safety:
            print("Apprenticeship learning policy is safe. Return policy.")
            print(theta)
            return opt_, opt_ 
        else:
            print("Appenticeship learning policy is unsafe. Start iteration.")        
            if cex is not None:
                features['cexs'].append(cex.copy())

        err = 0
        itr = 0
        diff_ = float('inf')
        QP_err = 0
        
        print("\n>>>>>>>>SAAL iteration start. Expert feature vector:")
        print(exp_mu)
        print(">>>>>>>>>>Max iteration number: %d" % self.max_iter)
        print(">>>>>>>>>>epsilon: %f" % self.M.epsilon)
        print(">>>>>>>>>>Safety constraint: %f" % safety)
        while True:
            print("\n>>>>>>Iteration %d, parameter K = %f, INF = %f<<<<<<<<<<<\n" % (itr, K, INF))    
            if itr >= self.max_iter:      
                print("Reached maximum iteration. Return best learnt policy.")

                break

            print("\n>>>>>>>>>Lastly learnt policy weight vector:")
            print(theta)
            
           
            if INF == K and abs(diff - diff_) < self.M.epsilon:
                #print("Stuck in local optimum. End iteration")
                #K = (K + INF)/2.0
                #break
                if INF == K:
                    print("Stuck in local optimum of AL. Return best learnt policy.")
                    return opt, opt_
            diff_ = diff

            itr += 1
            
             
                        
            if itr > 1:
                cex, prob = self.verify(policy = policy, mus = mus, safety = safety, steps = steps)
            if prob <= safety: 
                print("\n>>>>>>>Lastly learnt policy is verified to be safe<<<< %f\n" % prob)
                diff = np.linalg.norm(mu - exp_mu, ord = 2)
                print("Feature margin: %f" % diff)
                if diff <= self.M.epsilon:
                    print("\n>>>>>>>>>>>Converge<<<<<<<<<epsilon-close policy is found. Return.\n")
                    opt = {'diff': diff, 
                            'theta': np.array(theta), 
                            'policy': policy.copy(), 
                            'mu':mu.copy(), 
                            'prob': prob} 
                    return opt, opt_
                elif diff <= opt['diff']:
                    opt = {'diff': diff, 
                            'theta': np.array(theta), 
                            'policy': policy.copy(), 
                            'mu':mu.copy(), 
                            'prob': prob} 
                    print(">>>>>>>>>>>>>Update best policy weight vector:")
                    print(theta)
                    
                else:
                    features['cands'].append(mu.copy())
                    features['safes'].append(mu.copy())

                if True or K != SUP:
                    INF = K
                K = SUP

            if prob > safety:
                print("\n>>>>>>>Lastly verified policy is verified to be unsafe<<<< %f\n" % prob)
                print("Counterexample generated.")
                if cex is not None:
                    features['cexs'].append(cex.copy())     
                #features['cands'].append(mu)
        
                K = (K + INF)/2.0

                if abs(K - INF) < self.M.epsilon:
                    print("\n>>>>>>>>>>>Converge<<<<<<<<K is too close to INF.\n")
                    return opt, opt_
            theta, _  = self.MOQP(expert = exp_mu, features = features, K = K)
            '''
                if QP_err > 10:
                    print("\nXXXXXXXXXXXXQP numerical errorXXXXXXXXX\n")
                    return opt, opt_
                else:
                    QP_err += 1
            '''
        
            theta = theta/np.linalg.norm(theta, ord = 2)
            mus, policy = self.M.optimal_policy(theta)
            mu = mus[-2].copy()

        return opt, opt_

    def set_unsafes_transitions(self, P):
        P[-1] = 0.0 * P[-1]
        P[-1, -1] = 1.0
        for s in self.M.unsafes:
            P[s] = 0.0 * P[s]
            P[s, -1] = 1.0
        return P

    def check_transitions(self, P):
        #for s in self.M.S:
        #    p_tot = P[s].sum()
            #while p_tot > 0.99:
                    #P[s, self.M.S[-2]] = 0.001
            #    P[s] = P[s] * 0.99
            #    p_tot = P[s].sum()
            #if p_tot == 0.0:
            #    P[s, s] = 1.0
        P = self.set_unsafes_transitions(P)
        return P


    def write_policy_file(self, policy = None,  path = './data/optimal_policy'):
        if policy is None:
            policy = self.M.policy.copy()
        self.M.set_policy(policy)

        P = self.M.P.copy()
        if isinstance(P, np.ndarray) is False: 
            P = P.todense()
        #P = self.set_unsafes_transitions(P)
        P = self.check_transitions(P)

        file = open(path, 'w')
        for s in self.M.S:
            for s_ in self.M.S:
               file.write(str(s) + ' ' + str(s_) + ' ' + str(P[s, s_]) + '\n')
        file.close()



    def write_conf_file(self, safety = None):
        if safety is None:
            safety = self.safety
	file = open('grid_world.conf', 'w')
	file.write('TASK counterexample\n')
	file.write('PROBABILITY_BOUND ' + str(safety) + '\n')
	file.write('DTMC_FILE ./grid_world.dtmc' + '\n')
	file.write('REPRESENTATION pathset' + '\n')
	file.write('SEARCH_ALGORITHM global' + '\n')
	file.write('ABSTRACTION concrete' + '\n')
	file.close()

    def model_check(self, policy = None, steps = None):
        if policy is None:
            policy = self.M.policy.copy()
        if steps is None:
            steps = self.steps

        self.write_policy_file(policy)

	os.system(str(os.path.dirname(os.path.realpath(__file__))) + '/prism-4.4.beta-src/src/demos/run ' + str(os.path.dirname(os.path.realpath(__file__))))
	file = open('grid_world.pctl', 'w')
	file.write('P=?[true U<=' 
                    + str(steps) 
                    + '(s=' 
                    + str(int(self.M.S[-1])) 
                    + ')]')                 
	file.close()
	
	prob = self.run_prism()
        return prob
        

    def run_prism(self, cmd = None, timeout_sec = 5.0):
	if cmd is None:
	    cmd = [str(os.path.dirname(os.path.realpath(__file__))) + '/prism-4.4.beta-src/bin/prism', './grid_world.pm', './grid_world.pctl']
	kill_proc = lambda p: p.kill()
  	proc = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
  	timer = Timer(timeout_sec, kill_proc, [proc])
	probs = []
	prob = 0.0
  	try:
    	    timer.start()
   	    stdout, stderr = proc.communicate()
  	finally:
    	    timer.cancel()
  	    try:
	        lines = "".join(stdout).split('\n')
	    	for line in lines:
    		    if line.split(':')[0] == 'Result':
	    	    	prob_strs =  re.split('\[|\]| |,', line.split(':')[1].split('(')[0])
		        for prob_str in prob_strs:
	        	    if prob_str != '' and prob_str != ' ':
        		        probs.append(float(prob_str))
	        		break
			for p in probs:
			    prob += p
			prob /= len(probs)
  			return prob
	    except:
		print("PRISM model checking failed, return None")
	return None


    def cex_comics_timer(self, cmd = None, timeout_sec = 5.0):
        if cmd is None:
	    cmd = ['sh',  str(os.path.dirname(os.path.realpath(__file__))) + '/comics-1.0/comics.sh', './grid_world.conf']
  	kill_proc = lambda p: os.system('kill -9 $(pidof comics)')
  	proc = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
  	timer = Timer(timeout_sec, kill_proc, [proc])
  	try:
    	    timer.start()
   	    stdout, stderr = proc.communicate()
  	finally:
    	    timer.cancel()
	    print(stderr)	

    def counterexample(self, mus, safety = None, steps = None):
        if safety is None:
            safety = self.safety
        if steps is None:
            steps = self.steps

        self.write_conf_file(safety)

        epsilon = self.M.epsilon
        gamma = self.M.discount

	safety_ = safety
	print("Removing last counterexample file")
	os.system('rm counter_example.path')
	while safety_ > 1e-15:
            self.write_conf_file(safety_)
	    self.cex_comics_timer(['sh', str(os.path.dirname(os.path.realpath(__file__))) + '/comics-1.0/comics.sh', './grid_world.conf'], 2)
	    try:
	        file = open('counter_example.path', 'r')
		break
	    except:
		print("No counterexample found for spec = "\
 + str(safety_) + ". Lower down the safety threshold.")
		file = None
		safety_ = safety_ / 2.0
        if safety_ < 1e-15:
            #raise Exception("COMICS can't find counterexample!")  
            print("COMICS can't find counterexample!")  
            return None #mus[self.M.unsafes[0]]
	#if safety_ <= safety * epsilon**2:
	if file is None:
            print("COMICS can't find counterexample!")  
	    return None #mus[self.M.unsafes[0]]
	print("Generated counterexample for %f" % safety_)
	mu_cex = np.zeros(self.M.features[-1].shape)
	total_p = 0
	paths = []
	path_strings = []
	lines = file.readlines()
	file.close()
	for line in range(len(lines)-1):
	    path_strings.append(lines[line].split(' ')[0].split('->'))
	for path_string in range(len(path_strings)):
	    path = []
	    path.append(float(lines[path_string].split(' ')[2].split(')')[0]))
	    for state_string in path_strings[path_string]:
		if int(state_string) > len(self.M.S) - 1:
		    continue
		else:
		    path.append(int(state_string))
	    paths.append(path)
        for path in range(len(paths)):
        #for path in range(0, 1):
	    p = paths[path][0]
	    mu_path = np.zeros(self.M.features[-1].shape)
        
            '''
            Attention: 
            The last state is the dummy unsafe terminal  
            with zero feature vector. Make the seconde last
            state, which is a true state, absorbing.
            '''    
            state = 0
	    for state in range(1, len(paths[path]) - 1):
	        s = paths[path][state]
	        mu_path = mu_path + (gamma**(state - 1)) * self.M.features[s] 
	    length = len(paths[path]) - 2
	    s = paths[path][state]
	    while gamma**(length + 0)  > epsilon and steps >= length:
	        #mu_path = mu_path + (gamma**(length - 1)) * mus[s]
                #break
	        mu_path = mu_path + (gamma**(length + 0)) * self.M.features[s]
		length = length + 1
                
	    mu_cex = mu_cex + p * mu_path
	    total_p = total_p + p
	print("Counterexample for spec = " + str(safety) +  ": " + str(total_p))
	print("Counterexample feature:")
        print(mu_cex)
        if total_p > safety or True: ##True for cartpole, False for mountaincar
	    mu_cex = mu_cex/total_p
        else:
            mu_cex = mu_cex/safety
	print("Normalized Counterexample feature:")
        print(mu_cex)
	return mu_cex		


    def verify(self, policy = None, mus = None, safety = None, steps = None):
        if safety is None:
            safety = self.safety
        if policy is None:
            policy = self.policy.copy()
        if steps is None:
            steps = self.steps
        prob = self.model_check(policy, steps) 
        if prob is None:
            #raise Exception('PRISM check get None probability!')  
            print('PRISM check get None probability!')  
            prob = 1.0
            
        elif prob <= safety:
            print("\n>>>>Safe<<<<<Verified policy unsafe probability is %f\n" % prob)
            return None, prob
        else:
            print("\n>>>>Unsafe<<<<<Verified policy unsafe probability is %f\n" % prob)
            cex = self.counterexample(mus = mus, safety = safety, steps = steps)
            return cex, prob
    
