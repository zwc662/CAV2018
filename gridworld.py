from mdp import mdp
from grids import grids
import numpy as np
import scipy.optimize as optimize
from cvxopt import matrix, solvers
from scipy import sparse
import sys
import os
import ast
import time
import mdptoolbox
import math
import pylab
from apirl import apirl
from cegal import cegal

class gridworld(grids, object):
    def __init__(self, dim = None, prob = None):
        if sys.version_info[0] >= 3:  
            super().__init__()
        else:
            super(gridworld, self).__init__()

        if dim is None:
            self.dim = 8    
        else:
            self.dim = dim
    
        if prob is None:
            self.prob = 0.8
        else:
            self.prob = prob

        self.grids = [self.dim] * 2
 
        self.M = mdp(self.dim**2 + 2, 5)

        self.opt = None

        self.steps = self.dim**2
    
    def set_initial_opt(self):
        self.opt = {}
        self.opt['policy'] = np.zeros([len(self.M.S), len(self.M.A)])
        self.opt['policy'][: -2][0] = 1.0

        self.M.set_policy(self.opt['policy'])
        self.opt['mu'] = self.M.expected_features_manual()[-2]
        self.opt['theta'] = np.zeros([len(self.M.features[0])])
    

    def build_mdp_from_file(self):
        os.system('cp ./data/state_space_gridworld ./data/state_space')
        os.system('cp ./data/unsafe_gridworld ./data/unsafe')
        os.system('cp ./data/mdp_gridworld ./data/mdp')
        os.system('cp ./data/start_gridworld ./data/start')
        self.M.input()

        self.M.targets.append(self.coord_to_index([self.dim - 1, self.dim - 1]))
        self.M.targets.append(self.coord_to_index([self.dim - 1, self.dim - 2]))

        self.build_features()
        
        self.M.set_initial_transitions()

        self.M.output()
        self.set_initial_opt()


    def build_mdp(self):
        self.M.starts = [0]

        self.M.targets.append(self.coord_to_index([self.dim - 1, self.dim - 1]))
        self.M.targets.append(self.coord_to_index([self.dim - 1, self.dim - 2]))

        self.M.unsafes.append(self.coord_to_index([self.dim - 2, 2]))
        self.M.unsafes.append(self.coord_to_index([2, self.dim - 2]))

	for x in range(self.dim/2, self.dim):
		for y in range(0, self.dim):
			if x - self.dim/2 + 2 >= y:
				self.M.unsafes.append(self.coord_to_index([y, x]))
	for y in range(self.dim/2, self.dim):
		for x in range(0, self.dim):
			if y - self.dim/2 + 2 >= x:
				self.M.unsafes.append(self.coord_to_index([y, x]))

        self.build_features()

        self.build_transitions()

        self.M.set_initial_transitions()

        self.M.output()
        self.set_initial_opt()
        #self.M.set_targets_transitions()
        '''
        for a in self.M.A:
            self.M.T[a][self.M.S[-2]] = 0.0 * self.M.T[a][self.M.S[-2]]
            for s in self.M.starts:
                self.M.T[a][self.M.S[-2],s] = 1.0/len(self.M.starts) 
        for a in self.M.A:
            for s in self.M.targets:
                self.M.T[a][s] = 0.0 * self.M.T[a][s]
                self.M.T[a][s, s] = 1.0
        #self.M.set_unsafes_transitions()
        for a in self.M.A:
            self.M.T[a][self.M.S[-1], self.M.S[-1]] = 1.0
            for s in self.M.unsafes[0:2]:
                self.M.T[a][s] = 0.0 * self.M.T[a][s]
                self.M.T[a][s, s] = 1.0 
        '''

    def build_features(self):
        self.M.features = list()
        for s in self.M.S:
            self.M.features.append(list())
            for c in (self.M.targets + self.M.unsafes[0:2]):
                self.M.features[s].append(math.exp(-1.0 * 
                                                np.linalg.norm(
                                                np.array(self.index_to_coord(s)) 
                                                - np.array(self.index_to_coord(c)), ord = 2)))           
        
        self.M.features = np.array(self.M.features)
        self.M.features[-2] = self.M.features[-2] * 0.0
        self.M.features[-1] = self.M.features[-1] * 0.0

    def build_transitions(self):
        self.M.T = list()
        for a in self.M.A:
            self.M.T.append(np.zeros([len(self.M.S), len(self.M.S)]).astype(float))
            for y in range(self.dim):
                for x in range(self.dim):
                    s = self.coord_to_index([y, x])
                    if s == self.M.S[-1]:
                        self.M.T[a][s, s] = 1.0
                        break
                    if a == 0:
                        self.M.T[a][s, s] = 1.0
                        continue
                    self.M.T[a][s, s] += (1 - self.prob)/5.0

                    s_ = self.coord_to_index([abs(y-1), x])
                    self.M.T[a][s, s_] += (1 - self.prob)/5.0 + int(a == 4) * self.prob

                    s_ = self.coord_to_index([y, abs(x-1)])
                    self.M.T[a][s, s_] += (1 - self.prob)/5.0 + int(a == 3) * self.prob

                    s_ = self.coord_to_index([self.dim - 1 - abs(self.dim - 1  - y - 1), x])
                    self.M.T[a][s, s_] += (1 - self.prob)/5.0 + int(a == 2) * self.prob

                    s_ = self.coord_to_index([y, self.dim - 1 - abs(self.dim - 1 - x - 1)])
                    self.M.T[a][s, s_] += (1 - self.prob)/5.0 + int(a == 1) * self.prob
                    

    def draw_grids(self, rewards, trajectory = None, title = None):
        R = np.zeros([self.dim, self.dim])
        for i in range(self.dim):
            for j in range(self.dim):
                R[i, j] = rewards[self.coord_to_index([i, j])]

        if title is None:
            pylab.title('Close window to continue')    
        else:
            pylab.title(title)
        pylab.set_cmap('gray')
        pylab.axis([0, self.grids[0], self.grids[1], 0])
        c = pylab.pcolor(R, edgecolors='w', linewidths=1)
    
        x=[]
        y=[]
	for i in range(0, self.dim):
            j = i + self.dim/2 - 2
            if j < self.dim and j >= self.dim/2:
                x.append(j)
	        y.append(i)
                x.append(j + 1)
                y.append(i)
            elif j < self.dim - 2:
                x.append(self.dim/2)
                y.append(i)
        pylab.plot(x, y, 'r')

        x=[]
        y=[]
	for i in range(0, self.dim):
            j = i + self.dim/2 - 2
            if j < self.dim and j >= self.dim/2:
                y.append(j)
	        x.append(i)
                y.append(j + 1)
                x.append(i)
            elif j < self.dim - 1:
                y.append(self.dim/2)
                x.append(i)
        pylab.plot(x, y, 'r')
        

        y=[]
        x=[]
        if trajectory != None:
            for trans in trajectory:
                coord = self.index_to_coord(trans[-1])
                y.append(coord[0])
                x.append(coord[1])
                pylab.plot(x, y, 'bo', x, y, 'b-', [x[-1]], [y[-1]], 'ro')
        pylab.show()

                        
    def episode(self, rewards, steps = 64):
        if steps is None:
            steps = self.steps

        print("Go to the white goal states on the bottom right.")
        print("States surrounded by the red lines, especially the two dark cells, are unsafe states. Try not to cross.\n")
        file = open('./data/demo_gridworld', 'a')

        trajectory = []
	state = self.M.starts[0]
        t = 0
        
	pylab.close()
	pylab.ion()

	title = "Input action in terminal [0: end, 1: left, 2: down, 3: right, 4: up]."
	self.draw_grids(rewards, None, title)
	while steps > 0:
            try:
		action = int(raw_input("Choose next action: "))
		if action!= 0 and action != 1 and action !=2 and action !=3 and action !=4:
		    print("Invalid action, input again")
		    next
		elif action == 0:
                    print("Trajectory ends")
                    while t < steps:
                        file.write(str(t) + ' ' + str(state) + ' ' + str(action) + ' ' + str(state) + '\n')
                        t += 1
		    pylab.ioff()
		    pylab.close('all')
		    break
		else:
	            trajectory.append([t, state])
		    trajectory[-1].append(action)

                    state_ = self.M.move(state, action)
                    trajectory[-1].append(state_)
            
                    file.write(str(t) + ' ' + str(state) + ' ' + str(action) + ' ' + str(state_) + '\n')

                    
		    self.draw_grids(rewards, trajectory, title)

                    t += 1
                    steps -= 1
                    state = state_

	    except:
		print("Invalid action, input again")
		next
        if steps == 0:
            print("Reached maximum step length")
        file.close()

    def demo(self, rewards, steps = float('inf')):
        path_num = 0
        os.system('rm ./data/demo_gridworld')
        start = raw_input("Human demonstrate? [Y/N]")
        while start == 'y' or start == 'Y':
            self.episode(rewards, steps)
            start = raw_input("Human demonstrate? [Y/N]")

        
    def policy_simulation(self, policy = None, max_iter = 10000):
        if policy is None:
            policy = self.M.policy

        exp_mu = np.zeros((len(self.M.features[0])))
        diff = float('inf')
        itr = 0
        while diff > self.M.epsilon and itr < max_iter:
            exp_mu_ = exp_mu.copy()
            itr += 1

            diff_ = float('inf')
            s = self.M.S[-2]
            mu = self.M.features[s].copy()
            t = 0
            while diff_ > self.M.epsilon:
                mu_ = mu.copy()
                t += 1
                a = policy[s].argmax()
                s = self.M.move(s, a)
                mu += self.M.features[s] * self.M.discount**t
                diff_ = np.linalg.norm(mu - mu_, ord = 2)
            exp_mu = (exp_mu * (itr - 1) + mu)/itr
            diff = np.linalg.norm(exp_mu - exp_mu_, ord = 2)
        
        return exp_mu
        
         
    def learn_from_human_demo(self, steps = None):        
        if steps is None:
            steps = self.steps
        theta = np.array([1., 1., -1., -1.])
        theta = theta/np.linalg.norm(theta, ord = 2)
        self.M.rewards = np.dot(self.M.features, theta) 
        self.demo(self.M.rewards, steps = steps)
        self.learn_from_demo_file()

    def learn_from_demo_file(self):
        learn = cegal(self.M, max_iter = 30)
        demo_mu = learn.read_demo_file('./data/demo_gridworld')
        self.AL(demo_mu)
    
    def AL(self, exp_mu):
    
        learn = cegal(self.M, max_iter = 30)
        opt = super(cegal, learn).iteration(exp_mu)
        prob = learn.model_check(opt['policy'], steps = self.steps)
        
        print("\n>>>>>>>>Apprenticeship Learning learns a policy \
 which is an optimal policy of reward function as in the figure.")
        #print(opt['theta'])
        #print("\nFeature vector margin: %f" % opt['diff'])
        print("\nGiven safety spec:\nP=? [U<= " + str(self.dim**2) + " 'unsafe']\n")
        print("\nPRISM model checking the probability of reaching\
 the unsafe states: %f\n" % prob)
        
        self.M.rewards = np.dot(self.M.features, opt['theta']) 
        self.draw_grids(self.M.rewards)
        return opt

    def synthesize_from_human_demo(self, safety, steps = 64):
        if steps is None:
            steps = self.steps
        theta = np.array([1., 1., -1., -1.])
        theta = theta/np.linalg.norm(theta, ord = 2)
        self.M.rewards = np.dot(self.M.features, theta) 
        self.demo(self.M.rewards, steps = steps)
        synthesize_from_demo_file(safety, steps)
        
    
    def synthesize_from_demo_file(self, safety, steps = 64, path = './data/demo_gridworld'):
        if safety is None:
            safety = self.safety
        if steps is None:
            steps = self.steps

        learn = cegal(self.M, safety = safety, steps = steps, max_iter = 30)
        exp_mu = learn.read_demo_file(path)

        opt, opt_ = self.synthesize(learn = learn, exp_mu = exp_mu, safety = safety, steps = steps)

        while True:
            n = raw_input('1. Check reward mapping learnt via AL\n\
2. Check reward mapping learnt via Safety-Aware AL\n3. Quit\n')
            if n == '1':
                theta = opt_['theta']
            elif n == '2':
                theta = opt['theta']
            elif n == '3':
                break
            else:
                print("Invalid input")
                continue
            if theta.any() == 0:
                print("The initial safe policy is returned. There is no reward mapping.")
             
            self.M.rewards = np.dot(self.M.features, theta) 
            self.draw_grids(self.M.rewards)
        return opt, opt_

    def synthesize(self, learn, exp_mu, opt = None, safety = None, steps = None):
        if opt is None:
            opt = self.opt
        if safety is None:
            safety = self.safety
        if steps is None:
            steps = self.steps

        opt, opt_ = learn.iteration(exp_mu = exp_mu, opt = opt, safety = safety, steps = steps)
        ## cegal.iteration returns SAAL and AL learning results
        ## opt = (diff, theta, policy, mu)

        print("\n\n\nLearning result for safety specification:\n")
        print("\nP<=" + str(safety) + "[true U<=" + str(steps) + " 'unsafe']\n") 

        print("\n>>>>>>>>Apprenticeship Learning learnt policy weight vector:")
        print(opt_['theta'])
        #print("\nFeature vector margin: %f" % opt_['diff'])
        print("\nPRISM model checking result: %f\n" % opt_['prob'])

        print("\n>>>>>>>>Safety-Aware Apprenticeship Learning learnt policy weight vector:")
        print(opt['theta'])
        #print("\nFeature vector margin: %f" % opt['diff'])
        print("\nPRISM model checking result: %f\n" % opt['prob'])

        return opt, opt_
        
    
'''
        
if __name__ == "__main__":
    grids = gridworld()    
    grids.build_mdp()


    #grids.learn_from_human_demo(steps = 64)
    #opt, opt_ = grids.synthesize_from_demo_file(safety = 0.5)
''' 
