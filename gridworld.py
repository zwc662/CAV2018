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


    def build_mdp(self):
        self.M.starts = [0]

        self.M.targets.append(self.coord_to_index([self.dim - 1, self.dim - 1]))
        self.M.targets.append(self.coord_to_index([self.dim - 1, self.dim - 2]))

        self.M.unsafes.append(self.coord_to_index([self.dim - 3, 2]))
        self.M.unsafes.append(self.coord_to_index([2, self.dim - 3]))

	for x in range(self.dim/2, self.dim):
		for y in range(0, self.dim):
			if x - 3 >= y:
				self.M.unsafes.append(self.coord_to_index([y, x]))
	for y in range(self.dim/2, self.dim):
		for x in range(0, self.dim):
			if y >= x + 3:
				self.M.unsafes.append(self.coord_to_index([y, x]))

        self.build_features()

        self.build_transitions()

        self.M.set_initial_transitions()

        self.M.output()
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
                R[i, j] = rewards[grids.coord_to_index([i, j])]


        if title is None:
            pylab.title('Close window to continue learning')    
        else:
            pylab.title(title)
        pylab.set_cmap('gray')
        pylab.axis([0, self.grids[0], self.grids[1],0])
        c = pylab.pcolor(R, edgecolors='w', linewidths=1)
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

        file = open('./data/demo_gridworld', 'a')

        trajectory = []
	state = self.M.starts[0]
        t = 0
        
	pylab.close()
	pylab.ion()

	title = "Input action in terminal [0:end, 1: left, 2: down, 3: right, 4: up]"
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
        
         
    def learn_from_human_demo(self, steps = 64):        
        if steps is None:
            steps = self.steps
        theta = np.array([1., 1., -1., -1.])
        theta = theta/np.linalg.norm(theta, ord = 2)
        self.M.rewards = np.dot(self.M.features, theta) 
        self.demo(self.M.rewards, steps = steps)
        self.learn_from_demo_file()

    def learn_from_demo_file(self):
        demo_mu = learn.read_demo_file('./data/demo_gridworld')

        #mus, _ = self.M.optimal_policy(theta)
        #demo_mu = mus[-2]

        #theta = np.array([1., 1., 0., 0.])
        #demo_mu = np.array([ 34.27191387, 84.02248669,   2.59569051,   0.84842763])
    
        #theta = np.array([ 0.67988384,  0.68017034,  0.0580262,  -0.26787914])
        #demo_mu = np.array([ 33.81998363,  83.00976111,   2.04596755,   0.89145121])
    
        #Learn from \approx[0, 0, -1, -0.1]
        #demo_mu = np.array([  57.58416513,  58.19742098,   1.10344096,   1.43110657])
    
        #Learn from \approx[0.1, 0.1, -0.7, -0.7]
        #demo_mu = np.array([31.48191543,  85.06620586,   1.74326134,   1.18968865])
        self.AL(demo_mu)
    
    def AL(self, exp_mu):
    
        learn = cegal(self.M, max_iter = 30)
        opt = super(cegal, learn).iteration(exp_mu)
        prob = learn.model_check(opt['policy'], steps = 64)
        
        print("\n>>>>>>>>Apprenticeship Learning learnt policy weight vector:")
        print(opt['theta'])
        print("\nFeature vector margin: %f" % opt['diff'])
        print("\nPRISM model checking result: %f\n" % prob)
        
        grids.M.rewards = np.dot(grids.M.features, opt['theta']) 
        grids.draw_grids(grids.M.rewards)

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
            n = raw_input('1. Try weight vector learnt via AL\n\
2. Try weight vector learnt via SAAL\n3. Quit\n')
            if n == '1':
                theta = opt_['theta']
            elif n == '2':
                theta = opt['theta']
            elif n == '3':
                break
            else:
                print("Invalid input")
                continue
             
            grids.M.rewards = np.dot(grids.M.features, theta) 
            grids.draw_grids(grids.M.rewards)
        return opt, opt_

    def synthesize(self, learn, exp_mu, opt = None, safety = None, steps = None):

        if safety is None:
            safety = self.safety
        if steps is None:
            steps = self.steps

        opt, opt_ = learn.iteration(exp_mu = exp_mu, opt = None, safety = safety, steps = steps)
        ## cegal.iteration returns SAAL and AL learning results
        ## opt = (diff, theta, policy, mu)

        print("\n\n\nLearning result for safety specification:\n")
        print("\nP<=" + str(safety) + "[true U<=" + str(steps) + " 'unsafe']\n") 

        print("\n>>>>>>>>Apprenticeship Learning learnt policy weight vector:")
        print(opt_['theta'])
        print("\nFeature vector margin: %f" % opt_['diff'])
        print("\nPRISM model checking result: %f\n" % opt_['prob'])

        print("\n>>>>>>>>Safety-Aware Apprenticeship Learning learnt policy weight vector:")
        print(opt['theta'])
        print("\nFeature vector margin: %f" % opt['diff'])
        print("\nPRISM model checking result: %f\n" % opt['prob'])

        return opt, opt_
        
    
        
if __name__ == "__main__":
    grids = gridworld()    
    grids.build_mdp()


    #grids.learn_from_human_demo(steps = 64)
    #opt, opt_ = grids.synthesize_from_demo_file(safety = 0.1)
    #theta = opt['theta'] 
    
    #theta = np.array([1., 1., -1., -1.])
    #theta = np.array([0.59033466,  0.59567414, -0.39910104, -0.3706692 ])
    theta = np.array([0.00812865,  0.00252285, -0.70708771, -0.70707463])
    theta = theta/np.linalg.norm(theta, ord = 2)
    mus, policy = grids.M.optimal_policy(theta)
    exp_mu = mus[-2]
    exp_mu_ = grids.policy_simulation(policy, 10000)


    print("Analytical expected features:")
    print(exp_mu)
    print("Simulated expected features:")
    print(exp_mu_)
    print("Feature difference:")
    print(np.linalg.norm(exp_mu - exp_mu_, ord = 2))
    
    grids.AL(exp_mu)
    grids.AL(exp_mu_)



