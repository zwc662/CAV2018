import sys
from mdp import mdp
from grids import grids
import numpy as np
import scipy.optimize as optimize
from cvxopt import matrix, solvers
from scipy import sparse
import os
import time
import mdptoolbox
import math
import pylab
from apirl import apirl
from cegal import cegal
import util

import ast
import gym
import re
import cProfile
import math
from grids import grids

# In order to run this script you must download and compile
# the following library: https://github.com/Svalorzen/AI-Toolbox
# Building it will create an MDP.so file which you can then
# include here.
import MDP

env = gym.make('MountainCar-v0')

class mountaincar(grids, object):
    def __init__(self):
        if sys.version_info[0] >= 3:  
            super().__init__()
        else:
            super(mountaincar, self).__init__()

        self.grids = [38, 30]

        ranges = [[-1.2, 0.6], [-0.07, 0.07]] 
        self.build_threshes(ranges)

        num_S = 1
        for i in self.grids:
            num_S *= i
        num_S += 2

        num_A = env.action_space.n;

        self.M = mdp(num_S, num_A)

        self.set_unsafe()

        self.steps = 200
        self.maxepisodes = 100000

        self.combo = 2

        self.steps = int(self.steps/(self.combo + 1))
        if self.steps * (self.combo + 1) < 200:
	    self.steps = self.steps + 1

    
    def set_unsafe(self):
        self.M.unsafe = []
        for s in self.M.S[:-2]:
            coords = self.index_to_coord(s)
	    if (coords[0] <= 3 \
                and coords[1] <= self.grids[1]/2 - 3) \
                or (coords[0] >= self.grids[0] - 4 \
                and coords[1] >= self.grids[1]/2 + 3):
                #(-\infty, -1.0] or [1.0, \infty)
	        self.M.unsafes.append(s)
        #self.M.unsafes.append(self.M.S[-1])
        #self.M.set_unsafes_transitions()

    def check_transitions(self):
        for a in self.M.A:
            if isinstance(self.M.T[a], np.ndarray) is False:
                self.M.T[a] = np.array(self.M.T[a])
            for s in self.M.S:
                p_tot = self.M.T[a][s].sum() 
                if p_tot < 1.0:
                    self.M.T[a][s, s] = 1.0 - p_tot

    def build_MDP_from_file_old(self):
        self.build_features()
        self.read_MDP_file_old()
        
        self.set_unsafe()
        self.check_transitions()

        self.M.set_initial_transitions()

        self.M.output()
        os.system('cp ./data/state_space ./data/state_space_mountaincar')
        os.system('cp ./data/unsafe ./data/unsafe_mountaincar')
        os.system('cp ./data/mdp ./data/mdp_mountaincar')
        os.system('cp ./data/start ./data/start_mountaincar')

    def read_MDP_file_old(self):
        file = open('./data/MDP_mountaincar', 'r')
        lines = file.readlines()
        if lines[0] == 'starts' or lines[0] == 'starts\n':
            index = 1
        else:
            exit()
        self.M.starts = list()
        for index in range(1, len(lines)):
            if lines[index] != 'features' and lines[index] != 'features\n':
                line = re.split(':|\n|''', lines[index])
                j = int(line[1])
                i = int(line[0])
                coord = [j, i%4, (i/4)%4, ((i/4)/4)]
                s = self.coord_to_state(coord)
                self.M.starts.append(s)
            else:
                break
        index += 1  

        demo_mu = []
        line = lines[index].split(':')
        for feature in line:
            try:
                demo_mu.append(float(feature))
            except:         
                continue
        demo_mu = np.array(demo_mu)
        index += 2

        self.M.T = list()
        for a in self.M.A:
            self.M.T.append(np.zeros([len(self.M.S), len(self.M.S)]))
            self.M.T[a][-1, -1] = 1.0
        for i in range(index, len(lines)):
            line = re.split(':|\n|''', lines[i])
            j = int(line[1])
            i = int(line[0])
            coord = [j, i%4, (i/4)%4, ((i/4)/4)]
            s = self.coord_to_state(coord)
            
            j = int(line[4])
            i = int(line[3])
            coord = [j, i%4, (i/4)%4, ((i/4)/4)]
            s_ = self.coord_to_state(coord)

            self.M.T[int(line[2])][s, s_] = float(line[5])
        file.close()

    def build_features(self):
        f = 28
        self.M.features = np.zeros([len(self.M.S), 2 + f])
        feature_states = []
        for i in range(f):
            s = int(len(self.M.S) * i/f)
            feature_states.append(s)
       
        for s in self.M.S: 
            coord = self.index_to_coord(s)
            for i in range(len(coord)):
                self.M.features[s, i] = math.exp(-1.0 *  coord[i])
            y = coord[1]
            x = coord[0]
            for i in range(f):
                s_ = feature_states[i]
                coord_ = self.index_to_coord(s_)
                y_ = coord_[1] 
                x_ = coord_[0]
                #y_ = feature_states[i]/self.grids[0]
                #x_ = feature_states[i]%self.grids[0]
                self.M.features[s, i + 2] = math.exp(-0.25 * math.sqrt((1.0 * y - y_)**2 + (1.0 * x - x_)**2))
        
        self.M.features[-2] = self.M.features[-2] * 0.0
        self.M.features[-1] = self.M.features[-1] * 0.0


    def build_MDP_from_file(self):
        os.system('cp ./data/state_space_mountaincar ./data/state_space')
        os.system('cp ./data/unsafe_mountaincar ./data/unsafe')
        os.system('cp ./data/mdp_mountaincar ./data/mdp')
        os.system('cp ./data/start_mountaincar ./data/start')
        self.M.input()

        self.set_unsafe()
        self.check_transitions()

        self.build_features()
        
        self.M.set_initial_transitions()

        self.M.output()

    def learn_from_feature_file(self):
        learn = apirl(self.M, max_iter = 30)

        file = open('./data/demo_mountaincar', 'r')
        line = file.readlines()[-1]
        print(len(line.split('\n')[0].split(' ')))

        learn.exp_mu = list()
        try:
            for f in line.split('\n')[0].split(' '):
                learn.exp_mu.append(float(f))
        except:
            pass
        learn.exp_mu = np.array(learn.exp_mu)
        
        print(learn.exp_mu)
        opt = learn.iteration(learn.exp_mu) 
        return opt

    def learn_from_demo_file(self):
        learn = apirl(self.M, max_iter = 30)
        learn.exp_mu = learn.read_demo_file('./data/demo_mountaincar') 
        print(learn.exp_mu)
        opt = learn.iteration(learn.exp_mu) 
        return opt

    def copy_from_policy_file(self):
        self.M.policy = np.zeros([len(self.M.S), len(self.M.A)])
        file = open('./data/demo_policy_mountaincar', 'r')
        lines = file.readlines()
    	for i in range(len(lines)):
    	    line = re.split(':|\n|''', lines[i])
    	    for j in range(len(line)):
    	        if line[j] != '':
                    coord = [j, i%4, (i/4)%4, ((i/4)/4)]
                    s = self.coord_to_state(coord)
    		    a = int(float(line[j]))
                    self.M.policy[s, a] = 1.0
        file.close()

        for s in self.M.S:
            p_tot = self.M.policy[s].sum()
            a_max = self.M.policy[s].argmax()
            if p_tot < 1.0:
                self.M.policy[s, a_max] += 1.0 - p_tot
                 
        self.M.set_initial_transitions()
        self.M.set_policy()
    
    def write_policy_file(self, policy = None, path = './data/policy_mountaincar'):
        if policy is None:
            policy = self.M.policy

        os.system('rm ' + str(path))
        os.system('touch ' + str(path))
        file = open(str(path), 'w')
        for s in self.M.S:
            for a in self.M.A:
                file.write(str(policy[s, a]) + ' ')
            file.write('\n')
        file.close()
        
    def read_policy_file(self,  path = './data/policy_mountaincar'):
        self.M.policy = np.zeros([len(self.M.S), len(self.M.A)])
        file = open(str(path), 'r')
        lines = file.readlines()
        for s in self.M.S:
            line = lines[s].split('\n')[0].split(' ')
            for a in self.M.A:
                self.M.policy[s, a] = float(line[a])
        file.close()
        self.M.set_policy()
        
    def learn_from_policy_file(self):
        learn = apirl(self.M, max_iter = 50)
        self.read_policy_file()
        mus = self.M.expected_features_manual()
        learn.exp_mu = mus[-2]
        print(learn.exp_mu)
        opt = learn.iteration(learn.exp_mu) 
        return opt

    def learn_from_policy_file_old(self):
        learn = apirl(self.M, max_iter = 50)
        self.copy_from_policy_file()
        mus = self.M.expected_features_manual()
        learn.exp_mu = mus[-2]
        print(learn.exp_mu)
        opt = learn.iteration(learn.exp_mu) 
        return opt

    def episode(self, demo = False, safe = False, performance = False, policy = None, steps = None):
        if steps is None:
            steps = self.steps
        
        if policy is None:
            policy = self.M.policy

        unsafes = np.zeros([len(self.M.S)]).astype(bool)
        for s in self.M.unsafes:
            unsafes[s] = True

        count_down = 5
        
        cut = True
        if demo or safe or performance:
            cut = False

	o = env.reset()
	t = 0
	s = self.observation_to_index(o); 

        path = []   
	#print("state %d" % s)
	for t in xrange(steps):
            s =  self.observation_to_index(o); 
	    #print("state %d" % s)
     	    # Convert the observation into our own space
      	    # Select the best action according to the policy
     	    a = int(policy[s].argmax())
	    #print("action %d" % a)
                
       	    # Combo Act
            for i in range(self.combo):
      	        o_, rew, done, info = env.step(a);
     	        s_ = self.observation_to_index(o_);

	        if done:
                    path.append([t, s, a, s_])
                    print("End after steps %d" % t)
                    return path


                if unsafes[s_]:
                    path.append([t, s, a, s_])
                    print("End in unsafe state after steps %d" % t)
                    if demo or safe:
                        return list()

      	    o_, rew, done, info = env.step(a);
      	    # See where we arrived
     	    s_ = self.observation_to_index(o_);

            path.append([t, s, a, s_])

            if unsafes[s_]:
                print("End in unsafe state after steps %d" % t)
                if demo or safe:
                    return list()

	    if done:
                print("End after steps %d" % t)
                return path
            
     	    # Record information, and then run PrioritizedSweeping
            o = o_;

            if not (demo or performance or safe):
    	        env.render()
	    if cut:
	        while count_down > 0:
	            count_down -= 1
		    cut = raw_input("cut!!!")
		    cut = False
		cut = raw_input("cut!!!")


    def demo(self, policy = None, episodes = 5000):
        os.system('rm ./data/demo_mountaincar')
        os.system('toucn ./data/demo_mountaincar')
        print("Generate %d episodes" % episodes)
        use = 0 
        file = open('./data/demo_mountaincar', 'w')
        while use < 1000:
            path = self.episode(demo = True, policy = policy)
            episodes -= 1
            if len(path) >= self.steps - 2:
                use += 1
                print('%d validate demo have been generated' % use)
                for t in range(len(path)):
                    file.write(str(path[t][0]) + ' ' 
                            + str(path[t][1]) + ' ' 
                            + str(path[t][2]) + ' '
                            + str(path[t][3]) + '\n')
        file.close()
        print("Use %d demos" % use)
        
    def test(self, policy = None, episodes = 1000):
        dead = 0.0 
        i_episode = 0

        while i_episode <= episodes:
            print("Episode %d" % i_episode)
            path = self.episode(safe = True, policy = policy)
            if len(path) == 0:
                dead += 1
            i_episode += 1
        dead = dead/episodes

        avg = 0
        i_episode = 0
        while i_episode < episodes:
            print("Episode %d" % i_episode)
            path  = self.episode(performance =True, policy = policy)
            avg += len(path)
            i_episode += 1
        avg /= episodes

        print('Unsafe ratio: %f' % dead)
        print("Average step length: %f" % avg)

    def synthesize_from_demo_file(self, safety = 0.3, steps = 200, path = './data/demo_mountaincar'):
        if safety is None:
            safety = self.safety
        if steps is None:
            steps = self.steps

        learn = cegal(self.M, max_iter = 50, safety = safety, steps = steps)
        exp_mu = learn.read_demo_file(path)

        opt, opt_ = self.synthesize(learn = learn, exp_mu = exp_mu, safety = safety, steps = steps)
        
        
        while True:
            n = raw_input('1. Try AL policy, 2. Try CEGAL policy, 3. Quit')
            if n == '1':
                policy = opt_['policy']
            elif n == '2':
                policy = opt['policy']
            elif n == '3':
                break
            else:
                print("Invalid")
                continue
             
            self.test(policy = policy)
            
            w = raw_input('Write policy file?[w]')
            if w == 'w' or w == 'W':
                if n == '1':
                    self.write_policy_file(policy = policy, path = './data/policy_mountaincar')
                else:
                    self.write_policy_file(policy = policy, path = './data/policy_mountaincar_' + str(safety))
            

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

    def run_tool_box(self):

        paths = []

        self.M.starts = list()

        unsafes = np.zeros([len(self.M.S)]).astype(bool)
        for u in self.M.unsafes:
            unsafes[u] = True

        starts = np.zeros([len(self.M.S)]).astype(bool)
        for u in self.M.starts:
            starts[u] = True
        
        self.M.T = list()
        for a in self.M.A:
            self.M.T.append(np.zeros([len(self.M.S), len(self.M.S)]))
        
        gamma = self.M.discount
        exp = MDP.SparseExperience(len(self.M.S), len(self.M.A));
        model = MDP.SparseRLModel(exp, gamma);
        solver = MDP.PrioritizedSweepingSparseRLModel(model, 0.1, 500);
        policy = MDP.QGreedyPolicy(solver.getQFunction());

        using = 0
	episodes=0
	win = 0
	streak = list()
	
        maxepisodes = 10000
	for i_episode in xrange(maxepisodes):
                path = []
    		o = env.reset()

                s_i = self.observation_to_index(o)

		dead = False
                rec = self.steps
                done = False

    		for t in xrange(self.steps):
        	    # Convert the observation into our own space
        	    s = self.observation_to_index(o);
        	    # Select the best action according to the policy
        	    a = policy.sampleAction(s)
        	    # Combo act
                    for i in range(self.combo): 
        	        o_, rew, done, info = env.step(a);
        	        # See where we arrived
        	        s_ = self.observation_to_index(o_);

                        if unsafes[s_]:
                            self.M.T[a][s, s_] += 1.0
		            dead = True

        	    o_, rew, done, info = env.step(a);
        	    # See where we arrived
        	    s_ = self.observation_to_index(o_);

                    self.M.T[a][s, s_] += 1.0
                    path.append([t, s, a, s_])

                    if unsafes[s_]:
		        dead = True
                        
        	    if done:
                        break
                    # Record information, and then run PrioritizedSweeping
        	    exp.record(s, a, s_, rew);
       		    model.sync(s, a, s_);
		    solver.stepUpdateQ(s, a);
		    solver.batchUpdateQ();

		    o = o_;

   		#   if render or i_episode == maxepisodes - 1:
      		#    env.render()

    		  
                # Here we have to set the reward since otherwise rewards are
                # always 1.0, so there would be no way for the agent to distinguish
                # between bad actions and good actions.
                if t >= self.steps - 1:
                    rew = -200
                    tag ='xxx'
		    streak.append(0)
                else:
		    rew = self.steps - t
       		    tag = '###';
        	    streak.append(1)
      		    win += 1;

  		    if not dead:
                        using += 1
                        paths.append(path)
                        if not starts[s_i]:
                            self.M.starts.append(s_i)

  		if len(streak) > 100:
      			streak.pop(0)


    		episodes +=1;
    		exp.record(s, a, s_, rew);
    		model.sync(s, a, s_);
    		solver.stepUpdateQ(s, a);
    		solver.batchUpdateQ();
                # If the learning process gets stuck in some local optima without
                # winning we just reset the learning. We don't want to try to change
                # what the agent has learned because this task is very easy to fail
                # when trying to learn something new (simple exploration will probably
                # just make the pole topple over). We just want to learn the correct
                # thing once and be done with it.
		if episodes == 100:
    		    if sum(streak) < 30:
	                exp = MDP.SparseExperience(len(self.M.S), len(self.M.A));
       			model = MDP.SparseRLModel(exp, gamma);
        		solver = MDP.PrioritizedSweepingSparseRLModel(model, 0.1, 500);
      	  		policy = MDP.QGreedyPolicy(solver.getQFunction());
		    if sum(streak) < 80:
                        paths = list()
			using = 0
                        self.M.starts = list()
		    episodes = 0
		if using > 1000 and i_episode > maxepisodes/2:
		    break

	    	print "Episode {} finished after {} timecoords {} win:{} use:{}".format(i_episode, t+1, tag, win, using)

	
        file = open('./data/start_mountaincar', 'w')
	for start in self.M.starts:
		file.write(str(start) + '\n')
        file.close()

    
        for a in range(len(self.M.A)):
            for s in range(len(self.M.S)):
                tot = np.sum(self.M.T[a][s])
                if tot == 0.0:
                    self.M.T[a][s, s] = 0.0
            self.M.T[a] = sparse.bsr_matrix(self.M.T[a])        
            self.M.T[a] = sparse.diags(1.0/self.M.T[a].sum(axis = 1).A.ravel()).dot(self.M.T[a]).todense()

	file = open('./data/mdp_mountaincar', 'w')
        for s in self.M.S:
            for a in self.M.A:
                for s_ in self.M.S:
                    file.write(str(s) + ' ' 
                                + str(a) + ' ' 
                                + str(s_) + ' ' 
                                + str(self.M.T[a][s, s_]) + '\n')
        file.close()

        file = open('./data/unsafe_mountaincar', 'w')
        for s in self.M.unsafes:
            file.write(str(s) + '\n')
        file.close()

        file = open('./data/state_space_mountaincar', 'w')
        file.write('states\n' + str(len(self.M.S)) + '\nactions' + str(len(self.M.A)))
        file.close()

        file = open('./data/demo_mountaincar', 'w')
        for path in paths:
            for t in range(len(path)):
                file.write(str(path[t][0]) + ' ' 
                            + str(path[t][1]) + ' ' 
                            + str(path[t][2]) + ' '
                            + str(path[t][3]) + '\n')
        file.close()
        

    
if __name__ == "__main__":
    mountaincar = mountaincar()

    #mountaincar.run_tool_box()
    mountaincar.build_MDP_from_file()



    opt = mountaincar.learn_from_demo_file()
    policy = opt['policy']
    mountaincar.test(policy = policy)
    mountaincar.synthesize_from_demo_file(safety = 0.1)
    '''

    policy = mountaincar.read_policy_file('./data/policy_mountaincar')
    real = raw_input("AL policy. Ready? [Y/N]")
    while real == 'y' or real == 'Y':
        mountaincar.test(policy = policy)
        real = raw_input("AL policy. Ready? [Y/N]")
    

    policy = mountaincar.read_policy_file('./data/policy_mountaincar_0.1')
    real = raw_input("SAAL policy. Ready? [Y/N]")
    while real == 'y' or real == 'Y':
        mountaincar.test(policy = policy)
        real = raw_input("SAAL policy. Ready? [Y/N]")
    '''
    

    

