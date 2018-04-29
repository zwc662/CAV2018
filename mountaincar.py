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

import ast
import gym
import re
import cProfile
import math
from grids import grids

#import MDP

env = gym.make('MountainCar-v0')

class mountaincar(grids, object):
    def __init__(self, steps = 200, combo = 2, safety = 0.5):
        if sys.version_info[0] >= 3:  
            super().__init__()
        else:
            super(mountaincar, self).__init__()

        self.grids = [20, 16]

        ranges = [[-1.2, 0.6], [-0.07, 0.07]] 
        self.build_threshes(ranges)

        num_S = 1
        for i in self.grids:
            num_S *= i
        num_S += 2

        num_A = env.action_space.n;

        self.M = mdp(num_S, num_A)

        self.set_unsafe()

        self.safety = safety

    
        self.maxepisodes = 100000

        self.steps = steps
        self.combo = combo

        self.steps = int(self.steps/(self.combo + 1)) + 1

        self.opt = {}



    def set_initial_opt(self):
        self.opt['policy'] = np.zeros([len(self.M.S), len(self.M.A)])
        self.opt['policy'][:, 1] = 1.0

        self.M.set_policy(self.opt['policy'])
        self.opt['mu'] = self.M.expected_features_manual()[-2]
        self.opt['theta'] = list()
    
    def set_unsafe(self):
        self.M.unsafe = []
        for s in self.M.S[:-2]:
            coords = self.index_to_coord(s)
	    if (coords[0] <= 1 * (self.grids[0] - 2)/18.0 \
            and coords[1] <= (7 - 4) * (self.grids[1] - 2)/14.0) \
            or (coords[0] >=  (18 - 1) * (self.grids[0] - 2)/18.0 \
            and coords[1] >= (7 + 4) * (self.grids[1] - 2)/14.0):
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


    def build_features(self):
        f = 18
        self.M.features = np.zeros([len(self.M.S), f + 2])
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

        self.set_initial_opt()


    def learn_from_demo_file(self, steps = None):
        if steps is None:
            steps = self.steps
        learn = cegal(self.M, max_iter = 50, epsilon = 10)
        learn.exp_mu = learn.read_demo_file('./data/demo_mountaincar') 
        print(learn.exp_mu)
        opt = super(cegal, learn).iteration(learn.exp_mu) 
        prob = learn.model_check(opt['policy'], steps)
    
        print("\n>>>>>>>>Apprenticeship Learning learnt policy weight vector:")
        print(opt['theta'])
        print("\nFeature vector margin: %f" % opt['diff'])
        print("\nGiven safety spec:\nP=? [U<= 66 ((position < -1.1 && velocity < -0.04)||(position > 0.5 && velocity > 0.04))]\n")
        print("\nPRISM model checking the probability of reaching unsafe states: %f\n" % prob)
        opt['prob'] = prob
        
        file = open('./data/log', 'a')
        file.write("\n>>>>>>>>Apprenticeship Learning learns a policy.")
        file.write("\nGiven safety spec:\nP=? [U<= 66 ((position < -1.1 && velocity < -0.04)||(position > 0.5 && velocity > 0.04))]\n")
        file.write("\nPRISM model checking the probability of reaching\
 the unsafe states: %f\n" % prob)
        file.close()

        while True:
            test = raw_input('1. Play learnt policy visually\n\
2. Run AL policy to collect statistical data\n3. Store policy \n4. Quit\nInput the selection:\n')
            if test == '1':
                self.episode(policy = opt['policy'], steps = steps)
            elif test == '2':
		file = open('./data/log', 'a')
		file.write('\nTest AL policy\n')
                file.close()
                self.test(policy = opt['policy'])
            elif test == '3':
                self.write_policy_file(policy = opt['policy'], path = './data/policy_mountaincar')
            elif test == '4':
                break
            else:
                print("Invalid input")
        return opt


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
        return self.M.policy
        
    def learn_from_policy_file(self):
        learn = apirl(self.M, max_iter = 50)
        self.read_policy_file()
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
                if not (demo or performance or safe):
    	            env.render()

	        if done:
                    path.append([t, s, a, s_])
                    print("End after steps %d" % t)
                    return path


                if unsafes[s_]:
                    path.append([t, s, a, s_])
                    print("Reach unsafe state after steps %d" % t)
                    if demo or safe:
                        return list()

      	    o_, rew, done, info = env.step(a);
      	    # See where we arrived
     	    s_ = self.observation_to_index(o_);

            path.append([t, s, a, s_])

            if unsafes[s_]:
                print("Reach unsafe state after steps %d" % t)
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
		    cut = raw_input("Click Enter!!!")
		    cut = False
		cut = raw_input("Click Enter for one last time!!!")
        print("Used up %d" % len(path))
        return path

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

        while i_episode < episodes:
            print("Episode %d" % i_episode)
            path = self.episode(safe = True, policy = policy)
            if len(path) == 0:
                dead += 1
            i_episode += 1
        dead = dead/episodes

        avg = 0.0
        i_episode = 0
        while i_episode < episodes:
            print("Episode %d" % i_episode)
            path  = self.episode(performance =True, policy = policy)
            if len(path) <= 0:
                raise "Error: path length <= 0"
            avg += len(path)
            i_episode += 1
        avg /= episodes
        if avg > 200/(1+self.combo):
	   avg = 200/(1+self.combo)

        print('Unsafe ratio: %f' % dead)
        print("Average step length: %f" % avg)

        file = open('./data/log', 'a')
        file.write('Unsafe ratio: %f\n' % dead)
        file.write("Average step length: %f\n" % avg)
        file.close()


    def synthesize_from_demo_file(self, safety = None, steps = None, path = './data/demo_mountaincar'):
        if safety is None:
            safety = self.safety
        if steps is None:
            steps = self.steps

        learn = cegal(self.M, max_iter = 50, safety = safety, steps = steps, epsilon = 10)
        exp_mu = learn.read_demo_file(path)

        opt, opt_ = self.synthesize(learn = learn, exp_mu = exp_mu, safety = safety, steps = steps)
        while True:
            #n = raw_input('\n1. Try AL policy, 2. Try CEGAL policy, 3. Quit\nInput your selection:\n')
            n = '2'
            if n == '1':
		file = open('./data/log', 'a')
		file.write('\nTest AL policy\n')
                file.close()
                policy = opt_['policy'].copy()
            elif n == '2':
		file = open('./data/log', 'a')
		file.write('\nTest CEGAL policy\n')
                file.close()
                policy = opt['policy'].copy()
            elif n == '3':
                break
            else:
                print("Invalid")
                continue
            while True:
                #test = raw_input('\n1. Play learnt policy visually\n\
#2. Run policy to collect statistical data\n3. Store policy\n4. Quit\nInput the selection:\n')
                test = '2'
                if test == '1':
                    self.episode(policy = policy, steps = steps)
                elif test == '2':
                    self.test(policy = policy)
                    break
                elif test == '3':
                    if n == '1':
                        self.write_policy_file(policy = policy, path = './data/policy_mountaincar')
                    else:
                        self.write_policy_file(policy = policy, path = './data/policy_mountaincar_' + str(safety))
                elif test == '4':
                    break
                else:
                    print("Invalid input")
            break
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

        print("\nLearning result for safety specification:\n")
        print("\nP<=" + str(safety) + " [U<= 66 (position < -1.1 && velocity < -0.04)||(position > 0.5 && velocity > 0.04)]\n")

        print("\n>>>>>>>>Safety-Aware Apprenticeship Learning learnt policy weight vector:")
        print(opt['theta'])
        print("\nFeature vector margin: %f" % opt['diff'])
        print("\nPRISM model checking result: %f\n" % opt['prob'])
        
        file = open('./data/log', 'a')
        file.write("\n\n\nLearning result for safety specification:\n")
        file.write("\nP<=" + str(safety) + " [U<= 66 (position < -1.1 && velocity < -0.04)||(position > 0.5 && velocity > 0.04)]\n")

        file.write("\n>>>>>>>>Safety-Aware Apprenticeship Learning learnt policy")
        #print("\nFeature vector margin: %f" % opt['diff'])
        file.write("\nPRISM model checking result: %f\n" % opt['prob'])
        file.close()

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
                        #path.append([(self.combo + 1) * t + i, s, a, s_])
                        #self.M.T[a][s, s_] += 1.0

                    if unsafes[s_]:
                        dead = True

                    if done:
                        break

                o_, rew, done, info = env.step(a);
                # See where we arrived
                s_ = self.observation_to_index(o_);

                self.M.T[a][s, s_] += 1.0
                    #path.append([(self.combo + 1) * t + self.combo, s, a, s_])
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
            if True:
                if t >= self.steps - 1:
                    rew = -100
                    tag ='xxx'
                    streak.append(0)
                else:
                    rew = self.steps - t
                    tag = '###';
                    streak.append(1)
                    win += 1;

                    if not dead:
                        using += 1
                #if True:
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
                #if sum(streak) < 80:
                    paths = list()
                    using = 0
                    self.M.starts = list()
                    pass
            episodes = 0

            if using >= 2000:# and i_episode > maxepisodes/2:
                break

            print "Episode {} finished after {} timecoords {} win:{} use:{}".format(i_episode, len(path), tag, win, using)

    
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

        '''
        file = open('./data/mdp_mountaincar', 'w')
        for s in self.M.S:
            for a in self.M.A:
                for s_ in self.M.S:
                    file.write(str(s) + ' ' 
                                + str(a) + ' ' 
                                + str(s_) + ' ' 
                                + str(self.M.T[a][s, s_]) + '\n')
        file.close()
        '''
        
        file = open('./data/unsafe_mountaincar', 'w')
        for s in self.M.unsafes:
            file.write(str(s) + '\n')
        file.close()

        file = open('./data/state_space_mountaincar', 'w')
        file.write('states\n' + str(len(self.M.S)) + '\nactions\n' + str(len(self.M.A)))
        file.close()

        file = open('./data/demo_mountaincar', 'w')
        for path in paths:
            for t in range(len(path)):
                file.write(str(path[t][0]) + ' ' 
                            + str(path[t][1]) + ' ' 
                            + str(path[t][2]) + ' '
                            + str(path[t][3]) + '\n')
        file.close()

    def model_check(self, policy, safety = None, steps = None):
        if steps is None:
            steps = self.steps
        if safety is None:
            safety = self.safety
        learn = cegal(self.M, max_iter = 50, safety = safety, steps = steps)
        prob = learn.model_check(policy, steps)
        print("Unsafe probability: %f" % prob)
        
        

    
if __name__ == "__main__":
    safety = 0.4
    steps = 200
    combo = 2

    mountaincar = mountaincar(safety = safety, combo = combo, steps = steps)

    #mountaincar.run_tool_box() 
    mountaincar.build_MDP_from_file()
    #mountaincar.read_policy_file(path = './data/policy_mountaincar_0.2')
    #mountaincar.test(mountaincar.M.policy)
    opt = mountaincar.learn_from_demo_file()
    safety = (int(opt['prob'] * 10) - 1)/10.0

    for i in [0.5, 0.4, 0.3, 0.2]:
        mountaincar.synthesize_from_demo_file(i)

