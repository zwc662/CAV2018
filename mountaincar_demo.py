import scipy.sparse as sparse
import gym
import cProfile
import math
import numpy as np
import util
import re
from grids import grids
# In order to run this script you must download and compile
# the following library: https://github.com/Svalorzen/AI-Toolbox
# Building it will create an MDP.so file which you can then
# include here.
import MDP

# Number of distretized pieces for each observation component
coords = [38, 30]
# We disregard the cart position on the screen to make learning
# faster
threshes = np.array([[-1.2, 0.6], [-0.07, 0.07]]);
maxepisodes = 10000 

gamma = 0.99
combo = 2
steps = int(200/(combo+1))
if steps * (combo + 1) < 200:
	steps = steps + 1
# Gym parameters
render = 0;
record = 0;
recordfolder = './mountaincar'


env = gym.make('MountainCar-v0')
print "hehe"

# Number of distretized pieces for each observation component
epsilon = 1e-5
order = 5
iteration = 20
# We disregard the cart position on the screen to make learning
# faster

# Gym parameters

# Action space is 2, State space depends on coords
A = env.action_space.n;
print A
#S = coords**4 * env.observation_space.shape[0];
#S = env.observation_space.shape[0] 
S = 1
for coord in coords:
	S *= coord
S += 2
print S



grids = grids()
grids.grids = [38, 30]
ranges = [[-1.2, 0.6], [-0.07, 0.07]] 
grids.build_threshes(ranges)

def observationToState(o, thresh):
    return grids.observation_to_index(o)

unsafes = np.zeros([S]).astype(bool)
for s in range(S - 2):
    coords = grids.index_to_coord(s)
    if (coords[0] <= 6 and coords[1] <= 6) \
        or (coords[0] >= grids.grids[0] - 7  \
        and coords[1] >= grids.grids[1] - 7):
        unsafes[s] = True


starts = np.zeros([S]).astype(bool)

transitions = list()
for a in range(A):
    transitions.append(np.zeros([S, S]))
    
paths = list()
# We are not going to assume anything here. We are just going to
# approximate the observation space in a finite number of states.
# In particular, we approximate each vector component in 4 coords.
# If we discard the first component (the cart position on the
# screen) we can learn faster, but adding it still works.
# Then we use PrioritizedSweeping in order to extract as much
# information as possible from each datapoint. Finally we select
# actions using a softmax policy.


exp = MDP.SparseExperience(S, A);
model = MDP.SparseRLModel(exp, gamma);
solver = MDP.PrioritizedSweepingSparseRLModel(model, 0.1, 500);
policy = MDP.QGreedyPolicy(solver.getQFunction());


if record:
    env.monitor.start(recordfolder)

# We use the following two variables to track how we are doing.
# Read more at the bottom

            

using = 0

real = raw_input('Setting up MDP? [Y/N]')
while real == 'Y' or real == 'y':
	episodes=0
	win = 0
	streak = list()
        maxepisodes = 10000	
	for i_episode in xrange(maxepisodes):
                path = list()

    		o = env.reset()
                s_i = observationToState(o, threshes)

		dead = False
                rec = steps
                done = False 

    		for t in xrange(steps):
        		if render or i_episode == maxepisodes - 1:
        	    		env.render()

        		# Convert the observation into our own space
        		s = observationToState(o, threshes);
        		# Select the best action according to the policy
        		a = policy.sampleAction(s)
        		# Act
			for i in range(combo):
				o1, rew, done, info = env.step(a);
        			s1 = observationToState(o1, threshes);
				#transitions[s, a, s1] += 1.0
                                if unsafes[s1]:
                                    dead = True
        		# See where we arrived
			o1, rew, done, info = env.step(a);
        		s1 = observationToState(o1, threshes);

			transitions[a][s, s1] += 1.0
                        path.append([t, s, a, s1])

                        if unsafes[s1]:
                            dead = True

        		if done:
            			break;

        # Record information, and then run PrioritizedSweeping
        		exp.record(s, a, s1, rew);
       		 	model.sync(s, a, s1);
		        solver.stepUpdateQ(s, a);
		        solver.batchUpdateQ();

		        o = o1;

   		#	if render or i_episode == maxepisodes - 1:
      		#		env.render()

    		tag = '   ';
    # Here we have to set the reward since otherwise rewards are
    # always 1.0, so there would be no way for the agent to distinguish
    # between bad actions and good actions.

    		if t < steps - 1:
       		    tag = '###';
      		    win += 1;
		    rew = steps - t
        	    streak.append(1)
                    if not dead:
                        using += 1
                        paths.append(path)
                        starts[s_i] = True
  		else:
			rew = -200
		        streak.append(0)


  		if len(streak) > 100:
      			streak.pop(0)

    		episodes +=1;
    		exp.record(s, a, s1, rew);
    		model.sync(s, a, s1);
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
	                exp = MDP.SparseExperience(S, A);
       			model = MDP.SparseRLModel(exp, gamma);
        		solver = MDP.PrioritizedSweepingSparseRLModel(model, 0.1, 500);
      	  		policy = MDP.QGreedyPolicy(solver.getQFunction());
		    if sum(streak) < 80:
			using = 0
                        paths = list()
                        starts = (0.0 * starts).astype(bool)

		    episodes = 0

		if i_episode > 5000 and using > 1000:
			break

	    	print "Episode {} finished after {} timecoords {} {} {}".format(i_episode, t+1, tag, win, using)
	real = raw_input('Setting up MDP? [Y/N] Or write to file? [W]')

if real == 'W' or real == 'w':
    file = open('./data/start_mountaincar', 'w')
    for s in range(S):
        if starts[s]:
	    file.write(str(s) + '\n')
    file.close()

    
    for a in range(A):
        for s in range(S):
            tot = np.sum(transitions[a][s])
            if tot == 0.0:
                transitions[a][s, s] = 0.0
        transitions[a] = sparse.bsr_matrix(transitions[a])        
        transitions[a] = sparse.diags(1.0/transitions[a].sum(axis = 1).A.ravel()).dot(transitions[a]).todense()

    file = open('./data/mdp_mountaincar', 'w')
    for s in range(S):
        for a in range(A):
            for s_ in range(S):
                file.write(str(s) + ' ' 
                            + str(a) + ' ' 
                            + str(s_) + ' ' 
                            + str(transitions[a][s, s_]) + '\n')
    file.close()

    file = open('./data/unsafe_cartple', 'w')
    for s in range(S):
        if unsafes[s]:
            file.write(str(s) + '\n')
    file.close()

    file = open('./data/state_space_mountaincar', 'w')
    file.write('states\n' + str(S) + '\nactions' + str(A))
    file.close()

    file = open('./data/demo_mountaincar', 'w')
    for path in paths:
        for t in range(len(path)):
            file.write(str(path[t][0]) + ' ' \
                            + str(path[t][1]) + ' ' \
                            + str(path[t][2]) + ' ' \
                            + str(path[t][3]) + '\n')
    file.close()
