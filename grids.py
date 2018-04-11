import ast
import numpy as np

class grids():
    def __init__(self):
        self.threshes = list()
        self.grids = list()
        ###From MSB to LSB######

    def build_grids(self, threshes):
        self.threshes = threshes
        for i in self.threshes:
            self.grids.append(len(i) + 1)

    def build_threshes(self, ranges):
        self.threshes = list()
        for i in range(len(self.grids)):
            self.threshes.append([])
            for j in range(self.grids[i] - 1):
                thresh = ranges[i][0] + \
                        (ranges[i][1] - ranges[i][0]) * \
                        j / (self.grids[i] - 2)
                self.threshes[-1].append(thresh)
        

    def observation_to_coord(self, observation):
        # translate observation to coordinates
        coord = np.zeros([len(self.grids)]).astype(int)
        for i in range(len(observation)):
            for j in range(len(self.threshes[i])):
                if observation[i] >= self.threshes[i][j]:
                    continue
                else:
                    coord[i] = int(j)
                    break
            if observation[i] >= self.threshes[i][-1]:
                coord[i] = len(self.threshes[i])
        return coord

    def observation_to_index(self, observation):
        coord = self.observation_to_coord(observation)
        index = self.coord_to_index(coord)
        return int(index)


    def index_to_coord(self, index):
        coord = self.grids[:]
        for i in range(len(self.grids)):
            coord[len(self.grids) - 1 - i] = index%self.grids[len(self.grids) - 1 - i]
            index = int(index/self.grids[len(self.grids) - 1 - i])
        return coord

    def coord_to_index(self, coord):
        # Translate observation to coordinate by calling the grids
        # Then translate the coordinate to index
        index = 0
        base = 1
        for i in range(len(coord)):
            index += coord[len(coord) - 1 - i] * base 
            base *= self.grids[len(self.grids) - 1 - i]
        return int(index)
        
    '''
    def write_transitions_file(self, data):
        # Preprocess the data set file which contains trajectories
        # Each trajectory is a list in which each element in a list is a list of time step, dict of observations and ...
        # This method translates the observations to indexs
        tuples = []
        file_i = open(data, 'r')
        print("read list file")
        for line_str in file_i.readlines():
            line = ast.literal_eval(line_str)
            time = line[0]
            observation = line[1]
            index = self.observation_to_index(observation)
            action = line[2]
            observation_ = line[3]
            index_ = self.observation_to_index(observation_)
            if self.check_unsafe_index(observation_):
                self.unsafes.append(int(index_))
            if self.check_safe_index(observation_):
                self.targets.append(int(index_))

            tuples.append(
                (str(time) + ' ' + 
                 str(index) + ' ' +
                 str(action) + ' ' +
                 str(index_) + '\n'))
        file_i.close()
        
        file_o = open('./data/transitions', 'w')
        for line_str in tuples:
            file_o.write(line_str)
        file_o.close()
        '''
