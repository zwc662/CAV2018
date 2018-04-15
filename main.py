from mountaincar import mountaincar
from cartpole import cartpole
from gridworld import gridworld

def play_gridworld():
    dim = raw_input('\n#######Play nxn gridworld. n = ? Please input an integer or press Enter to use default n = 8\n')
    if str.isdigit(dim):
        dim = int(dim)
    else:
        dim = 8
    grids = gridworld(dim)
    grids.build_mdp()
    
    t = raw_input('\n#######Use Apprenticeship Learning to learn from human demonstration. Maximum step length is ' + str(dim**2) + '.\nPress Enter to continue\n')
    grids.learn_from_human_demo(steps = dim**2)

    print('\n#######Please input the safety threshold p* for the safety specification below.\nP <= p* [U<=' + str(dim**2) + " 'unsafe']\n")
    safety = raw_input("\nPlease input p*=?\n")
    safety = float(safety)
    opt, opt_ = grids.synthesize_from_demo_file(safety)

def play_cartpole():
    print("Cartpole environment:\n\
The task is to maintain the balance of a pole on a cart for as long as possible.\n\
The maximu step length is 200.\n")
    t = raw_input("\nPress Enter to continue\n")
    print("\nExpert demonstration has been stored in ./data/demo_cartpole.\n\
All the episodes reached the maximum 200 step length.\n")
    t = raw_input("\nPress Enter to continue\n")
    cp = cartpole()
    cp.build_MDP_from_file()
    
    print("\nFirst, run Apprenticeship Learning to learn from expert demonstration.\n")
    t = raw_input("\nPress Enter to continue\n")
    opt = cp.learn_from_demo_file()

    print("\nNext, try Safety-Aware apprenticeship learning\n")
    print("\n#######Please input the safety threshold p* for the safety specification below.\nP <= p* [U<= 200 ((position < -0.3 && angle < -20)||(position > 0.3 && angle > 20))]\n")
    safety = raw_input("\nPlease input p*=?\n")
    safety = float(safety)
    opt, opt_ = cp.synthesize_from_demo_file(safety)

def play_mountaincar():
    print("Mountaincar environment:\n\
The task is to drive the car to the right mountaintop as soon as possible.\n\
The maximu step length is 66.\n")
    print("\nExpert demonstration has been stored in ./data/demo_mountaincar.\n\
Average step length is 51.\n\
No unsafe behavior defined as below is performed in any of the episodes.\n")
    print("\nUnsafe behavior: (position < -0.9 && velocity < -0.03)||(position > 0.9 && velocity > 0.03)\n")
    t = raw_input("\nPress Enter to continue\n")
    mc = mountaincar()
    mc.build_MDP_from_file()
    
    print("\nFirst, run Apprenticeship Learning to learn from expert demonstration.\n")
    t = raw_input("\nPress Enter to continue\n")
    opt = mc.learn_from_demo_file()

    print("\nNext, try Safety-Aware apprenticeship learning\n")
    print("\n#######Please input the safety threshold p* for the safety specification below.\nP <= p* [U<= 200 ((position < -0.9 && velocity < -0.03)||(position > 0.9 && velocity > 0.03))]\n")
    safety = raw_input("\nPlease input p*=?\n")
    safety = float(safety)
    opt, opt_ = mc.synthesize_from_demo_file(safety)
            


if __name__ == "__main__":
    index = raw_input('\n\n\nSelect experiment:\n\
    1. gridworld\n\
    2. cartpole\n\
    3. mountaincar\n\
Please input the index:\n')
    index = int(index)
    file = open('./data/log', 'a')
    if index == 1:
        file.write('Experiment: gridworld\n')
        file.close()
        play_gridworld()
    elif index == 2:
        file.write('Experiment: cartpole\n')
        file.close()
        play_cartpole()
    elif index == 3:
        file.write('Experiment: mountaincar\n')
        file.close()
        play_mountaincar()

