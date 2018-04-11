file = open('grid_world.dtmc', 'r')
S = 898
p_tot = 0.0
ss = 0
lines = file.readlines()[4:]
for line in lines:
    tran = line.split('\n')[0].split(' ')
    s = int(tran[0])
    s_ = int(tran[1])
    p = float(tran[2])
    #print([s, s_, p, 0, 0, 0, 0, ss, p_tot])
    if s == ss:
        p_tot += p
    else:
        p_tot = p
        ss = s

    if p_tot <= 0.0 or p_tot > 1.0:
        print("state %d, %f > 1.0 or %f < 0.0? %s" % (ss, p_tot, p_tot, p_tot > 1.0 or p_tot < 0.0))
file.close()
            
