import numpy as np

file = "data/loop_antenna_2.s1p"
header = 12
footer = 0
every = 10
comments = '!'
delimiter = ' '

data =  np.loadtxt(open(file,'rt').readlines()[header::every], comments=comments,delimiter=delimiter)
np.savetxt(f"data/antenna_every10.s1p",data, fmt='%f')
