import numpy as np

file = "data/45d_1.s1p"
header = 19
footer = 1
every = 200
comments = '!'
delimiter = ','

f_L = 3.69117999*1E9
ts = 2 * (freqs - f_L)/f_L
data =  np.loadtxt(open(file,'rt').readlines()[header:-footer:every], comments=comments,delimiter=delimiter)

alpha = np.radians(50)
dist_from_center = 0.98
gammas = gamma_fit_function(ts, [-1000 * dist_from_center*(np.cos(alpha) + 1j* np.sin(alpha)), -dist_from_center*(np.cos(alpha) + 1j* np.sin(alpha))+(np.cos(alpha) + 1j* np.sin(alpha))+0.1, 1000j])
gamma_abs_sq = np.real(gammas)*np.real(gammas) + np.imag(gammas) * np.imag(gammas)


np.savetxt(f"data/made_up.csv",data, fmt='%f')
