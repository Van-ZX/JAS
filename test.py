import numpy as np
import random

rand_ind = np.array(random.sample(range(10), k = 10))
conv_ind = rand_ind[0 : int((1-0.2)*10)]
adv_ind = rand_ind[int((1-0.2)*10) : 10]
print(rand_ind)
print(conv_ind)
print(adv_ind)
