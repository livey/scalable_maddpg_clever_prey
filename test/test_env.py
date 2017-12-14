from env import Environ
import numpy as np
Env = Environ(3)
ob1, ob2 = Env.reset()

for ii in range(100):
    s = Env.step(np.random.uniform(-1,1,[3,1]),np.random.uniform(-1,1,1))
    print(s)
    if s[4]:
        print('reset')
        Env.reset()
