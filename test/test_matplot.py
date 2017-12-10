import matplotlib.pyplot as plt
import time
import numpy as np

"""
matplot axes 
https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.scatter.html

Figure, subplot, axes 
https://www.labri.fr/perso/nrougier/teaching/matplotlib/#figures-subplots-axes-and-ticks

Scatter markersize 
https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size

update plot 
https://matplotlib.org/devdocs/api/_as_gen/matplotlib.pyplot.pause.html
http://block.arch.ethz.ch/blog/2016/08/dynamic-plotting-with-matplotlib/
"""
#plt.ion()
fig = plt.figure()
#ax = fig.add_axes([0,0,1,1], frameon=False, aspect=1)
ax = fig.add_subplot(111)
ax.axis('equal')
# particles holds the locations of the particles
#agents_sc = ax.scatter([.1,.2],[0,.2],s=15**2)
#prey_sc   = ax.scatter(.5,.5,s = 15**2)

agents_sc = ax.scatter([],[],s=20**2)
prey_sc   = ax.scatter([],[],s = 20**2)
#particles.set_xdata()
#particles.set_ydata()
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
agents_sc.set_offsets([[-.1,-.1], [.1,.3]])
prey_sc.set_offsets([[0,0]])
for ii in range(10):
    print('step{}'.format(ii))
    agents_sc.set_offsets(np.random.uniform(-1,1,[2,2]))
    plt.pause(1e-100)
    #fig.canvas.draw()
    time.sleep(.5)

plt.show()
# rect is the box edge
#plt.show()
