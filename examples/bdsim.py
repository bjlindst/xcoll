%autoreload 2
import bdsim
import xcoll as xc
import xtrack as xt
import matplotlib.pyplot as plt
import numpy as np
import xpart as xp
import os
os.chdir('/tmp')
 
p_ref = xt.Particles(pdg_id=2212, p0c=6.8e12)


coll1 = xc.BdsimElement(length=0.6, jaw=0.001, material='W')

xc.bdsim.engine.start(particle_ref=p_ref, elements=coll1)

npart = 10000
sigma = 1.e-3   # 1 mm, in metres

part = xp.Particles(
    x=np.random.normal(0, sigma, npart),
    px=np.zeros(npart),
    y=np.random.normal(0, sigma, npart),
    py=np.zeros(npart),
    zeta=np.zeros(npart),
    delta=np.zeros(npart),
    p0c=p_ref.p0c[0],
    pdg_id=p_ref.pdg_id[0],
    _capacity=2*npart)

part0 = part.copy()
part1 = part.copy()

print(xc.bdsim.engine._element_index)
name = next(iter(xc.bdsim.engine._element_index))
xc.bdsim.engine._link.TrackXSuite(0, name, part0, p_ref.p0c[0]*1e-6)
part0.beta0 = np.ones(len(part0.beta0)) * part0.beta0[0]
part0.update_delta(part0.delta)

x_alive = part0.x[part0.state == 1]      # survived
x_lost  = part0.x[part0.state == -333]   # lost on collimator
y_alive = part0.y[part0.state == 1]      # survived
y_lost  = part0.y[part0.state == -333]   # lost on collimator

bins = np.linspace(-5e-3, 5e-3, 200)

plt.figure()
plt.hist(x_alive, bins=bins, color='green', alpha=0.6, label='alive (state=1)')
plt.hist(x_lost,  bins=bins, color='red',   alpha=0.6, label='lost (state=-333)')
plt.xlabel('x [m]')
plt.ylabel('count')
plt.legend()
plt.show()

plt.figure()
plt.hist(y_alive, bins=bins, color='green', alpha=0.6, label='alive (state=1)')
plt.hist(y_lost,  bins=bins, color='red',   alpha=0.6, label='lost (state=-333)')
plt.xlabel('y [m]')
plt.ylabel('count')
plt.legend()
plt.show()

coll1.track(part1)

x_alive = part1.x[part1.state == 1]      # survived
x_lost  = part1.x[part1.state == -333]   # lost on collimator
y_alive = part1.y[part1.state == 1]      # survived
y_lost  = part1.y[part1.state == -333]   # lost on collimator

bins = np.linspace(-5e-3, 5e-3, 200)

plt.figure()
plt.hist(x_alive, bins=bins, color='green', alpha=0.6, label='alive (state=1)')
plt.hist(x_lost,  bins=bins, color='red',   alpha=0.6, label='lost (state=-333)')
plt.xlabel('x [m]')
plt.ylabel('count')
plt.legend()
plt.show()

plt.figure()
plt.hist(y_alive, bins=bins, color='green', alpha=0.6, label='alive (state=1)')
plt.hist(y_lost,  bins=bins, color='red',   alpha=0.6, label='lost (state=-333)')
plt.xlabel('y [m]')
plt.ylabel('count')
plt.legend()
plt.show()    
