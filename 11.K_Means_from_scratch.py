import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

X = np.array([
	[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11]
	])

plt.scatter(X[:,0],X[:,1],s=150)
plt.show()

color =["g.","r.","c.","b.","k.","y."]