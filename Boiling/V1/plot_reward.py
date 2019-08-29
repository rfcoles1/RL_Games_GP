import numpy as np
import matplotlib.pyplot as plt

Results = np.loadtxt("Results.txt")

n = np.shape(Results)[0]

mean = np.zeros(n)
maxi = np.zeros(n)

for i in range(n):
    mean[i] = np.mean(Results[i])
    maxi[i] = np.max(Results[i])

plt.plot(np.arange(n), mean, label = 'Mean')
plt.plot(np.arange(n), maxi, label = 'Maximum')
plt.xlabel('Generation')
plt.ylabel('Score')
plt.legend()
plt.show()

