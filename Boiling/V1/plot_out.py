import numpy as np
import matplotlib.pyplot as plt

f = open("out.txt", "r")
contents = f.readlines()

def isnumber(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

contents = contents[1:] #remove first line
contents = contents[::2] #remove every other line

Episode = np.zeros(len(contents))
Mean = np.zeros(len(contents))
Max = np.zeros(len(contents))
i = 0
for string in contents:
    string = string.replace(',', '')
    n = [float(s) for s in string.split() if isnumber(s)]
    Episode[i] = n[0]
    Mean[i] = n[1]
    Max[i] = n[2]
    i += 1

plt.plot(Episode, Mean, label = 'Mean')
plt.plot(Episode, Max, label = 'Maximum')
plt.xlabel('Generation')
plt.ylabel('Score') 
plt.legend()
plt.show()
