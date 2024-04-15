import numpy as np

a = 50.0

bin = np.array([80, 30 ,50, 60 ,7])

mask = bin > a
print(bin[mask])