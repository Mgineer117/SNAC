import numpy as np

a = [(1,3), (1,5), (1,3), (1,4)]
print((1,3) in a)

positions = [i for i, value in enumerate(a) if value == (1, 3)]
print(positions)