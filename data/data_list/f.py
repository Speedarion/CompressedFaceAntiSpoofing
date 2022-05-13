import pandas as pd

csv1 = pd.read_csv('CASIA-TRAIN.csv', header=None)
f1 = list(csv1.get(0))

csv2 = pd.read_csv('CASIA-TRAIN.csv', header=None)
f2 = list(csv2.get(0))


F1 = []
F2 = []
for i in range(len(f1)):
    f1[i] = f1[i].upper()

for i in range(len(f2)):
    f2[i] = f2[i].upper()

for s in f1:
    if s not in f2:
        print(1)
        F1.append(s)

for s in f2:
    if s not in f1:
        print(1)
        F2.append(s)

import IPython; IPython.embed()