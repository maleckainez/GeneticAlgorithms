import numpy as np
np.random.seed(2137)
f = open("dane AG 2/large_scale/knapPI_1_100_1000_1")
content = f.read()
genLen = content.count('\n')
listaPrzedmiotow = content.split('\n')
slownik = {}
for i in range(len(listaPrzedmiotow)):
     slownik[i] = listaPrzedmiotow[i].split(" ")
print(slownik)