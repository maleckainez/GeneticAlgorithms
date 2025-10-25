import numpy as np
np.random.seed(2137)
f = open("dane AG 2/large_scale/knapPI_1_100_1000_1")#("dane AG 2/low-dimensional/f1_l-d_kp_10_269")
content = f.read()
amountOfThings = content.count('\n')
listaPrzedmiotow = content.split('\n')
slownik = {}
for i in range(len(listaPrzedmiotow)):
    slownik[i] = [int(x) for x in listaPrzedmiotow[i].split()]


crossover = 0.5
mutation = 0
max_volume = 18000


if amountOfThings<2:
    raise Exception(f"The result is: {amountOfThings}")
if crossover<0.5 or crossover>1:
    raise Exception("Interbreeding cannot be set below 0.5 or over 1")
if mutation<0 or mutation>0.1:
    raise Exception("Mutation cannot have negative values or be set over 0.1")

def createGenome(population):
    return np.random.randint(2,size=(population,amountOfThings))

def interBreed():
    return 0

def fitnessValue():
    fitnessScore = {}
    populationAmount = (2**round(amountOfThings/5))
    populacja1 = createGenome(populationAmount)
    for i in range(0,populationAmount):
        tempVolume = 0
        tempScore = 0
        for j in range(0,amountOfThings):
            if tempVolume>max_volume:
                tempScore = 0
                break
            tempScore+=populacja1[i][j]*slownik[j][0]
            tempVolume+=populacja1[i][j]*slownik[j][1]
        fitnessScore[i]=int(tempScore)
    return fitnessScore

def selection():
    return 0


wyniki = fitnessValue()
sumaFitness = sum(wyniki.values())
#print(sumaFitness)

#print(wyniki)
#for i in wyniki:
#    if wyniki[i] != 0:
#        print(wyniki[i])
