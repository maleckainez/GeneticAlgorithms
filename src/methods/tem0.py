import random

random.seed(2137)

f = open("../../dane AG 2/low-dimensional/f1_l-d_kp_10_269")
content = f.read()
nLines = content.count("\n")


def createGenome(nLines):
    chromosome = []
    for i in range(0, nLines):
        chromosome.append(random.randint(0, 1))
    print(chromosome)


for i in range(0, 10):
    createGenome(nLines)
