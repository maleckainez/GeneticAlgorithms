def loadData(path):
    """
     This module takes path to the file that contains data in format:
     <value> <weight>
     each line represents different item
     :param path: path to the text file containing data
     :type path: str
     :return: dictionary {key: [value] [weight]}
     :rtype dict
     """
    with open(path) as f:
        fileContent = f.read()
        f.close()
    listOfItems = fileContent.split('\n')
    items = {}
    for i in range(len(listOfItems)):
        items[i] = [int(x) for x in items[i].split()]
    return items


def createPopulation(populationSize, genomeLength):
    """
    Generates an initial binary population for Genetic Algorithm.
    Each individual is a binary genome of length `genomeLength`.

    Gene value 1 means the item is taken, 0 means it is not.
     :param populationSize: number of the individuals in the population (height of the numpy matrix)
     :type populationSize: int
     :param genomeLength: number of genes per individual (must equal number of items)
     :type genomeLength: int
     :return: 2D array of shape (populationSize, genomeLength) with binary values in {0,1}
     :rtype numpy ndarray of ints
     """
    import numpy as np
    return np.random.randint(2, size=(populationSize, genomeLength))
