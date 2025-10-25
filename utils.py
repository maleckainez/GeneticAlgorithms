def load_data(path):
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
        file_content = f.read()
        f.close()
    list_of_items = file_content.split('\n')
    items = {}
    for i in range(len(list_of_items)):
        items[i] = [int(x) for x in list_of_items[i].split(" ")]
    return items


def create_population(population_size, genome_length):
    """
    Generates an initial binary population for Genetic Algorithm.
    Each individual is a binary genome of length `genomeLength`.

    Gene value 1 means the item is taken, 0 means it is not.
     :param population_size: number of the individuals in the population (height of the numpy matrix)
     :type population_size: int
     :param genome_length: number of genes per individual (must equal number of items)
     :type genome_length: int
     :return: 2D array of shape (populationSize, genomeLength) with binary values in {0,1}
     :rtype numpy ndarray of ints
     """
    import numpy as np
    return np.random.randint(2, size=(population_size, genome_length))
