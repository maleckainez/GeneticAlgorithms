from methods.utils import create_population
def calc_fitness_score(ITEMS):
    num_of_items = len(ITEMS)
    fitnessScore = {}
    populationAmount = 2 ** round(num_of_items / 5)
    populacja1 = create_population(populationAmount,num_of_items)
    for i in range(0, populationAmount):
        tempVolume = 0
        tempScore = 0
        for j in range(0, num_of_items):
            if tempVolume > 1800:
                tempScore = 0
                break
            tempScore += populacja1[i][j] * ITEMS[j][0]
            tempVolume += populacja1[i][j] * ITEMS[j][1]
        fitnessScore[i] = int(tempScore)
    return fitnessScore
