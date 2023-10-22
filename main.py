"""
Worksheet 1: Simple Genetic Algorithm.
"""

import random
import copy
import matplotlib.pyplot as plt
import numpy as np

## Initial Population
N = 10 # N is the number of genome in a chromosome of an Individual. It will also be used as a random.randint(1, N) for a crossover point.
P = 4 # P is the size of the population.

class Individual:    
    def __init__(
        self
    ):
        self.gene = [0]*N
        self.fitness = 0


population = []
        
# INITIALISE population with random candidate solutions
for _ in range(0, P):
    temp_gene = [random.randint(0, 1) for j in range(0, N)]
    new_individual = Individual()
    new_individual.gene = temp_gene.copy()
    population.append(new_individual)

## EVALUATE each candidate
def fitness(individual):
    """ Fitness function - loop through genome to get the total fitness. """
    
    ## Count all the binary 1s in a chromosome. 
    utility = 0
    for i in range(N):
        utility = utility + individual.gene[i]
    return utility

## Tournament Selection
offspring = []  
for i in range(0, P):
    parent_one = random.randint(0, P - 1)
    parent_two = random.randint(0, P - 1)
    offspring_one = copy.deepcopy(population[parent_one])
    offspring_two = copy.deepcopy(population[parent_two])

    if offspring_one.fitness > offspring_two.fitness: # Compare the attributes of a generator object
        offspring.append(offspring_one)
    else:
        offspring.append(offspring_two)
        
## Crossover
toff1 = Individual() 
toff2 = Individual() 
temp = Individual() 
for i in range(0, P, 2):
    toff1 = copy.deepcopy(offspring[i]) 
    toff2 = copy.deepcopy(offspring[i+1]) 
    temp = copy.deepcopy(offspring[i]) 
    crosspoint = random.randint(1, N)
     
    for j in range(crosspoint, N): 
        toff1.gene[j] = toff2.gene[j] 
        toff2.gene[j] = temp.gene[j]  
        
    offspring[i] = copy.deepcopy(toff1)
    offspring[i+1] = copy.deepcopy(toff2)


## Mutation
for i in range(0, P): 
    newind = Individual(); 
    newind.gene = [] 
    for j in range(0, N): 
        gene = offspring[i].gene[j] 
        mutprob = random.random() 
        if mutprob < MUTRATE: 
            if (gene == 1): 
                gene = 0 
            else: 
                gene = 1 
        newind.gene.append(gene) 

## Set population[i] to each iterative fitness total.
for i in range(P):
    ## Set the fitness for each individual in population
    population[i] = fitness(population[i])
    
np_population = np.array(population)

plt.plot(np_population)
plt.ylabel("Population Fitness")
plt.xlabel("Individual Positional Number")
plt.show()



# Step of Genetic Algorithm:
# (1) Start: Generate random population of n chromosomes (suitable solutions for the problem).
# (2) Fitness: Evaluate the fitness f (x) of each chromosome x in the population.					
# (3) New Population: Create a new population by repeating the following steps until a new population is complete						
#   (a) Selection: Select two parent chromosomes from a population according 
# 		to their fitness (the better fitness, the higher the chance to be selected).				
# 	(b) Crossover: With a crossover probability, cross over the
# 		parents to form a new offspring (child). If no crossover
# 		was performed, offspring is an exact copy of parents.				
# 	(c) Mutation: With a mutation probability, mutate new offspring at each 
# 		locus (position in chromosome).					
# (d) Accepting: Place new offspring in a new population.					
# (4) Replace: Use newly generated population for a further run of algorithm.					
# (5) Test: If the end condition is satisfied, stop, and return the best solution in current population					
# (6) Loop: Go to step 2