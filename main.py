"""
Worksheet 1: Simple Genetic Algorithm.
"""

import random
import copy
import matplotlib.pyplot as plt
import numpy as np

## Initial Population
N = 10 # N is the number of genome in a chromosome of an Individual. It will also be used as a random.randint(1, N) for a crossover point.
P = 50 # P is the size of the population.
MUTRATE = 0.5 # Mutation rate of the ? 0.001

# Record the best fitness of an individual from a population
best_fitness = None
mean_fitness = []

class Individual:    
    def __init__(
        self
    ):
        self.gene = [0]*N
        self.fitness = 0
        
print("\n**** Genetic Algorithm Summary ****")
        
# INITIALISE population with random candidate solutions
population = []
for _ in range(0, P):
    temp_gene = [random.randint(0, 1) for j in range(0, N)]
    new_individual = Individual()
    new_individual.gene = temp_gene.copy()
    population.append(new_individual)

print("Intialised population size: %s... \nWith chromosome length: %s" % (len(population), N))

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
        
## EVALUATE each candidate
def fitness(individual):
    """ Fitness function - loop through genome to get the total fitness. """
    
    ## Count all the binary 1s in a chromosome. 
    utility = 0
    for i in range(N):
        utility = utility + individual.gene[i]
    return utility
        
# # Set population[i] to each iterative fitness total.
# for i in range(P):
#     ## Set the fitness for each individual in population
#     population[i] = fitness(population[i])
        
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
population.clear()
print("\nIntialised population being mutated...")

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
                
        ## Append genome into a chromosome set of genes
        newind.gene.append(gene)
        
    ## You must then append new individual or overwrite offspring
    population.append(newind)
    
    ## Set the fitness for each offspring in population
    newind.fitness = fitness(newind)
    
    
    if best_fitness is None:
        best_fitness = newind.fitness
        
    if newind.fitness > best_fitness:
        best_fitness = newind.fitness
        
    if mean_fitness is None:
        mean_fitness = newind.fitness
        
    else:
        mean_fitness.append(newind.fitness)

    population[i] = fitness(population[i])

## Get best fitness and mean fitness
mean_fitness = np.mean(mean_fitness)
best_fitness = best_fitness








print("Best Fitness: %s \nMean Fitness: %s" % (best_fitness, mean_fitness))



# print(population)
plt.plot(population)
plt.xlabel("Generation")
plt.ylabel("Fitness")
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

# https://www.researchgate.net/post/How-can-one-get-the-average-and-best-fitness-in-genetic-algorithm
# https://softwareengineering.stackexchange.com/questions/177474/matlab-best-fitness-vs-mean-fitness-initial-range