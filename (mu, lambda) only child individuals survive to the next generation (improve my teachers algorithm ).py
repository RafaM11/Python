# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 19:25:24 2021

@author: Javi
"""

import random
import math
from matplotlib import pyplot as plt

def apply_function(individual):
    x = individual["x"]
    y = individual["y"]
    firstSum = x**2.0 + y**2.0
    secondSum = math.cos(2.0*math.pi*x) + math.cos(2.0*math.pi*y) 
    n = 2
    return -(-20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e)

def generate_population(size, x_boundaries, y_boundaries):
    lower_x_boundary, upper_x_boundary = x_boundaries
    lower_y_boundary, upper_y_boundary = y_boundaries

    population = []
    for i in range(size):
        individual = {
            "x": random.uniform(lower_x_boundary, upper_x_boundary),
            "y": random.uniform(lower_y_boundary, upper_y_boundary),
        }
        population.append(individual)

    return population

def select_by_roulette(sorted_population, fitness_sum):
    offset = 0
    normalized_fitness_sum = fitness_sum

    lowest_fitness = apply_function(sorted_population[0])
    if lowest_fitness < 0:
        offset = -lowest_fitness
        normalized_fitness_sum += offset * len(sorted_population)

    draw = random.uniform(0, 1)

    accumulated = 0
    for individual in sorted_population:
        fitness = apply_function(individual) + offset
        probability = fitness / normalized_fitness_sum
        accumulated += probability

        if draw <= accumulated:
            return individual
        
def sort_population_by_fitness(population):
    return sorted(population, key=apply_function)


def crossover(individual_a, individual_b):
    xa = individual_a["x"]
    ya = individual_a["y"]

    xb = individual_b["x"]
    yb = individual_b["y"]

    return {"x": (xa + xb) / 2, "y": (ya + yb) / 2}


def mutate(individual):
    next_x = individual["x"] + random.gauss(0, 0.1)#random.uniform(-0.05, 0.05)
    next_y = individual["y"] + random.gauss(0, 0.1)#random.uniform(-0.05, 0.05)

    lower_boundary, upper_boundary = (-4, 4)

    # Guarantee we keep inside boundaries
    next_x = min(max(next_x, lower_boundary), upper_boundary)
    next_y = min(max(next_y, lower_boundary), upper_boundary)

    return {"x": next_x, "y": next_y}


def make_next_generation(previous_population):
    next_generation = []
    sorted_by_fitness_population = sort_population_by_fitness(previous_population)
    population_size = len(previous_population)
    fitness_sum = sum(apply_function(individual) for individual in population)

    for i in range(population_size):
        father = None
        mother = None
        while(father == mother):
            father = select_by_roulette(sorted_by_fitness_population, fitness_sum)
            mother = select_by_roulette(sorted_by_fitness_population, fitness_sum)
            if (father != mother):
                break
        
        print('Father: ', i, father,  ' fitness :', apply_function(father))
        print('Mother: ', i, mother,  ' fitness :', apply_function(mother))

        individual = crossover(father, mother)
        print('Individual: ', i, individual, ' fitness :', apply_function(individual))
        individual = mutate(individual)
        print('Individual: ', i, individual, ' fitness  :', apply_function(individual))
        
        best_parent = None
        if (apply_function(father) > apply_function(mother)):
            best_parent = father
        else:
            best_parent = mother
            
        if (apply_function(individual) > apply_function(best_parent)):
            next_generation.append(individual)
        else:
            next_generation.append(best_parent)

    return next_generation

# =============================================================================
# MAIN
# =============================================================================

generations = 5
population = generate_population(size=10, x_boundaries=(-5, 5), y_boundaries=(-5, 5))

i = 1
bestFitness = []
while True:
    
    print(str(i))

    for individual in population:
        print(individual, apply_function(individual))

    if i == generations:
        break

    i += 1

    population = make_next_generation(population)
    best_individual = sort_population_by_fitness(population)[-1]
    bestFitness.append(apply_function(best_individual))
    
best_individual = sort_population_by_fitness(population)[-1]
plt.plot(bestFitness)

print("\nFINAL RESULT")
print(best_individual, apply_function(best_individual))