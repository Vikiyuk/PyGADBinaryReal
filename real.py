import random
import numpy as np

import benchmark_functions as bf
import pygad
import logging

def fitness_function_Styblinski(ga_instance, solution, solution_idx):
    print(type(solution))
    return -bf.StyblinskiTang(len(solution))(solution)

def fitness_function_Rosenbrock(ga_instance, solution, solution_idx):
    return 1./bf.Rosenbrock(len(solution))(solution.tolist())

def position_crossover(parents, offspring_size, ga_instance):
    alpha = 0.01
    beta = 0.02

    offspring = np.empty(offspring_size)

    for k in range(offspring_size[0]):
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k + 1) % parents.shape[0]
        parent1 = parents[parent1_idx].copy()
        parent2 = parents[parent2_idx].copy()
        
        num_variables = parent1.shape[0]
        for j in range(num_variables):
            if random.random() > 0.5:
                parent1[j] -= alpha
                parent2[j] += beta
            else:
                parent1[j] += alpha
                parent2[j] -= beta

        offspring[k] = parent1 if k % 2 == 0 else parent2

    return offspring

def multiple_crossover(parents, offspring_size, ga_instance):
    population_size = parents.shape[0]
    num_variables = parents.shape[1]
    offspring = np.empty(offspring_size)
    
    for j in range(offspring_size[0]):
        child = np.empty(num_variables)
        for i in range(num_variables):
            alpha = random.randint(0, population_size - 1)
            child[i] = parents[alpha, i]
        offspring[j, :] = child
    
    return offspring

def q_rand_point_crossover(parents, offspring_size, ga_instance):
    n = parents.shape[1]
    offspring = np.empty(offspring_size)

    for idx in range(0, offspring_size[0], 2):
        parent1_idx = idx % parents.shape[0]
        parent2_idx = (idx + 1) % parents.shape[0]
        Xt = parents[parent1_idx].copy()
        Yt = parents[parent2_idx].copy()

        k = random.randint(1, n-1)
        cp = sorted(random.sample(range(1, n), k))

        Xt_1 = np.zeros(n)
        Yt_1 = np.zeros(n)

        przełącz = 0
        for i in range(cp[0]):
            Xt_1[i] = Xt[i]
            Yt_1[i] = Yt[i]

        for j in range(1, k):
            if przełącz == 0:
                for i in range(cp[j - 1], cp[j]):
                    Xt_1[i] = Yt[i]
                    Yt_1[i] = Xt[i]
                przełącz = 1
            else:
                for i in range(cp[j - 1], cp[j]):
                    Xt_1[i] = Xt[i]
                    Yt_1[i] = Yt[i]
                przełącz = 0

        if przełącz == 0:
            for i in range(cp[k-1], n):
                Xt_1[i] = Yt[i]
                Yt_1[i] = Xt[i]
        else:
            for i in range(cp[k-1], n):
                Xt_1[i] = Xt[i]
                Yt_1[i] = Yt[i]

        offspring[idx, :] = Xt_1
        if idx + 1 < offspring_size[0]:
            offspring[idx + 1, :] = Yt_1

    return offspring

def pygadPerformance(parent_selection_type="tournament", crossover_type="single_point", mutation_type="random"):
    level = logging.DEBUG
    name = 'logfile.txt'
    logger = logging.getLogger(name)
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    num_generations = 100
    sol_per_pop = 50
    num_parents_mating = 20
    num_genes = 2

    ga_instance = pygad.GA(num_generations=num_generations,
                           sol_per_pop=sol_per_pop,
                           num_parents_mating=num_parents_mating,
                           num_genes=num_genes,
                           gene_type=float,
                           fitness_func=fitness_function_Rosenbrock,
                           init_range_low=-10,
                           init_range_high=10,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           logger=logger)


    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    if ga_instance.fitness_func==fitness_function_Rosenbrock:
        print(f"Parameters: {parent_selection_type}, {crossover_type}, {mutation_type}")
        print(f"Best solution: {solution}\nBest solution fitness: {1./solution_fitness}")
    else:
        print(f"Parameters: {parent_selection_type}, {crossover_type}, {mutation_type}")
        print(f"Best solution: {solution}\nBest solution fitness: {solution_fitness}")



pygadPerformance("tournament")
pygadPerformance("rws")
pygadPerformance("random")

pygadPerformance("tournament", "single_point")
pygadPerformance("tournament", "two_points")
pygadPerformance("tournament", "uniform")

pygadPerformance("tournament", "single_point", "random")
pygadPerformance("tournament", "single_point", "swap")

pygadPerformance("tournament", position_crossover)
pygadPerformance("tournament", multiple_crossover)
pygadPerformance("tournament", q_rand_point_crossover)