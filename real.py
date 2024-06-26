import random
import time

import numpy as np

import benchmark_functions as bf
import pygad
import logging

from matplotlib import pyplot as plt


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


def average_crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)

    for i in range(0, offspring_size[0], 2):
        parent1_idx = i % parents.shape[0]
        parent2_idx = (i + 1) % parents.shape[0]
        parent1 = parents[parent1_idx].copy()
        parent2 = parents[parent2_idx].copy()

        offspring_1 = (parent1 + parent2) / 2
        offspring_2 = (parent1 + parent2) / 2

        offspring[i, :] = offspring_1
        if i + 1 < offspring_size[0]:
            offspring[i + 1, :] = offspring_2

    return offspring

def arithmetical_crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    num_variables = parents.shape[1]

    for i in range(0, offspring_size[0], 2):
        parent1_idx = i % parents.shape[0]
        parent2_idx = (i + 1) % parents.shape[0]
        parent1 = parents[parent1_idx].copy()
        parent2 = parents[parent2_idx].copy()

        alphas = np.random.uniform(0, 1, num_variables)

        if num_variables == 2:
            alphas[1] = 1 - alphas[0]

        offspring1 = parent1 + alphas * (parent2 - parent1)
        offspring2 = parent2 + alphas * (parent1 - parent2)

        offspring[i, :] = offspring1
        if i + 1 < offspring_size[0]:
            offspring[i + 1, :] = offspring2

    return offspring

def blx_alpha_beta_crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    num_variables = parents.shape[1]

    for i in range(0, offspring_size[0], 2):
        parent1_idx = i % parents.shape[0]
        parent2_idx = (i + 1) % parents.shape[0]
        parent1 = parents[parent1_idx].copy()
        parent2 = parents[parent2_idx].copy()

        offspring1 = np.empty(num_variables)
        offspring2 = np.empty(num_variables)

        for j in range(num_variables):
            parent1_value = parent1[j]
            parent2_value = parent2[j]

            dx = abs(parent1_value - parent2_value)
            min_val = min(parent1_value, parent2_value)
            max_val = max(parent1_value, parent2_value)

            alpha = random.uniform(0, 1)
            beta = random.uniform(0, 1)

            min_range_offspring_1 = min_val - alpha * dx
            max_range_offspring_1 = max_val + beta * dx
            offspring1[j] = random.uniform(min_range_offspring_1, max_range_offspring_1)

            min_range_offspring_2 = min_val - alpha * dx
            max_range_offspring_2 = max_val + beta * dx
            offspring2[j] = random.uniform(min_range_offspring_2, max_range_offspring_2)

        offspring[i, :] = offspring1
        if i + 1 < offspring_size[0]:
            offspring[i + 1, :] = offspring2

    return offspring

def flat_crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    num_variables = parents.shape[1]

    for i in range(0, offspring_size[0], 2):
        parent1_idx = i % parents.shape[0]
        parent2_idx = (i + 1) % parents.shape[0]
        parent1 = parents[parent1_idx].copy()
        parent2 = parents[parent2_idx].copy()

        offspring1 = np.empty(num_variables)
        offspring2 = np.empty(num_variables)

        for j in range(num_variables):
            parent1_value = parent1[j]
            parent2_value = parent2[j]

            offspring1[j] = random.uniform(parent1_value, parent2_value)
            offspring2[j] = random.uniform(parent1_value, parent2_value)

        offspring[i, :] = offspring1
        if i + 1 < offspring_size[0]:
            offspring[i + 1, :] = offspring2

    return offspring

def linear_crossover(parents, offspring_size, ga_instance):
    offspring = np.empty(offspring_size)
    num_variables = parents.shape[1]

    for i in range(0, offspring_size[0], 2):
        parent1_idx = i % parents.shape[0]
        parent2_idx = (i + 1) % parents.shape[0]
        parent1 = parents[parent1_idx].copy()
        parent2 = parents[parent2_idx].copy()

        offspring1 = np.empty(num_variables)
        offspring2 = np.empty(num_variables)

        for j in range(num_variables):
            parent1_value = parent1[j]
            parent2_value = parent2[j]

            z = 0.5 * (parent1_value + parent2_value)
            v = 1.5 * parent1_value - 0.5 * parent2_value
            w = -0.5 * parent1_value + 1.5 * parent2_value

            candidates = [z, v, w]
            candidates.sort()
            offspring1[j], offspring2[j] = candidates[0], candidates[1]

        offspring[i, :] = offspring1
        if i + 1 < offspring_size[0]:
            offspring[i + 1, :] = offspring2

    return offspring


def gauss_mutation(offspring, ga_instance):
    mutation_rate = 0.1

    for chromosome_idx in range(offspring.shape[0]):
        for gene_idx in range(offspring.shape[1]):
            if np.random.rand() < mutation_rate:
                offspring[chromosome_idx, gene_idx] += np.random.normal(0, 1)

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

    average_fitness = []
    std_fitness = []
    best_fitness = []

    def callback_generation(ga_instance):
        fitness = ga_instance.last_generation_fitness
        average_fitness.append(np.mean(fitness))
        std_fitness.append(np.std(fitness))
        best_fitness.append(np.max(fitness))

    ga_instance = pygad.GA(num_generations=num_generations,
                           sol_per_pop=sol_per_pop,
                           num_parents_mating=num_parents_mating,
                           num_genes=num_genes,
                           gene_type=float,
                           fitness_func=fitness_function_Rosenbrock,
                           init_range_low=-10,
                           on_generation=callback_generation,
                           init_range_high=10,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           logger=logger)

    start_time = time.time()
    ga_instance.run()
    end_time = time.time()
    elapsed_time = end_time - start_time

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    if ga_instance.fitness_func==fitness_function_Rosenbrock:
        print(f"Parameters: {parent_selection_type}, {crossover_type}, {mutation_type}")
        print(f"Best solution: {solution}\nBest solution fitness: {1./solution_fitness}")
    else:
        print(f"Parameters: {parent_selection_type}, {crossover_type}, {mutation_type}")
        print(f"Best solution: {solution}\nBest solution fitness: {solution_fitness}")

    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print("")
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(average_fitness, label='Average Fitness')
    plt.title(f"Parameters: {parent_selection_type}, {crossover_type}, {mutation_type}\nAverage Fitness over Generations")
    plt.xlabel('Generation')
    plt.ylabel('Average Fitness')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(std_fitness, label='Standard Deviation of Fitness', color='orange')
    plt.title('Standard Deviation of Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Standard Deviation')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(best_fitness, label='Best Solution Fitness', color='green')
    plt.title('Best Solution Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.legend()

    plt.tight_layout()
    plt.show()


pygadPerformance("tournament")
pygadPerformance("rws")
pygadPerformance("random")

pygadPerformance("tournament", "single_point")
pygadPerformance("tournament", "two_points")
pygadPerformance("tournament", "uniform")

pygadPerformance("tournament", "single_point", "random")
pygadPerformance("tournament", "single_point", "swap")
pygadPerformance("tournament", "single_point", gauss_mutation)

pygadPerformance("tournament", position_crossover)
pygadPerformance("tournament", multiple_crossover)
pygadPerformance("tournament", q_rand_point_crossover)
pygadPerformance("tournament", average_crossover)
pygadPerformance("tournament", arithmetical_crossover)
pygadPerformance("tournament", linear_crossover)
pygadPerformance("tournament", blx_alpha_beta_crossover)
