import random


import benchmark_functions as bf
import numpy as np
import pygad
import logging

from matplotlib import pyplot as plt


def decodeInd(individual):
    a=-10
    b=10
    precision=10
    num_variables = len(individual) // precision
    real_values = []

    for i in range(num_variables):
        start_index = i * precision
        end_index = start_index + precision
        binary_str = ''.join(str(bit) for bit in individual[start_index:end_index])
        decimal_value = int(binary_str, 2)
        real_value = a + ((b - a) / (2 ** precision - 1)) * decimal_value

        real_values.append(real_value)

    return real_values
def fitness_function_Styblinski(ga_instance, solution, solution_idx):
    ind = decodeInd(solution)
    return -bf.StyblinskiTang(len(ind))(ind)
def fitness_function_Rosenbrock(ga_instance, solution, solution_idx):
    ind = decodeInd(solution)
    return 1./bf.Rosenbrock(len(ind))(ind)

def granular_crossover(parents, offspring_size, ga_instance):
    offspring = []
    idx = 0
    while len(offspring) != offspring_size[0]:
        parent1 = parents[idx % parents.shape[0], :].copy()
        parent2 = parents[(idx + 1) % parents.shape[0], :].copy()
        for j in range(len(parent1)):
            if np.random.random() < 0.5:
                parent1[j], parent2[j] = parent1[j], parent2[j]
            else:
                parent1[j], parent2[j] = parent2[j], parent2[j]
        offspring.append(parent1)
        if len(offspring) != offspring_size[0]:
            offspring.append(parent2)
        idx += 1

    return np.array(offspring)


def restricted_crossover(parents, offspring_size, ga_instance):
    offspring = []
    num_offspring = offspring_size[0] // 2
    for _ in range(num_offspring):
        parent_a = parents[0]
        parent_b = parents[1]
        l = np.random.randint(0, len(parent_a) - 4)
        rnd = np.random.random()

        child_c = np.concatenate((parent_a[:l + 1], parent_b[l + 1:len(parent_b) - 2]))
        child_d = np.concatenate((parent_b[:l + 1], parent_a[l + 1:len(parent_b) - 2]))

        if rnd <= 0.5:
            child_c = np.concatenate((child_c, [parent_a[len(parent_a) - 2], parent_b[len(parent_b) - 1]]))
        else:
            child_c = np.concatenate((child_c, [parent_b[len(parent_b) - 2], parent_a[len(parent_a) - 1]]))

        if rnd <= 0.5:
            child_d = np.concatenate((child_d, [parent_a[len(parent_a) - 2], parent_b[len(parent_b) - 1]]))
        else:
            child_d = np.concatenate((child_d, [parent_b[len(parent_b) - 2], parent_a[len(parent_a) - 1]]))

        offspring.append(child_c)
        offspring.append(child_d)
    if len(offspring) < offspring_size[0]:
        parent_a = parents[0]
        parent_b = parents[1]
        l = np.random.randint(0, len(parent_a) - 4)
        rnd = np.random.random()

        child_e = np.concatenate((parent_a[:l + 1], parent_b[l + 1:len(parent_b) - 2]))
        child_f = np.concatenate((parent_b[:l + 1], parent_a[l + 1:len(parent_b) - 2]))

        if rnd <= 0.5:
            child_e = np.concatenate((child_e, [parent_a[len(parent_a) - 2], parent_b[len(parent_b) - 1]]))
        else:
            child_e = np.concatenate((child_e, [parent_b[len(parent_b) - 2], parent_a[len(parent_a) - 1]]))

        if rnd <= 0.5:
            child_f = np.concatenate((child_f, [parent_a[len(parent_a) - 2], parent_b[len(parent_b) - 1]]))
        else:
            child_f = np.concatenate((child_f, [parent_b[len(parent_b) - 2], parent_a[len(parent_a) - 1]]))

        offspring.append(child_e)

    return np.array(offspring)

def homologous_crossover(parents, offspring_size, ga_instance):
    m = 3
    w = 2
    u = 0.4
    crossover_points = sorted(np.random.choice(range(0, len(parents[0]) - 1), m, replace=False))
    offspring = []
    idx = 0
    while len(offspring) < offspring_size[0]:
        parent_a = parents[idx % parents.shape[0]].copy()
        parent_b = parents[(idx + 1) % parents.shape[0]].copy()
        offspring_a = []
        offspring_b = []
        offspring_a.extend(parent_a[:crossover_points[0]])
        offspring_b.extend(parent_b[:crossover_points[0]])
        for i in range(len(crossover_points) - 1):
            start = crossover_points[i]
            end = crossover_points[i + 1]
            substring_a = parent_a[start+1:end]
            substring_b = parent_b[start+1:end]
            if len(substring_a) >= w:
                number_of_1 = sum(1 for a, b in zip(substring_a[:w], substring_b[:w]) if a != b)
                DS = number_of_1 / len(substring_a)
                if DS >= u:
                    substring_a, substring_b = substring_b, substring_a
            offspring_a.extend([parent_a[start]])
            offspring_a.extend(substring_a)
            offspring_b.extend([parent_b[start]])
            offspring_b.extend(substring_b)
        offspring_a.extend(parent_a[end:])
        offspring_b.extend(parent_b[end:])
        offspring.append(offspring_a)
        offspring.append(offspring_b)
        idx += 2
    return np.array(offspring[:offspring_size[0]])




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
    num_genes = 20
    gene_space = [0, 1]
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
                           gene_type=int,
                           fitness_func=fitness_function_Rosenbrock,
                           gene_space=gene_space,
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           on_generation=callback_generation,
                           logger=logger)


    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    if ga_instance.fitness_func==fitness_function_Rosenbrock:
        print(f"Parameters: {parent_selection_type}, {crossover_type}, {mutation_type}")
        print(f"Best solution: {decodeInd(solution)}\nBest solution fitness: {1./solution_fitness}")
    else:
        print(f"Parameters: {parent_selection_type}, {crossover_type}, {mutation_type}")
        print(f"Best solution: {decodeInd(solution)}\nBest solution fitness: {solution_fitness}")
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(average_fitness, label='Average Fitness')
    plt.title('Average Fitness over Generations')
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
pygadPerformance("tournament", granular_crossover)
pygadPerformance("tournament", restricted_crossover)
pygadPerformance("tournament", homologous_crossover)