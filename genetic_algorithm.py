import random
import json
import numpy as np


class GeneticAlgorithm:
    def __init__(self, maze, source, pop_size, mut_rate):
        with open('Utils/config.json', 'r') as file:
            self.params = json.load(file)

        self.maze = maze
        self.src = source
        self.population_size = pop_size
        self.lower_bound = self.params['CHROMOSOME_LOWER']
        self.upper_bound = self.params['CHROMOSOME_UPPER']
        self.mutation_rate = mut_rate
        self.tour_size_percent = self.params['TOURNAMENT_PERCENT']
        self.parent_percent = self.params['PARENT_PERCENT']
        self.tournament_size = None
        self.n_parents = None

    def generate_initial_population(self):
        population = []
        for i in range(self.population_size):
            path_length = np.random.randint(self.lower_bound, self.upper_bound)
            chromosome = []

            previous_move = random.choice(['U', 'D', 'L', 'R'])
            chromosome.append(previous_move)
            for _ in range(1, path_length):
                next_move = random.choice(['U', 'D', 'L', 'R'])
                while (previous_move == 'U' and next_move == 'D') or (previous_move == 'D' and next_move == 'U') or \
                        (previous_move == 'L' and next_move == 'R') or (previous_move == 'R' and next_move == 'L'):
                    next_move = random.choice(['U', 'D', 'L', 'R'])
                chromosome.append(next_move)
                previous_move = next_move
            population.append(''.join(chromosome))
        print(f'Info: Initial Population has been created. \nInfo: Initial Population Size: {self.population_size}')
        print(f'-' * 101)
        return population

    # def select_parents(self, population, fitness_scores):
    #     self.population_size = len(population)
    #     self.tournament_size = int(self.tour_size_percent * self.population_size)
    #     self.n_parents = int(self.parent_percent * self.tournament_size)
    #
    #     print(f'Info: Size of the Tournament - {self.tournament_size}')
    #     parent1, parent2 = [], []
    #     # Select two parents from the population using tournament selection
    #     for i in range(self.n_parents):
    #         parent1.append(self.tournament_selection(population, fitness_scores))
    #         parent2.append(self.tournament_selection(population, fitness_scores))
    #     return parent1, parent2

    def select_parents(self, population, fitness_scores):
        filtered_population, filtered_fitness = zip(*[(individual, fit) for individual, fit in
                                                      zip(population, fitness_scores) if fit >= 0]) \
            if any(fit > 0 for fit in fitness_scores) else ([], [])

        if len(filtered_population) == 0:
            print(f"Warning: All individuals had negative fitness! Regenerating population...")
            exit(0)

        sorted_population = [x for _, x in sorted(zip(filtered_fitness, filtered_population), reverse=True)]
        sorted_fitness = sorted(filtered_fitness, reverse=True)
        print(sorted_fitness)

        self.population_size = len(sorted_population)
        self.tournament_size = int(self.tour_size_percent * self.population_size)
        self.n_parents = int(self.parent_percent * self.tournament_size)

        print(f'Info: Size of the Tournament - {self.tournament_size}, Sorted Population Size: {self.population_size}')
        parent1, parent2 = [], []
        for i in range(self.n_parents):
            parent1.append(self.tournament_selection(sorted_population))
            parent2.append(self.tournament_selection(sorted_population))
        return parent1, parent2

    def tournament_selection(self, population):
        # Randomly select 'tournament_size' individuals (by index) from the population.
        contestant_indices = random.sample(range(self.population_size), self.tournament_size)
        best_index = contestant_indices[0]
        best_parent = population[best_index]
        return best_parent

    # def crossover(self, parent1, parent2):
    #     # Perform crossover between two parents to create a child
    #     split_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
    #     child1 = parent1[:split_point] + parent2[split_point:]
    #     child2 = parent2[:split_point] + parent1[split_point:]
    #     return child1, child2

    def crossover(self, parent1, parent2):
        k1 = self.find_k_point(parent1)
        k2 = self.find_k_point(parent2)

        child = None
        if k1 == 0 and k2 == 0:
            k = random.randint(1, min(len(parent1), len(parent2)) - 1)
            child = parent1[:k] + parent2[k:]
        elif k1 == k2 > 0:
            child = parent1[:k1] + parent2[k2:]
        elif k1 > k2:
            if k2 == 0:
                k = random.randint(1, len(parent2))
                child = parent1[:k1] + parent2[k:]
            else:
                child = parent1[:k1] + parent2[k2:]
        elif k1 < k2:
            if k1 == 0:
                k = random.randint(1, len(parent1))
                child = parent2[:k2] + parent1[k:]
            else:
                child = parent2[:k2] + parent1[k1:]

        return child

    def find_k_point(self, chromosome):
        x, y = self.src
        k = 0
        for gene in chromosome:
            x, y = x + (gene == 'D') - (gene == 'U'), y + (gene == 'R') - (gene == 'L')
            if x < 0 or x >= self.maze.shape[0] or y < 0 or y >= self.maze.shape[1] or self.maze[x][y] == 1:
                return k
            k += 1
        return k

    def mutate(self, chromosome):
        new_chromosome = ''
        for i in range(len(chromosome)-1):
            gene = chromosome[i]
            next_gene = chromosome[i+1]
            if (gene == 'L' and next_gene == 'R') or (gene == 'R' and next_gene == 'L') or \
                (gene == 'U' and next_gene == 'D') or (gene == 'D' and next_gene == 'U'):
                if random.random() < self.mutation_rate:
                    possible_gene = random.choice(['U', 'D', 'L', 'R'])
                # new_chromosome +=
            else:
                new_chromosome += gene
        return chromosome
