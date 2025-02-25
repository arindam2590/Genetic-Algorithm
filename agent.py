import random
import json
import numpy as np
from genetic_algorithm import GeneticAlgorithm


class GeneticAgent:
    def __init__(self, source, goal, maze):
        param_dir = 'Utils/'
        with open(param_dir + 'config.json', 'r') as file:
            self.params = json.load(file)

        self.src = source
        self.dst = goal
        self.maze = maze
        self.position = source
        self.population_size = self.params['POPULATION_SIZE']
        self.mutation_rate = self.params['MUTATION_RATE']
        self.num_generations = self.params['NUM_GENERATIONS']
        self.gen_algo = GeneticAlgorithm(self.population_size, self.mutation_rate)
        self.population = None

    def move(self, path):
        x, y = self.position
        for direction in path:
            if direction == 'U':
                x -= 1
            elif direction == 'D':
                x += 1
            elif direction == 'L':
                y -= 1
            elif direction == 'R':
                y += 1
        self.position = np.array([x, y])

    def run_genetic_algorithm(self):
        best_path = None
        print(f'\n' + '%' * 20 + ' Genetic Algorithm based Path Generation in Maze Environment ' + '%' * 20)
        self.population = self.gen_algo.generate_initial_population()
        print(f'Info: Initial Population has been created')
        print(f'Info: Initial Population Size: {self.population_size}')
        print(f'-'*101)

        for generation in range(self.num_generations):
            print(f"Generation {generation + 1}:")
            fitness_scores = [self.evaluate_fitness(individual) for individual in self.population]
            new_population = self.run_generation(fitness_scores)
            fitness_scores_new_population = [self.evaluate_fitness(individual) for individual in new_population]
            best_fitness_index = np.argmax(fitness_scores_new_population)
            best_path = new_population[best_fitness_index]
            print(f'Best Fitness Score: {fitness_scores_new_population[best_fitness_index]} \n'
                  f'Best Sequence of Direction: {best_path}')
            self.population = new_population
            print(f'-' * 101)
        return best_path
            # self.move(best_path)
        #     if np.array_equal(self.position, self.dst):
        #         print("Goal reached!")
        #         return best_path, fitness_scores
        # print("No solution found within the maximum generations.")

    def run_generation(self, fitness_scores):
        new_population = []
        parent1, parent2 = self.gen_algo.select_parents(self.population, fitness_scores)
        print(f'Info: Size of first Parents - {len(parent1)} and Size of second Parents - {len(parent2)}')
        for p1, p2 in zip(parent1, parent2):
            child1, child2 = self.gen_algo.crossover(p1, p2)
            child1 = self.gen_algo.mutate(child1)
            child2 = self.gen_algo.mutate(child2)
            new_population.append(child1)
            new_population.append(child2)
        print(f'Info: Mutation Rate for the Child - {self.mutation_rate}')
        self.population_size = len(new_population)
        self.gen_algo.tournament_size = int(0.8 * self.population_size)
        self.gen_algo.n_parents = int(0.5 * self.gen_algo.tournament_size)
        print(f'Info: Size of the New Population - {self.population_size}')
        return new_population

    def evaluate_fitness(self, path):
        x, y = self.position
        fitness = 0.0

        for direction in path:
            if direction == 'U':
                x -= 1
            elif direction == 'D':
                x += 1
            elif direction == 'L':
                y -= 1
            elif direction == 'R':
                y += 1
            if x < 0 or x >= self.maze.shape[0] or y < 0 or y >= self.maze.shape[1]:
                break

            if self.maze[x][y] == 1:
                fitness -= 0.5
            else:
                fitness += 1.0

        position = np.array([x, y])
        if np.array_equal(position, self.dst):
            fitness += 5
        return fitness
