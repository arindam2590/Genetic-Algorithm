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
        self.weighted_feas = 100
        self.weighted_len = 10
        self.weighted_turn = 10

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
            fitness = [self.evaluate_fitness(individual) for individual in self.population]
            fitness_scores = self.normalized_fitness(fitness, True)
            new_population = self.run_generation(fitness_scores)
            fitness = [self.evaluate_fitness(individual) for individual in new_population]
            fitness_scores_new_population = self.normalized_fitness(fitness)
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
        individual_len, turn_count, feasibility_score = 1, 0, 0
        for i in range(1, len(path)):
            individual_len += 1
            if path[i] != path[i - 1]:
                turn_count += 1

        print(turn_count)
        feasibility = True
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
                feasibility = False
                break

            if self.maze[x][y] == 1:
                feasibility = False
                break

        position = np.array([x, y])
        if np.array_equal(position, self.dst):
            feasibility = True

        if feasibility:
            feasibility_score = 5

        return [turn_count, feasibility_score, individual_len]

    def normalized_fitness(self, fitness, flag=False):
        fitness_array = np.array(fitness, dtype=float)
        feasibility_values = fitness_array[:, 0]
        turn_count_values = fitness_array[:, 1]
        individual_len_values = fitness_array[:, 2]

        min_feasibility, max_feasibility = min(feasibility_values), max(feasibility_values)
        min_turns, max_turns = min(turn_count_values), max(turn_count_values)
        min_length, max_length = min(individual_len_values), max(individual_len_values)
        if flag:
            print(f'Info: Min. Feasibility: {min_feasibility: .3f}, Max. Feasibility: {max_feasibility: .3f}, '
                  f'Min. Turns: {min_turns: .3f}, Max. Turns: {max_turns: .3f}, Min. Length: {min_length: .3f}, '
                  f'Max. Length: {max_length: .3f}')

        normalized_fitness = [[1 - ((feasibility - min_feasibility) / (max_feasibility - min_feasibility))
                               if max_feasibility != min_feasibility else 1.0,
                               1 - ((turn - min_turns) / (max_turns - min_turns)) if max_turns != min_turns else 1.0,
                               1 - ((length - min_length) / (max_length - min_length))
                               if max_length != min_length else 1.0]
                              for feasibility, turn, length in fitness]

        final_fitness_values = [(self.weighted_feas * normalized_value[0]) *
                                (((self.weighted_turn * normalized_value[1]) + (self.weighted_len * normalized_value[1]))
                                /(self.weighted_turn + self.weighted_len))
                                for normalized_value in normalized_fitness]

        return final_fitness_values
