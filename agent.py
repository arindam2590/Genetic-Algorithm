import random
import json
import numpy as np
import pygame
from genetic_algorithm import GeneticAlgorithm
from maze_visualizer import visualize_maze
from Utils.utils import DataVisualization


class GeneticAgent:
    def __init__(self, source, goal, maze):
        with open('Utils/config.json', 'r') as file:
            self.params = json.load(file)

        self.src = source
        self.dst = goal
        self.maze = maze
        self.population = None
        self.population_size = self.params['POPULATION_SIZE']
        self.mutation_rate = self.params['MUTATION_RATE']
        self.num_generations = self.params['NUM_GENERATIONS']
        self.gen_algo = GeneticAlgorithm(self.population_size, self.mutation_rate)
        self.best_fitness = -float('inf')  # Track best fitness across generations
        self.best_fitness_list = []
        self.best_solution_list = []
        self.best_solution = None
        self.elitism = 2  # Keep top 2 best solutions unchanged
        self.visual_maze = visualize_maze(self.maze, self.src, self.dst)
        self.weighted_feas = 10
        self.weighted_len = 1
        self.weighted_turn = 1
        self.running = False
        self.data_visual = DataVisualization(self.num_generations)

    def run_genetic_algorithm(self):
        print(f'\n' + '%' * 20 + ' Genetic Algorithm based Path Generation in Maze Environment ' + '%' * 20)
        self.population = self.gen_algo.generate_initial_population()
        print(f'Info: Initial Population has been created. \nInfo: Initial Population Size: {self.population_size}')
        print(f'-' * 101)

        for generation in range(self.num_generations):
            print(f"Generation {generation + 1}:")

            fitness = [self.evaluate_fitness(individual) for individual in self.population]
            # filtered_population, filtered_fitness = zip(*[(individual, fit) for individual, fit in
            #                                               zip(self.population, fitness) if fit[1]]) \
            #     if any(fit[1] for fit in fitness) else ([], [])
            # self.population = filtered_population
            # self.population_size = len(filtered_population)
            # filtered_fitness = list(filtered_fitness)

            norm_fitness_scores = self.normalized_fitness(fitness, True)

            best_fitness_index = np.argmax(norm_fitness_scores)
            best_value_generation = norm_fitness_scores[best_fitness_index]
            best_solution_generation = self.population[best_fitness_index]
            self.best_fitness_list.append(best_value_generation)
            self.best_solution_list.append(best_solution_generation)

            print(f'Info: Best Fitness of the Generation: {best_value_generation:.3f} | '
                  f'Previous Best Fitness : {self.best_fitness:.3f} | Best Path of the Generation: {best_solution_generation} | '
                  f'Previous Best Solution: {self.best_solution}')
            if best_value_generation > self.best_fitness:
                print(f'Info: Best solution is improved in this Generation.')
                self.best_fitness = best_value_generation
                self.best_solution = best_solution_generation
            else:
                print(f'Info: Best solution is not improved in this Generation')

            new_population = self.run_generation(norm_fitness_scores)
            self.population = new_population
            print(f'-' * 101)

        self.data_visual.plot_fitness(self.best_fitness_list)
        return self.best_solution

    def run_generation(self, fitness_scores):
        new_population = []
        parent1, parent2 = self.gen_algo.select_parents(self.population, fitness_scores)
        print(f'Info: Size of first Parents - {len(parent1)} and Size of second Parents - {len(parent2)}')

        for p1, p2 in zip(parent1, parent2):
            child1, child2 = self.gen_algo.crossover(p1, p2)
            new_population.append(self.gen_algo.mutate(child1))
            new_population.append(self.gen_algo.mutate(child2))
        print(f'Info: Mutation Rate for the Child - {self.mutation_rate}')
        elite_individuals = [best_ind[0] for best_ind in
                             sorted(zip(self.population, fitness_scores), key=lambda x: x[1],
                                    reverse=True)[:self.elitism]]
        print(f'Info: Elite individuals of the Generation - {elite_individuals}')
        for individual in elite_individuals:
            new_population.append(individual)
        self.population_size = len(new_population)
        print(f'Info: Size of the New Population - {self.population_size}')
        return new_population

    def evaluate_fitness(self, individual):
        x, y = self.src
        individual_len, turn_count, feasibility_score, feasible_flag = 1, 0, 0, True

        for i in range(len(individual) - 1):
            if individual[i] != individual[i + 1]:
                turn_count += 1

        for gene in individual:
            individual_len += 1
            x, y = x + (gene == 'D') - (gene == 'U'), y + (gene == 'R') - (gene == 'L')

            if x < 0 or x >= self.maze.shape[0] or y < 0 or y >= self.maze.shape[1] or self.maze[x][y] == 1:
                feasible_flag = False
                return [-5, feasible_flag, turn_count, individual_len]

            feasibility_score += 1

        return [feasibility_score + 5 if (x, y) == tuple(self.dst) else feasibility_score,
                feasible_flag, turn_count, individual_len]

    def normalized_fitness(self, fitness, flag=False):
        fitness_array = np.array(fitness, dtype=float)
        min_vals, max_vals = fitness_array.min(axis=0), fitness_array.max(axis=0)

        if flag:
            print(f'Info: Min. Feasibility: {min_vals[0]: .3f}, Max. Feasibility: {max_vals[0]: .3f}, '
                  f'Min. Turns: {min_vals[1]: .6f}, Max. Turns: {max_vals[1]: .6f}, Min. Length: {min_vals[2]: .3f},'
                  f'Max. Length: {max_vals[2]: .3f}')

        norm_fitness = [((f[0] - min_vals[0]) / (max_vals[0] - min_vals[0]) if max_vals[0] != min_vals[0] else 1.0,
                         1 - (f[1] - min_vals[1]) / (max_vals[1] - min_vals[1]) if max_vals[1] != min_vals[1] else 1.0,
                         1 - (f[2] - min_vals[2]) / (max_vals[2] - min_vals[2]) if max_vals[2] != min_vals[2] else 1.0)
                        for f in fitness]

        weighted_norm_fitness = [(self.weighted_feas * score[0]) + (self.weighted_turn * score[1]) +
                                 (self.weighted_len * score[2]) for score in norm_fitness]

        return weighted_norm_fitness

    def display_maze(self):
        self.running = True
        self.visual_maze.env_setup()
        self.visual_maze.find_path(self.best_solution)
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.visual_maze.update_display()

        pygame.quit()

# class GeneticAgent:
#     def __init__(self, source, goal, maze):
#         param_dir = 'Utils/'
#         with open(param_dir + 'config.json', 'r') as file:
#             self.params = json.load(file)
#
#         self.src = source
#         self.dst = goal
#         self.maze = maze
#         self.position = source
#         self.population_size = self.params['POPULATION_SIZE']
#         self.mutation_rate = self.params['MUTATION_RATE']
#         self.num_generations = self.params['NUM_GENERATIONS']
#         self.gen_algo = GeneticAlgorithm(self.population_size, self.mutation_rate)
#         self.population = None
#         self.weighted_feas = 100
#         self.weighted_len = 10
#         self.weighted_turn = 10

#     def run_genetic_algorithm(self):
#         best_path = None
#         print(f'\n' + '%' * 20 + ' Genetic Algorithm based Path Generation in Maze Environment ' + '%' * 20)
#         self.population = self.gen_algo.generate_initial_population()
#
#
#         for generation in range(self.num_generations):
#             print(f"Generation {generation + 1}:")
#             fitness = [self.evaluate_fitness(individual) for individual in self.population]
#             fitness_scores = self.normalized_fitness(fitness, True)
#             print(fitness_scores)
#             new_population = self.run_generation(fitness_scores)
#             fitness = [self.evaluate_fitness(individual) for individual in new_population]
#             fitness_scores_new_population = self.normalized_fitness(fitness)
#             best_fitness_index = np.argmax(fitness_scores_new_population)
#             best_path = new_population[best_fitness_index]
#             print(f'Best Fitness Score: {fitness_scores_new_population[best_fitness_index]} \n'
#                   f'Best Sequence of Direction: {best_path}')
#             self.population = new_population
#             print(f'-' * 101)
#         return best_path
#         # self.move(best_path)
#         #     if np.array_equal(self.position, self.dst):
#         #         print("Goal reached!")
#         #         return best_path, fitness_scores
#         # print("No solution found within the maximum generations.")
#
#     def run_generation(self, fitness_scores):
#         new_population = []
#         parent1, parent2 = self.gen_algo.select_parents(self.population, fitness_scores)
#         print(f'Info: Size of first Parents - {len(parent1)} and Size of second Parents - {len(parent2)}')
#         for p1, p2 in zip(parent1, parent2):
#             child1, child2 = self.gen_algo.crossover(p1, p2)
#             child1 = self.gen_algo.mutate(child1)
#             child2 = self.gen_algo.mutate(child2)
#             new_population.append(child1)
#             new_population.append(child2)
#         print(f'Info: Mutation Rate for the Child - {self.mutation_rate}')
#         self.population_size = len(new_population)
#         self.gen_algo.tournament_size = int(0.8 * self.population_size)
#         self.gen_algo.n_parents = int(0.5 * self.gen_algo.tournament_size)
#         print(f'Info: Size of the New Population - {self.population_size}')
#         return new_population
#
#     def evaluate_fitness(self, path):
#         x, y = self.position
#         individual_len, turn_count, feasibility_score = 0, 0, 0
#         for i in range(len(path)-1):
#             if path[i] != path[i + 1]:
#                 turn_count += 1
#
#         # print(turn_count)
#         feasibility = True
#         for direction in path:
#             individual_len += 1
#             if direction == 'U':
#                 x -= 1
#             elif direction == 'D':
#                 x += 1
#             elif direction == 'L':
#                 y -= 1
#             elif direction == 'R':
#                 y += 1
#             if x < 0 or x >= self.maze.shape[0] or y < 0 or y >= self.maze.shape[1]:
#                 feasibility = False
#                 feasibility_score -= 5
#                 break
#
#             if self.maze[x][y] == 1:
#                 feasibility = False
#                 feasibility_score -= 5
#                 break
#             else:
#                 feasibility_score += 1
#
#         position = np.array([x, y])
#         if np.array_equal(position, self.dst):
#             feasibility = True
#
#         if feasibility:
#             feasibility_score += 5
#
#         return [feasibility_score, turn_count, individual_len]
#
#     def normalized_fitness(self, fitness, flag=False):
#         print(fitness)
#         fitness_array = np.array(fitness, dtype=float)
#         feasibility_values = fitness_array[:, 0]
#         turn_count_values = fitness_array[:, 1]
#         individual_len_values = fitness_array[:, 2]
#
#         min_feasibility, max_feasibility = min(feasibility_values), max(feasibility_values)
#         min_turns, max_turns = min(turn_count_values), max(turn_count_values)
#         min_length, max_length = min(individual_len_values), max(individual_len_values)
#         if flag:
#             print(turn_count_values, feasibility_values, individual_len_values)
#             print(f'Info: Min. Feasibility: {min_feasibility: .3f}, Max. Feasibility: {max_feasibility: .3f}, '
#                   f'Min. Turns: {min_turns: .6f}, Max. Turns: {max_turns: .6f}, Min. Length: {min_length: .3f}, '
#                   f'Max. Length: {max_length: .3f}')
#
#         normalized_fitness = [[1 - ((feasibility - min_feasibility) / (max_feasibility - min_feasibility))
#                                if max_feasibility != min_feasibility else 1.0,
#                                1 - ((turn - min_turns) / (max_turns - min_turns)) if max_turns != min_turns else 1.0,
#                                1 - ((length - min_length) / (max_length - min_length))
#                                if max_length != min_length else 1.0]
#                               for feasibility, turn, length in fitness]
#
#         final_fitness_values = [(self.weighted_feas * normalized_value[0]) *
#                                 (((self.weighted_turn * normalized_value[1]) + (self.weighted_len * normalized_value[1]))
#                                 /(self.weighted_turn + self.weighted_len))
#                                 for normalized_value in normalized_fitness]
#
#         return final_fitness_values
