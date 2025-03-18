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
        self.gen_algo = GeneticAlgorithm(self.maze, self.src, self.population_size, self.mutation_rate)
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

            fitness_scores = [self.evaluate_fitness(individual) for individual in self.population]
            # print(fitness)
            # filtered_fitness = self.remove_infeasible_individuals(fitness)
            # filtered_population, filtered_fitness = zip(*[(individual, fit) for individual, fit in
            #                                               zip(self.population, fitness) if fit[1]]) \
            #     if any(fit[1] for fit in fitness) else ([], [])
            # self.population = filtered_population
            # self.population_size = len(filtered_population)
            # filtered_fitness = list(filtered_fitness)

            # norm_fitness_scores = self.normalized_fitness(fitness, True)

            best_fitness_index = np.argmax(fitness_scores)
            best_value_generation = fitness_scores[best_fitness_index]
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

            new_population = self.run_generation(fitness_scores)
            self.population = new_population
            print(f'-' * 101)

        self.data_visual.plot_fitness(self.best_fitness_list)
        return self.best_solution

    def run_generation(self, fitness_scores):
        new_population = []
        parent1, parent2 = self.gen_algo.select_parents(self.population, fitness_scores)
        print(f'Info: Size of first Parents - {len(parent1)} and Size of second Parents - {len(parent2)}')

        for p1, p2 in zip(parent1, parent2):
            child = self.gen_algo.crossover(p1, p2)
            new_population.append(self.gen_algo.mutate(child))
            # new_population.append(child)
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

    # def evaluate_fitness(self, individual):
    #     x, y = self.src
    #     individual_len, turn_count, feasibility_score, feasible_flag = 1, 0, 0, True
    #
    #     for i in range(len(individual) - 1):
    #         if individual[i] != individual[i + 1]:
    #             turn_count += 1
    #
    #     for gene in individual:
    #         individual_len += 1
    #         x, y = x + (gene == 'D') - (gene == 'U'), y + (gene == 'R') - (gene == 'L')
    #
    #         if x < 0 or x >= self.maze.shape[0] or y < 0 or y >= self.maze.shape[1] or self.maze[x][y] == 1:
    #             feasible_flag = False
    #             return [-5, feasible_flag, turn_count, individual_len]
    #
    #         feasibility_score += 1
    #
    #     return [feasibility_score + 5 if (x, y) == tuple(self.dst) else feasibility_score,
    #             feasible_flag, turn_count, individual_len]

    def evaluate_fitness(self, individual):
        x, y = self.src
        fitness = penalty = dist_fitness = 0.0
        for gene in individual:
            x, y = x + (gene == 'D') - (gene == 'U'), y + (gene == 'R') - (gene == 'L')
            if x < 0 or x >= self.maze.shape[0] or y < 0 or y >= self.maze.shape[1] or self.maze[x][y] == 1:
                penalty -= 1.0
            else:
                if (x, y) == tuple(self.dst):
                    fitness += 5.0
                else:
                    fitness += 1.0
            dist_fitness = self.euclidean_distance((x, y))
        fitness = fitness - (penalty + dist_fitness)
        return fitness

    def euclidean_distance(self, point1):
        x1, y1 = point1
        x2, y2 = self.dst
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    # def normalized_fitness(self, fitness, flag=False):
    #     fitness_array = np.array(fitness, dtype=float)
    #     print(fitness_array)
    #     min_vals, max_vals = fitness_array.min(axis=0), fitness_array.max(axis=0)
    #
    #     # if flag:
    #     #     print(f'Info: Min. Feasibility: {min_vals[0]: .3f}, Max. Feasibility: {max_vals[0]: .3f}, '
    #     #           f'Min. Turns: {min_vals[1]: .6f}, Max. Turns: {max_vals[1]: .6f}, Min. Length: {min_vals[2]: .3f},'
    #     #           f'Max. Length: {max_vals[2]: .3f}')
    #
    #     # norm_fitness = [(f[0] - min_vals[0]) / (max_vals[0] - min_vals[0]) if max_vals[0] != min_vals[0] else 1.0,
    #     #                 for f in fitness]
    #
    #     # weighted_norm_fitness = [(self.weighted_feas * score[0]) + (self.weighted_turn * score[1]) +
    #     #                          (self.weighted_len * score[2]) for score in norm_fitness]
    #
    #     return 0 #weighted_norm_fitness

    def display_maze(self):
        self.running = True
        self.visual_maze.env_setup()
        self.visual_maze.find_path(self.best_solution)
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.image.save(self.visual_maze.screen, "Screenshot/maze_window.png")
                    self.running = False

            self.visual_maze.update_display()

        pygame.quit()

    def remove_infeasible_individuals(self, fitness_scores):
        filtered_population = []
        print(fitness_scores)
        for individual, fit_score in zip(self.population, fitness_scores):
            if fit_score[1]:
                # fit_score
                filtered_population.append(individual)
                # filtered_fitness
