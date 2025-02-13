import random
import numpy as np
from genetic_algorithm import GeneticAlgorithm


POPULATION_SIZE = 50
MUTATION_RATE = 0.1
NUM_GENERATIONS = 100


class GeneticAgent:
    def __init__(self, source, goal, maze):
        self.src = source
        self.dst = goal
        self.position = source
        self.maze = maze
        self.gen_algo = GeneticAlgorithm(POPULATION_SIZE, MUTATION_RATE)
        self.num_generations = NUM_GENERATIONS
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
        self.population = self.gen_algo.generate_initial_population()
        for generation in range(self.num_generations):
            print(f"Generation {generation + 1}")
            new_population = self.run_generation()
            fitness_scores = [self.evaluate_fitness(individual) for individual in new_population]
            best_fitness_index = np.argmax(fitness_scores)
            best_path = new_population[best_fitness_index]
            self.move(best_path)
            if np.array_equal(self.position, self.dst):
                print("Goal reached!")
                return best_path, fitness_scores
        print("No solution found within the maximum generations.")
        return None, None

    def run_generation(self):
        new_population = []
        fitness_scores = [self.evaluate_fitness(individual) for individual in self.population]
        for i in range(len(self.population)):
            parent1, parent2 = self.gen_algo.select_parents(self.population, fitness_scores)
            child1, child2 = self.gen_algo.crossover(parent1, parent2)
            if random.random() < MUTATION_RATE:
                child1 = self.gen_algo.mutate(child1)
                child2 = self.gen_algo.mutate(child2)
            new_population.extend([child1, child2])
        return new_population

    def evaluate_fitness(self, path):
        x, y = self.position
        fitness = 0

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
                break
            fitness += 1

        position = np.array([x, y])
        if np.array_equal(position, self.dst):
            fitness += 5
        return fitness
