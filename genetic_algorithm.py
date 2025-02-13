import random


class GeneticAlgorithm:
    def __init__(self, POPULATION_SIZE, MUTATION_RATE):
        self.population_size = POPULATION_SIZE
        self.mutation_rate = MUTATION_RATE

    def generate_initial_population(self):
        population = []
        for i in range(self.population_size):
            path_length = random.randint(10, 25)  # No. of genes in the chromosome
            path = ''.join(random.choices(['U', 'D', 'L', 'R'], k=path_length))
            population.append(path)
        return population

    def select_parents(self, population, fitness_scores):
        # Select two parents from the population using tournament selection
        parent1 = random.choices(population, weights=fitness_scores, k=5)
        parent2 = random.choices(population, weights=fitness_scores, k=5)
        return parent1, parent2

    def crossover(self, parent1, parent2):
        # Perform crossover between two parents to create a child
        split_point = random.randint(1, min(len(parent1), len(parent2)) - 1)
        child1 = parent1[:split_point] + parent2[split_point:]
        child2 = parent2[:split_point] + parent1[split_point:]
        return child1, child2

    def mutate(self, chromosome):
        new_chromosome = ''
        for gene in chromosome:
            if random.random() < self.mutation_rate:
                new_chromosome += random.choice(['U', 'D', 'L', 'R'])
            else:
                new_chromosome += gene
        return new_chromosome
