import numpy as np
from agent import GeneticAgent

# from maze_visualizer import visualize_maze, visualize_fitness

source = np.array([0, 0])
destination = np.array([14, 14])

maze = np.array([
    [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]
])

ga_agent = GeneticAgent(source, destination, maze)
best_path = ga_agent.run_genetic_algorithm()

'''# Visualize the best ant's path through the maze
print(f"Best ant fitness: {best_ant.fitness}")
visualize_maze(maze, ant=best_ant)

# Visualize the fitness history of the genetic algorithm3
visualize_fitness(fitness_history)'''
