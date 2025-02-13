import numpy as np
from agent import GeneticAgent

# from maze_visualizer import visualize_maze, visualize_fitness

source = np.array([0, 0])
destination = np.array([3, 3])

maze = np.array([[0, 0, 0, 0],
                 [0, 1, 0, 1],
                 [0, 0, 0, 1],
                 [1, 0, 0, 0]])

# maze = np.array([[0,0,0,0,0,1,0,0],
#                  [0,1,0,1,0,1,0,1],
#                  [0,1,0,1,0,0,0,0],
#                  [0,1,0,1,1,1,1,0],
#                  [0,0,0,0,0,1,0,0],
#                  [1,1,0,1,0,0,0,1],
#                  [0,1,0,1,0,1,1,0],
#                  [0,0,0,0,0,0,0,0]])

ga_agent = GeneticAgent(source, destination, maze)
best_ant, fitness_history = ga_agent.run_genetic_algorithm()

'''# Visualize the best ant's path through the maze
print(f"Best ant fitness: {best_ant.fitness}")
visualize_maze(maze, ant=best_ant)

# Visualize the fitness history of the genetic algorithm
visualize_fitness(fitness_history)'''
