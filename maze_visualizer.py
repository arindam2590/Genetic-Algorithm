import json
import pygame
import numpy as np

class visualize_maze:
    def __init__(self, maze, source, goal):
        with open('Utils/config.json', 'r') as file:
            self.params = json.load(file)

        self.source = source
        self.destination = goal
        self.maze = maze
        self.position = self.source

        self.maze_size = self.params['SIZE']
        self.cell_size = self.params['CELL_SIZE']
        self.window_size = (self.cell_size * self.maze_size, self.cell_size * self.maze_size)
        self.screen = None
        self.clock = None
        self.fps = self.params['FPS']
        self.WHITE = (255, 255, 255)
        self.path_coord = []

    def find_path(self, directions):
        x, y = self.position
        self.path_coord.append(self.position)
        for direction in directions:
            if direction == 'U':
                x -= 1
            elif direction == 'D':
                x += 1
            elif direction == 'L':
                y -= 1
            elif direction == 'R':
                y += 1
            self.position = np.array([x, y])
            self.path_coord.append(self.position)

    def update_display(self):
        self.screen.fill(self.WHITE)

        for y in range(self.maze_size):
            for x in range(self.maze_size):
                color = (0, 0, 0) if self.maze[y, x] == 1 else (255, 255, 255)
                pygame.draw.rect(self.screen, color,
                                 pygame.Rect(x * self.cell_size, y * self.cell_size,
                                             self.cell_size, self.cell_size))

        if self.path_coord:
            for i in range(len(self.path_coord) - 1):
                start_pos = (self.path_coord[i][1] * self.cell_size + self.cell_size // 2,
                             self.path_coord[i][0] * self.cell_size + self.cell_size // 2)

                end_pos = (self.path_coord[i + 1][1] * self.cell_size + self.cell_size // 2,
                           self.path_coord[i + 1][0] * self.cell_size + self.cell_size // 2)

                pygame.draw.line(self.screen, (0, 0, 255), start_pos, end_pos, 3)

        pygame.draw.circle(self.screen, (0, 255, 0),
                           (self.source[1] * self.cell_size + self.cell_size // 2,
                            self.source[0] * self.cell_size + self.cell_size // 2), 8)

        pygame.draw.circle(self.screen, (255, 0, 255),
                           (self.destination[1] * self.cell_size + self.cell_size // 2,
                            self.destination[0] * self.cell_size + self.cell_size // 2), 8)

        pygame.display.update()
        self.clock.tick(self.fps)

    def env_setup(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption('Maze Simulation')
        self.clock = pygame.time.Clock()
















# import matplotlib.pyplot as plt
# import numpy as np
#
# def visualize_maze(maze, ant=None, save_path=None):
#     # Visualize the maze with optional ant path
#     cmap = plt.cm.binary
#     cmap.set_bad(color='red')
#     plt.imshow(maze, cmap=cmap, interpolation='nearest')
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     if ant is not None:
#         x, y = ant.move(maze)
#         plt.scatter(x, y, c='green', marker='o')
#         plt.plot([0], [0], marker='o', markersize=10, color='blue')
#         path_x = [0]
#         path_y = [0]
#         for direction in ant.path:
#             if direction == 'U':
#                 path_y.append(path_y[-1]-1)
#                 path_x.append(path_x[-1])
#             elif direction == 'D':
#                 path_y.append(path_y[-1]+1)
#                 path_x.append(path_x[-1])
#             elif direction == 'L':
#                 path_y.append(path_y[-1])
#                 path_x.append(path_x[-1]-1)
#             elif direction == 'R':
#                 path_y.append(path_y[-1])
#                 path_x.append(path_x[-1]+1)
#         plt.plot(path_x, path_y, c='green', linewidth=2)
#     if save_path is not None:
#         plt.savefig(save_path)
#     plt.show()
#
# def visualize_fitness(fitness_history):
#     # Visualize the fitness history of the genetic algorithm
#     plt.plot(np.arange(len(fitness_history)), fitness_history)
#     plt.xlabel('Generation')
#     plt.ylabel('Fitness')
#     plt.show()
