import numpy as np
from agent import GeneticAgent


def main():
    source = np.array([0, 0])
    destination = np.array([3, 3])

    maze = np.array([[0, 0, 0, 0],
                     [0, 1, 0, 1],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0]])

    # maze = np.array([[0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
    #                  [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #                  [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    #                  [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    #                  [0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    #                  [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],
    #                  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    #                  [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    #                  [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    #                  [0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    #                  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    #                  [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    #                  [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    #                  [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    #                  [1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0]])

    ga_agent = GeneticAgent(source, destination, maze)
    best_path = ga_agent.run_genetic_algorithm()

    if best_path:
        print(f"\nOptimal Path Found: {best_path}")
        ga_agent.display_maze()
    else:
        print("\nNo optimal path found within the given generations.")




if __name__ == '__main__':
    main()