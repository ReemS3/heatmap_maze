import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class MazeEnv:
    """A Maze Environment class to create a maze with rewards and punishments."""
    
    def __init__(self, rows=10, cols=10):
        """Initialize the MazeEnv with default or given dimensions."""
        self.maze = np.random.choice([0], size=(rows, cols))
        self.dim = (rows, cols)
        self.start_coord = 0
        self.goal_coord = 0
        self.mine_coord = 0

    def set_coord_start_goal_mine(self, start_coord, goal_coord, mine_coord):
        """Set the start, goal, and mine coordinates."""
        self.start_coord = start_coord
        self.goal_coord = goal_coord
        self.mine_coord = mine_coord

    def set_density_magnitude_goal_mine(
        self,
        reward_val=1,
        punishment_val=0.4,
        reward_density=1,
        punishment_density=2,
    ):
        """Set the density and magnitude values for the goal and mine."""
        self.reward_val = reward_val
        self.punishment_val = punishment_val
        self.reward_density = reward_density
        self.punishment_density = punishment_density

    def reset_maze(self, rows=10, cols=10):
        """Reset the maze with default or given dimensions."""
        self.maze = np.random.choice([0], size=(rows, cols))

    def __gaussian(self, x, sigma, mu=0):
        """Private Gaussian function for calculating rewards."""
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def __gaussian_rewards(self):
        """Private function to calculate Gaussian rewards."""
        x, y = np.indices(self.maze.shape)
        coords = np.column_stack((x.flat, y.flat))
        goal_coords = np.array(self.goal_coord).reshape(1, -1)
        mine_coords = np.array(self.mine_coord).reshape(1, -1)
        euclidean_distances_goal = cdist(
            coords, goal_coords, "euclidean"
        ).reshape(self.maze.shape)
        euclidean_distances_mine = cdist(
            coords, mine_coords, "euclidean"
        ).reshape(self.maze.shape)
        gaussian_distances_goal = self.__gaussian(
            euclidean_distances_goal, sigma=self.reward_density
        )
        gaussian_distances_mine = self.__gaussian(
            euclidean_distances_mine, sigma=self.punishment_density
        )
        rewards = (
            self.reward_val * gaussian_distances_goal
            - self.punishment_val * gaussian_distances_mine
        )
        rewards[self.goal_coord] = self.reward_val * 1
        rewards[self.mine_coord] = self.punishment_val * -1
        return rewards

    def calculate_rewards(self):
        """Calculate the Gaussian rewards."""
        return self.__gaussian_rewards()

    def plot_maze(self):
        """Plot the maze with rewards and punishments."""
        rewards = self.calculate_rewards()
        rows, cols = self.maze.shape[0], self.maze.shape[1]
        __, ax = plt.subplots()
        # Reverse the colours of coolwarm
        cmap = plt.cm.coolwarm.reversed()

        im = ax.imshow(
            rewards, cmap=cmap, origin="lower", norm=plt.Normalize(-1, 1)
        )

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Reward and Punishment", rotation=-90, va="bottom")

        # Plot the start (green) and goal (red) positions
        ax.scatter(
            self.start_coord[1],
            self.start_coord[0],
            color="darkorange",
            label="Start",
        )
        ax.scatter(
            self.goal_coord[1],
            self.goal_coord[0],
            color="green",
            label="Goal",
        )

        # Plot the mine (purple)
        ax.scatter(
            self.mine_coord[1],
            self.mine_coord[0],
            color="black",
            label="Mine",
        )
        # print(rewards[self.start_coord])

        # Add the legend
        # ax.legend()

        # Draw grid lines
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.tick_params(axis="both", which="both", length=0)
        ax.grid(which="minor", color="black", linewidth=0.3)
        ax.set_xlabel(
            f"reward's magnitude = {self.reward_val}, punishment's magnitude = {self.punishment_val}",
            va="center",
        )
        # Show the plot
        plt.show()


maze = MazeEnv(rows=10, cols=10)
maze.set_coord_start_goal_mine(
    start_coord=(0, 0), goal_coord=(9, 9), mine_coord=(4,5)
)

maze.set_density_magnitude_goal_mine(
    punishment_val=1,  reward_val=.2, punishment_density=2, reward_density=2
)
maze.plot_maze()
