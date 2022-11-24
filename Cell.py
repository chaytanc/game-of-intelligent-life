import numpy as np

'''
Interesting that the objects in this game are just the landscape, the environment
and the objects themselves are specific permutations of it that propagate but are not
explicitly embodied or modeled
'''


class Cell():

    def __init__(self, network_param_size, channel_dim=5, color=(0, 0, 0), network=None, fitness=-1):
        self.color = np.array(color)
        self.network = network
        if network:
            self.network_vec = network.getNetworkParamVector()
            self.color = network.getNetworkColor()
        else:
            self.network_vec = np.zeros(network_param_size)
        self.fitness = np.array([fitness])
        self.neighbors_fit_predictions = []
        # self.ch_dim = channel_dim

    def vector(self) -> np.ndarray:
        vec = np.concatenate([self.color, self.network_vec, self.fitness])
        return vec

    # TODO probably better as a prop of the cell not the grid?
    @staticmethod
    def getCellColor(x, y, grid):
        # select and return first three rgb channels
        return grid.data[x, y][:3]

    @staticmethod
    def getCellFitness(x, y, grid):
        return grid.data[x, y][-2:-1]

    @staticmethod
    def getCellNetwork(x, y, grid):
        return grid.data[x, y][3:4]
