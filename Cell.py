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
        self.last_neighbors = np.array([[0.0, 0.0, 0.0, 0.0] * 9])
        self.x = 0
        self.y = 0
        # self.ch_dim = channel_dim

    '''
    Represents the convnet cell as a numpy array, useful for storing in the CAGame().grid prop.
    '''
    def vector(self) -> np.ndarray:
        vec = np.concatenate([self.color, self.network_vec, self.fitness])
        return vec

    def updateColor(self):
        self.color = self.network.getNetworkColor()


    #todo may need to normalize the loss somehow such that the fitness value is numerically stable
    '''
    Updates the cell's fitness according to the accuracy of its predictions and how fit its neighbors predict it to be
    '''
    def updateFitness(self, loss):
        # todo call this somewhere
        # Social fitness term normalizes over the predictions the neighbors of this cell estimated its fitness to be
        # social_fitness = np.sum(self.neighbors_fit_predictions) / len(self.neighbors_fit_predictions)
        inv_loss_fitness = 1 / loss  # XXX add time alive term
        # self.fitness = 0.5 * inv_loss_fitness + 0.5 * social_fitness
        self.fitness = inv_loss_fitness

    @staticmethod
    def getCellColor(x, y, grid):
        # select and return first three rgb channels
        return grid.data[x, y][:3]

    @staticmethod
    def getCellFitness(x, y, grid):
        return grid.data[x, y][-2:-1]

    @staticmethod
    def getCellNetwork(x, y, grid):
        return grid.data[x, y][3:-2]

