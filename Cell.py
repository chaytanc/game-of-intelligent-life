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
        # default do nothing
        self.move = np.array([1, 0, 0, 0, 0])
        self.x = 0
        self.y = 0
        self.losses = []
        self.pred = None

    '''
    Represents the convnet cell as a numpy array, useful for storing in the CAGame().grid prop.
    '''

    def vector(self) -> np.ndarray:
        vec = np.concatenate([self.color, self.network_vec, self.move, self.fitness])
        return vec

    def updateColor(self):
        self.color = self.network.getNetworkColor()

    # todo may need to normalize the loss somehow such that the fitness value is numerically stable
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

    # Five channels before fitness are for movement
    @staticmethod
    def getMovement(x, y, grid):
        # nothing, left, right, up, down
        movement_vector = grid.data[y, x][-6:-1]
        movement = np.argmax(movement_vector)
        if x <= 1 and movement == 1:
            movement_vector = [1, 0, 0, 0, 0]
        if x >= 98 and movement == 2:
            movement_vector = [1, 0, 0, 0, 0]
        if y <= 1 and movement == 3:
            movement_vector = [1, 0, 0, 0, 0]
        if y >= 98 and movement == 4:
            movement_vector = [1, 0, 0, 0, 0]
        return movement_vector

    @staticmethod
    def getCellNetwork(x, y, grid):
        return grid.data[x, y][3:-6]

    def __str__(self):
        return str(self.color)


    def __repr__(self):
        return str(self.color)
