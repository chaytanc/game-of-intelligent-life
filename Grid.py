from numpy import full
from Cell import Cell


class Grid():

    def __init__(self, cell_size, grid_size, network_params_size):
        self.cell_size = cell_size
        self.size = grid_size
        self.channels = grid_size[2]
        self.res = (grid_size[0] * cell_size, grid_size[1] * cell_size)
        # Inits data with black cells by default, no network is configured
        # todo working here to init with rgb-ized cells, not cell objects bc we need a real numpy array, not
        # this non-number bullshit
        self.data = full(self.size, Cell(network_params_size).vector())

    # XXX todo working here should move this to the cell class and do an if statement where color is black if no network??
    # todo did move to cell now just need to replace
    @staticmethod
    def getColorChannels(data):
        # get first three depths of grid (rgb channels) while keeping all dimensions of the 100x100 grid
        channels = data[:, :, 0:3]
        return channels

    @staticmethod
    def getFitnessChannels(data):
        return data[:, :, -2:-1]

    @staticmethod
    def getNetworkChannels(data):
        # slice from the 3rd to the second to last in the channels layer for all cells in grid
        return data[:, :, 3:-2]
