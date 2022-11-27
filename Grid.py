from numpy import full
from Cell import Cell


class Grid():

    def __init__(self, cell_size, grid_size, network_params_size):
        self.cell_size = cell_size
        self.size = grid_size
        self.channels = grid_size[2]
        self.res = (grid_size[0] * cell_size, grid_size[1] * cell_size)
        # Inits data with black cells by default, no network is configured
        self.data = full(self.size, Cell(network_params_size).vector())
