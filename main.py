from sys import exit
from math import ceil
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from Cell import Cell
from CellConv import CellConv
from ResidualBlock import ResidualBlock
from CellConvSimple import CellConvSimple
import pygame
from Grid import Grid
from CellConvSimple import partial_CA_Loss

# Constants
OBSERVABILITY = 'partial'
CELL_SIZE = 10  # pixels
GRID_W = 100  # cells
GRID_H = 100  # cells
FIT_CHANNELS = 1
MOVE_CHANNELS = 5
NUM_EPOCHS = 1
# How many cells to stochastically choose to update at each next frame eval
# UPDATES_PER_STEP = 50
# initialize grid
CLOCK = pygame.time.Clock()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''Sets up the size constant of the channels that each cell in the grid will hold. We will have 3 for color, 
some amount for representing the parameters of the conv net at that location, and some to represent the fitness of 
the cell. '''


# TODO Biochemical signaling channels? movement channel
def setChannels(first_params, last_params):
    network_params_size = ((first_params.detach().numpy().flatten().size +
                            last_params.detach().numpy().flatten().size))
    # 3 channels for rgb, + network params size + fitness channels
    CHANNELS = 3 + network_params_size + MOVE_CHANNELS + FIT_CHANNELS
    return CHANNELS


'''
This class continuously draws the grid on the pygame window according to cell update rules and dynamics
'''


class CAGame():

    def __init__(self):
        # cell_net = CellConv(ResidualBlock, [3, 4, 6, 3], observability=OBSERVABILITY).to(device)
        cell_net = CellConvSimple().to(device)
        first_params = cell_net.layer0.parameters().__next__().detach()
        last_params = cell_net.layer3.parameters().__next__().detach()
        # first_params, last_params = CellConv.firstLastParams(first_params, last_params)
        first_params, last_params = CellConvSimple.firstLastParams(first_params, last_params)
        self.network_params_size = ((first_params.detach().numpy().flatten().size +
                                     last_params.detach().numpy().flatten().size))
        CHANNELS = setChannels(first_params, last_params)
        OUTPUT_SHAPE = (GRID_H, GRID_W, CHANNELS)
        cell_net.output_shape = OUTPUT_SHAPE

        # Grid of Cell objects, corresponding with their vectorized forms stored below for computation
        self.cell_grid: [[Cell]] = [[0] * GRID_W for _ in range(GRID_H)]
        # Has extra dim so multiple cells can occupy the same place in the grid
        # self.intermediate_cell_grid = [[[]] * GRID_W for _ in range(GRID_H)]
        # Want 100, 100, 1
        self.intermediate_cell_grid = [[[] for col in range(GRID_W)] for row in range(GRID_H)]
        for r in range(GRID_H):
            for c in range(GRID_W):
                new_cell = Cell(self.network_params_size)
                new_cell.y = r  # todo changing r and c
                new_cell.x = c
                self.cell_grid[r][c] = new_cell
                self.intermediate_cell_grid[r][c].append(new_cell)

        # Grid, holding vectorized cells in data used to actually get loss of cells
        self.grid = Grid(CELL_SIZE, grid_size=(GRID_W, GRID_H, CHANNELS), network_params_size=self.network_params_size)
        self.screen = pygame.display.set_mode(self.grid.res)
        self.empty_vector = np.zeros((CHANNELS))
        pygame.display.set_caption("Cellular Automata", "CA")

    '''
    Enforces updating the corresponding grid data when the cell object changes
    '''

    def updateCellGrid(self, cell, x, y):
        self.cell_grid[y][x] = cell
        self.grid.data[y][x] = cell.vector()
        cell.x = x
        cell.y = y

    def updateIntermediateCellGrid(self, cell, x, y):
        # Assume that cell.move contains direction already
        movement_vector = cell.getMovement(x, y, self.grid)
        direction = np.argmax(movement_vector)  # [0, 0, 0, 0, 0]
        #  stay, left, right, up, down
        if direction == 1:
            next_pos = x - 1, y
        elif direction == 2:
            next_pos = x + 1, y
        elif direction == 3:
            next_pos = x, y - 1
        elif direction == 4:
            next_pos = x, y + 1
        else:  # 0
            next_pos = x, y

        self.intermediate_cell_grid[next_pos[1]][next_pos[0]].append(cell)
        # Make cell be empty if it moved
        if x != next_pos[1] and y != next_pos[0]:
            cell = Cell(network_param_size=self.network_params_size)
            self.intermediate_cell_grid[y][x] = [cell]

    def getPartialFrame(self, cell, frame_size=(3, 3)):  # will break if cells are on the border
        vector_neighbors = np.zeros(shape=(frame_size[0], frame_size[1], 9))
        x = cell.x
        y = cell.y
        print('cell accessed partial frame at: (' + str(x) + ', ' + str(y) + ')')
        for nx in range(-1, 2):
            for ny in range(-1, 2):
                vector_neighbors[ny + 1][nx + 1][:3] = self.grid.data[y + ny, x + nx, :3]
                vector_neighbors[ny + 1][nx + 1][3:9] = self.grid.data[y + ny, x + nx, -6:]
        return vector_neighbors

    def getFullFrame(self):
        vector_neighbors = np.zeros(shape=(3, 3, 9))
        vector_neighbors[:, :, 3] = self.grid.data[:, :, :3]
        vector_neighbors[:, :, 3:9] = self.grid.data[:, :, -6:]
        return vector_neighbors

    def testCellConv(self):
        # cell_net = CellConv(ResidualBlock, [3, 4, 6, 3], observability=OBSERVABILITY).to(device)
        color = [256, 0, 0]
        locations = np.arange(1, 98)  # Can't spawn on the borders
        xy = np.random.choice(locations, (20, 2), replace=False)
        # nothing, left, right, up, down
        directions = [0, 1, 2, 3, 4]
        valid_directions = directions
        for i in range(0, 20):
            cell_net = CellConvSimple().to(device)
            cell = Cell(color=color, network_param_size=self.network_params_size, network=cell_net, fitness=10)
            x, y = xy[i]
            print('generated cell at: (' + str(x) + ', ' + str(y) + ')')
            if x == 98:
                valid_directions.remove(2)
            if x == 1:
                valid_directions.remove(1)
            if y == 98:
                valid_directions.remove(4)
            if y == 1:
                valid_directions.remove(3)
            direction = np.random.choice(valid_directions)
            cell.move = [0, 0, 0, 0, 0]
            cell.move[direction] = 1
            self.updateIntermediateCellGrid(cell, x, y)
            self.updateCellGrid(cell, x, y)


    # Loop through each cell, get movement, update intermediate cell grid
    # after temporarily moving all cells, loop through again and see if any moved to the same place
    # if so call eatCells to break the tie and updateCellGrid, else just updateCellGrid
    def moveCell(self, cell):
        # for y, row in enumerate(self.cell_grid):
        #     for x, cell in enumerate(row):
        if cell.network:
            movement_vector = Cell.getMovement(cell.x, cell.y, self.grid)
            cell.move = movement_vector
            self.updateIntermediateCellGrid(cell, cell.x, cell.y)

    def resolveIntermediateCellGrid(self):
        for y, row in enumerate(self.intermediate_cell_grid):
            for x, cell in enumerate(row):
                # try:
                if len(cell) > 1:
                    self.eatCells(x, y)
                else:
                    # Remove cells that moved away if intermediate cell grid is black and cell grid is not
                    if self.cell_grid[y][x].network and not cell[0].network:
                        self.cell_grid[y][x] = Cell(self.network_params_size)
                        self.grid.data[y][x] = self.empty_vector

                    self.updateCellGrid(cell[0], x, y)

    # If cells move on top of each other, check how to break ties / which gets eaten and which replicates
    # Free the memory used by the eaten cell and allocate new instance of replicated
    def eatCells(self, x, y):
        # call if computed updated grid has two cells
        # assumes more than one cell at self.intermediate_cell_grid
        dominant_cell = self.intermediate_cell_grid[y][x][0]
        for cell in self.intermediate_cell_grid[y][x]:
            if cell.fitness > dominant_cell.fitness:
                dominant_cell = cell
        self.updateCellGrid(dominant_cell, x, y)

    # do for every cell, add neighbors' fitness predictions to cell as we go
    # update each cell's fitness at end of running all training of cells in grid
    def updateCell(self, node: Cell, previous_grid=None):
        vector_neighbors = np.zeros(shape=(9, 3, 3))
        neighbors = []
        x = node.x
        y = node.y
        # Get cell's neighbors, 3x3
        for nx in range(-1, 2):
            for ny in range(-1, 2):
                vector_neighbors[0][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, 0]
                vector_neighbors[1][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, 1]
                vector_neighbors[2][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, 2]
                vector_neighbors[3][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, -6]
                vector_neighbors[4][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, -5]
                vector_neighbors[5][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, -4]
                vector_neighbors[6][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, -3]
                vector_neighbors[7][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, -2]
                vector_neighbors[8][ny + 1][nx + 1] = self.grid.data[y + ny, x + nx, -1]
                neighbor = self.cell_grid[y + ny][x + nx]
                neighbors.append(neighbor)

        node.last_neighbors = vector_neighbors
        # After we update the cell, update the previous neighbors to the current grid config
        # Removes the network params from the grid state
        # full_state = np.dstack((self.grid.data[:, :, :3], self.grid.data[:, :, -1]))
        # pred, loss = CellConv.train_module(node, full_state=full_state, prev_state=previous_grid, num_epochs=NUM_EPOCHS)
        pred = self.train_module(node, num_epochs=NUM_EPOCHS)
        # todo update cell.fitness property based on loss
        return pred

    ''' 
    Can switch between passing in full previous state or only partially observable prev state / neighbors
    '''

    def train_module(self, cell, full_state=None, prev_state=None, num_epochs=1):
        net = cell.network

        # note, can't run more than one epoch w partial-partial structure
        for epoch in tqdm(range(num_epochs)):
            net = net.float()
            input = torch.from_numpy(cell.last_neighbors.astype(np.double))
            # Adds dimension to input so that it has n, c, w, h format for pytorch
            input = input[None, :, :, :]
            input = input.float().requires_grad_()
            next_pred = net(input)
            partial_pred_shape = (3, 3, 9)  # 9 channels: 3 color, 5 movement, 1 fitness
            # next_full_state_pred = CellConvSimple.reshape_output(next_full_state_pred, full_state.shape)
            next_pred = CellConvSimple.reshape_output(next_pred, partial_pred_shape)
            # take movement from the middle cell of the output
            cell.move = next_pred[1][1][-6:]
            # give movement to the grid which updates intermediate grid
            self.moveCell(cell)
            # once movements have all been calculated, give next frame to cell and backprop the loss
            # note probably have to move this backprop stuff to a separate function

        # return next_pred, loss.item()
        return next_pred

    def cellBackprop(self, cell):
        # TODO hyperparam tuning
        learning_rate = 0.01
        net = cell.network
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=learning_rate,
            weight_decay=0.001,  # TODO Check weight decay params
            momentum=0.9)
        next_frame = self.getPartialFrame(cell)  # default numpy (3, 3, 9)
        loss = partial_CA_Loss(cell.pred, next_frame, cell.x, cell.y)
        # print('pred: ', next_full_state_pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        cell.updateColor()
        # cell.updateFitness()
        return loss.item()

    '''
    Handle keyboard presses and other events
    '''

    def eventHandler(self):
        # Handles events sent by the user
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # find the cords to "place" a cell
                    (Mx, My) = pygame.mouse.get_pos()
                    Nx, Ny = ceil(Mx / self.grid.cell_size), ceil(My / self.grid.cell_size)
                    # XXX todo place some color of cell there / init new cell
                    # if self.grid.data[Nx, Ny].key == 0:
                    #     self.GameFlip.get(Nx, Ny).key = 1
                    # else:
                    #     self.GameFlip.get(Nx, Ny).key = 0

                elif event.eky == pygame.K_ESCAPE or \
                        event.type == pygame.QUIT:
                    pygame.display.quit(), exit()

    # MARK: pygame stuff
    '''
    Draw everything on the screen
    '''

    def draw(self):

        # draws cells onto the screen
        for x, row in enumerate(self.grid.data):
            for y, cell in enumerate(row):
                # if self.GameFlip.get(node.Xm, node.Ym).key == 1:
                # rect is (left, top, width, height)
                pygame.draw.rect(self.screen, Cell.getCellColor(x, y, self.grid),
                                 rect=((x) * self.grid.cell_size,
                                       (y) * self.grid.cell_size,
                                       self.grid.cell_size, self.grid.cell_size))

        # draw lines on the grid
        # if self.showGrid:
        for column in range(1, GRID_W):
            pygame.draw.line(self.screen, "gray", (column * self.grid.cell_size, 0),
                             (column * self.grid.cell_size, GRID_H * self.grid.cell_size))

        for row in range(1, GRID_H):
            pygame.draw.line(self.screen, "gray", (0, row * self.grid.cell_size),
                             (GRID_W * self.grid.cell_size, row * self.grid.cell_size))

    def startGame(self):
        iterations = 1000
        itr = 0
        self.testCellConv()
        running = True
        # while i < iterations:
        while running:
            CLOCK.tick(70)  # Makes game run at 70 fps or slower
            for row in self.cell_grid:
                for cell in row:
                    if cell.network:
                        pred = self.updateCell(cell)
                        cell.pred = pred
            self.resolveIntermediateCellGrid()
            for i, row in enumerate(self.cell_grid):
                for j, cell in enumerate(row):
                    cell.x = j  # x is col
                    cell.y = i  # y is row
                    # print('iteration: (', j, ', ', i, ')')
                    if cell.network:
                        print('cell backprop called at: (', cell.x, ', ', cell.y, ')')
                        print('iteration called at: (', j, ', ', i, ')')
                        loss = self.cellBackprop(cell)
                        cell.losses.append(loss)

            # Clear intermediate cell grid for next iteration
            # self.intermediate_cell_grid = [[ ['#' for col in range(2)] for col in range(GRID_W)] for row in range(GRID_H)]
            self.intermediate_cell_grid = [[ [] for col in range(GRID_W)] for row in range(GRID_H)]
            for r in range(GRID_H):
                for c in range(GRID_W):
                    self.intermediate_cell_grid[r][c].append(self.cell_grid[r][c])

            self.draw()
            self.eventHandler()
            pygame.display.flip()
            itr += 1
            if itr == iterations:
                running = False

        count = 0
        for i, row in enumerate(self.cell_grid):
            for j, cell in enumerate(row):
                if cell.network and count < 5:  # arbitrary cell index
                    print(cell.losses)
                    plt.title('Loss vs. Epoch for cell (' + str(j) + ', ' + str(i) + ')')
                    plt.xlim((0, 100))
                    plt.plot(np.arange(len(cell.losses)), cell.losses, 'g-', label="means")
                    plt.legend(loc="upper right")
                    plt.show()
                    count += 1


def main():
    ca = CAGame()
    ca.startGame()


if __name__ == '__main__':
    main()
