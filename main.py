from sys import exit
from math import ceil
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from Cell import Cell
from CellConv import CellConv
from CellConvSimple import CellConvSimple
import pygame
from Grid import Grid
from ResidualBlock import ResidualBlock

# Constants
OBSERVABILITY = 'partial'
CELL_SIZE = 10  # pixels
GRID_W = 100  # cells
GRID_H = 100  # cells
FIT_CHANNELS = 1
MOVE_CHANNELS = 5
NUM_EPOCHS = 7
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
        self.cell_grid: [[Cell]] = [[0]*GRID_W for _ in range(GRID_H)]
        # Has extra dim so multiple cells can occupy the same place in the grid
        self.intermediate_cell_grid = [[[]]*GRID_W for _ in range(GRID_H)]
        for r in range(GRID_H):
            for c in range(GRID_W):
                new_cell = Cell(self.network_params_size)
                new_cell.x = r
                new_cell.y = c
                self.cell_grid[r][c] = new_cell
                self.intermediate_cell_grid[r][c].append(new_cell)

        # Grid, holding vectorized cells in data used to actually get loss of cells
        self.grid = Grid(CELL_SIZE, grid_size=(GRID_W, GRID_H, CHANNELS), network_params_size=self.network_params_size)
        self.screen = pygame.display.set_mode(self.grid.res)
        pygame.display.set_caption("Cellular Automata", "CA")

    '''
    Enforces updating the corresponding grid data when the cell object changes
    '''
    def updateCellGrid(self, cell, x, y):
        self.cell_grid[x][y] = cell
        self.grid.data[x][y] = cell.vector()
        cell.x = x
        cell.y = y
        self.updateIntermediateCellGrid(cell, x, y)

    def updateIntermediateCellGrid(self, cell, x, y):
        # Assume that cell.move contains direction already
        direction = np.argmax(cell.move)
        # right, left, down, up, stay
        if direction == 1:
            next_pos = x + 1, y
        elif direction == 2:
            next_pos = x - 1, y
        elif direction == 3:
            next_pos = x, y - 1
        elif direction == 4:
            next_pos = x, y + 1
        else:  # 0
            next_pos = x, y
        self.intermediate_cell_grid[next_pos[0]][next_pos[1]].append(cell)


    def testCellConv(self):
        # cell_net = CellConv(ResidualBlock, [3, 4, 6, 3], observability=OBSERVABILITY).to(device)
        color = [256, 0, 0]
        xy = np.random.choice(100, (20, 2), replace=False)
        # nothing, left, right, up, down
        directions = [0, 1, 2, 3, 4]
        valid_directions = directions
        for i in range(0, 20):
            cell_net = CellConvSimple().to(device)
            cell = Cell(color=color, network_param_size=self.network_params_size, network=cell_net, fitness=10)
            x, y = xy[i]
            if x == 99:
                valid_directions.remove(2)
            if x == 0:
                valid_directions.remove(1)
            if y == 99:
                valid_directions.remove(4)
            if y == 0:
                valid_directions.remove(3)
            direction = np.random.choice(valid_directions)
            cell.move[direction] = 1
            self.updateCellGrid(cell, x, y)

    # take next step by making all cells move according to their movement vector defined by the 5 movement channels
    # or move according to intermediate_cell_grid???

    # Loop through each cell, get movement, update intermediate cell grid
    # after temporarily movign all cells, loop through again and see if any moved to the same place
    # if so call eatCells to break the tie
    def moveCells(self):
        for y, row in enumerate(self.cell_grid):
            for x, cell in enumerate(row):
                if cell.network:
                    movement_vector = Cell.getMovement(x, y, self.grid)
                    cell.move = movement_vector
                    self.updateIntermediateCellGrid(cell, x, y)

        for y, row in enumerate(self.intermediate_cell_grid):
            for x, cell in enumerate(row):
                if cell.network:



    # do for every cell, add neighbors' fitness predictions to cell as we go
    # update each cell's fitness at end of running all training of cells in grid
    def updateCell(self, node: Cell, previous_grid=None):
        vector_neighbors = np.zeros(shape=(4, 3, 3))
        neighbors = []
        x = node.x
        y = node.y
        # Get cell's neighbors, 3x3
        for nx in range(-1, 2):
            for ny in range(-1, 2):
                # vector_row = []
                # if not (nx == 0 and ny == 0):
                # vector_neighbors.append(self.grid.data[x + nx, y + ny])
                vector_neighbors[0][nx + 1][ny + 1] = self.grid.data[x + nx, y + ny, 0]
                vector_neighbors[1][nx + 1][ny + 1] = self.grid.data[x + nx, y + ny, 1]
                vector_neighbors[2][nx + 1][ny + 1] = self.grid.data[x + nx, y + ny, 2]
                vector_neighbors[3][nx + 1][ny + 1] = self.grid.data[x + nx, y + ny, -1]
                neighbor = self.cell_grid[x + nx][y + ny]
                neighbors.append(neighbor)
                # vector_neighbors.append(vector_row)
                # else:
                #     neighbors.append()
        # vector_neighbors = np.array(vector_neighbors)
        node.last_neighbors = vector_neighbors
        # After we update the cell, update the previous neighbors to the current grid config
        # Removes the network params from the grid state
        #TODO change to partial state and get next frame of grid
        full_state = np.dstack((self.grid.data[:, :, :3], self.grid.data[:, :, -1]))

        # partial_state = vector_neighbors
        # pred, loss = CellConv.train_module(node, full_state=full_state, prev_state=previous_grid, num_epochs=NUM_EPOCHS)
        pred, loss = CellConvSimple.train_module(node, full_state=full_state, num_epochs=NUM_EPOCHS)
        # todo update cell.fitness property based on loss
        self.updateCellGrid(node, x, y)

        return loss

        # feed neighbors vectors as input array to conv net
        # train calls forward on cell
        # output next colors and fitness preds of neighbors
        # update neighbors' fitness pred prop based on this cell's prediction
        # concatenate into state prediction
        # take movement, maybe get eaten #todo
        # update cell colors after movements and weight changes
        # compare prediction to actual s' after movement to get loss
        # update cell weights


    # def writeFrame(self):
    #     # randomly sample grid to call updates on
    #     x_inds = np.choice(GRID_W, UPDATES_PER_STEP)
    #     y_inds = np.choice(GRID_H, UPDATES_PER_STEP)
    #     # update cells in sample
    #     for x, y in zip(x_inds, y_inds):
    #         self.updateCell(self.grid.data[x][y])
    #     # update grid, fitnesses? cell writes self to grid
    #     pass

    # If cells move on top of each other, check how to break ties / which gets eaten and which replicates
    # Free the memory used by the eaten cell and allocate new instance of replicated
    def eatCells(self, x, y):
        # call if computed updated grid has two cells
        # assumes more than one cell at self.intermediate_cell_grid
        dominant_cell = self.intermediate_cell_grid[x, y][0]
        for cell in self.intermediate_cell_grid[x, y]:
            if cell.fitness > dominant_cell.fitness:
                dominant_cell = cell
        self.updateCellGrid(dominant_cell, x, y)

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

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # start/stop
                    if self.isRunning:
                        self.isRunning = False
                    else:
                        self.isRunning = True

                # XXX do not know what this does
                # elif event.key == pygame.K_f:
                #     frame by frame
                # self.framebyframe = True

                # elif event.key == pygame.K_g:
                #     toggle grid
                #     if self.showGrid:
                #         self.showGrid = False

                # else:
                #     self.showGrid = True

                elif event.eky == pygame.K_ESCAPE or \
                        event.type == pygame.QUIT:
                    pygame.display.quit(), exit()

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
        iterations = 5
        i = 0
        losses = []
        self.testCellConv()
        while i < iterations:
            CLOCK.tick(70)  # Makes game run at 70 fps or slower
            p = True
            for row in self.cell_grid:
                for cell in row:
                    if cell.network:
                        loss = self.updateCell(cell)
                        if p:
                            losses.append(loss)
                            p = False

            self.draw()
            self.eventHandler()
            pygame.display.flip()
            i += 1
        print(losses)
        plt.title('Loss')
        plt.xlim((0, 100))
        plt.plot(np.arange(len(losses)), losses, 'g-', label="means")
        plt.legend(loc="upper right")
        plt.show()


def main():
    ca = CAGame()
    ca.startGame()


if __name__ == '__main__':
    main()

# dimensions of environment: rgb hashed color / conv net associated with that pixel, fitness
# Input to agents: neighbor's colors, 
# get next frame based on which agents move where and which get eaten
# get loss of each pixel based on predictions and on accuracy of judgment of neighbors' fitness functions
# update fitness functions as summations of neighbors' predictions and own predictions
# update frame
# pygame while loop
# channels=fitness, rgb of model architecture, environment constraints
# dimensions: in = (neighborsw x neighborsh x channels), out = (gridw x gridh x channels + movement)
