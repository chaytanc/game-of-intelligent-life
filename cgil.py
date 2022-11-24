from sys import exit
from math import ceil

import torch

import tqdm
# from IPython.display import Image, HTML, clear_output
# import moviepy.editor as mvp
# from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
# import PIL.Image, PIL.ImageDraw
import numpy as np
from Cell import Cell
from CellConv import CellConv
import pygame
from Grid import Grid
from ResidualBlock import ResidualBlock

# Constants
CELL_SIZE = 10  # pixels
GRID_W = 100  # cells
GRID_H = 100  # cells
FIT_CHANNELS = 1
# How many cells to stochastically choose to update at each next frame eval
UPDATES_PER_STEP = 50
# initialize grid
CLOCK = pygame.time.Clock()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# TODO Biochemical signaling channels?
def setChannels(first_params, last_params):
    network_params_size = ((first_params.numpy().flatten().size +
                            last_params.numpy().flatten().size))
    CHANNELS = 3 + network_params_size + FIT_CHANNELS
    return CHANNELS


class CAGame():

    def __init__(self):
        cell_net = CellConv(ResidualBlock, [3, 4, 6, 3]).to(device)
        first_params = cell_net.layer0.parameters().__next__().detach()
        last_params = cell_net.layer3.parameters().__next__().detach()
        first_params, last_params = self.firstLastParams(first_params, last_params)
        self.network_params_size = ((first_params.numpy().flatten().size +
                                     last_params.numpy().flatten().size))
        CHANNELS = setChannels(first_params, last_params)

        # Grid of Cell objects, corresponding with their vectorized forms stored below for computation
        self.cell_grid = [[Cell(self.network_params_size)] * GRID_W] * GRID_H
        # Grid, holding vectorized cells in data
        self.grid = Grid(CELL_SIZE, grid_size=(GRID_W, GRID_H, CHANNELS), network_params_size=self.network_params_size)
        self.screen = pygame.display.set_mode(self.grid.res)
        self.isRunning = False
        pygame.display.set_caption("Cellular Automata", "CA")

    def getParams(self, layer):
        size = 1
        for dim in layer.shape:
            size *= dim
        return size

    def firstLastParams(self, *args):
        layers = []
        # todo max pool instead of truncate!!!
        for layer in args:
            size = self.getParams(layer)
            while size > 1000:
                m = torch.nn.MaxPool2d(kernel_size=3, stride=2)
                layer = m(layer)
                size = self.getParams(layer)
            layers.append(layer)
        return layers

    '''
    Enforces updating the corresponding grid data when the cell object changes
    '''
    def updateCellGrid(self, cell, x, y):
        self.cell_grid[x][y] = cell
        self.grid.data[x][y] = cell.vector()

    def testCellConv(self):
        cell_net = CellConv(ResidualBlock, [3, 4, 6, 3]).to(device)
        color = [256, 0, 0]
        cell = Cell(color=color, network_param_size=self.network_params_size, network=cell_net, fitness=10)
        self.updateCellGrid(cell, 0, 0)

    # do for every cell
    def updateCell(self, node):
        neighbors = []
        # XXX how to get x y coords of cells
        x = node.Xm
        y = node.Ym
        # Get cell's neighbors XXX check that range works righ
        for nx in range(-1, 1):
            for ny in range(-1, 1):
                if nx != 0 and ny != 0:
                    neighbors.append(self.grid.data[x + nx, y + ny])
                    # XXX make numpy array of neighbor colors

        # feed neighbors colors as input array to conv net
        # call forward on cell
        # output next colors and fitness preds of neighbors
        # update neighbors' fitness pred prop based on this cell's prediction
        # concatenate into state prediction
        # take movement, maybe get eaten
        # new cell location assume color of predicted form
        # compare prediction to actual s' after movement to get loss
        # update cell weights

    '''
    Modifies cell.fitness
    '''

    def fitnessUpdate(cell, loss):
        norm_fitness = np.sum(cell.neighbors_fit_predictions) / len(cell.neighbors_fit_predictions)
        inv_loss_fitness = 1 / loss  # XXX add time alive term
        cell.fitness = 0.5 * inv_loss_fitness + 0.5 * norm_fitness

    def writeFrame(self):
        # randomly sample grid to call updates on
        x_inds = np.choice(GRID_W, UPDATES_PER_STEP)
        y_inds = np.choice(GRID_H, UPDATES_PER_STEP)
        # update cells in sample
        for x, y in zip(x_inds, y_inds):
            self.updateCell(self.grid.data[x][y])
        # update grid, fitnesses? cell writes self to grid
        pass

    # If cells move on top of each other, check how to break ties / which gets eaten and which replicates
    # Free the memory used by the eaten cell and allocate new instance of replicated
    def eatCells(self):
        pass

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

    def draw(self):
        # draws everything onto the screen

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
        while True:
            CLOCK.tick(70)  # Makes game run at 70 fps or slower
            self.testCellConv()
            # self.screen.fill(self.cellColors[0])
            # XXX removed or framebyframe
            if self.isRunning:
                pass
            # XXX todo finish updateCells func
            # self.update()

            self.draw()
            self.eventHandler()
            # XXX why flip
            pygame.display.flip()


def main():
    ca = CAGame()
    ca.startGame()


if __name__ == '__main__':
    main()

# hash network architecture and parameters to color
# dimensions of environment: rgb hashed color / conv net associated with that pixel, fitness
# Input to agents: neighbor's colors, 
# get next frame based on which agents move where and which get eaten
# get loss of each pixel based on predictions and on accuracy of judgment of neighbors' fitness functions
# update fitness functions as summations of neighbors' predictions and own predictions
# update frame, write to video
# pygame while loop
# channels=fitness, rgb of model architecture, environment constraints
# dimensions: in = (neighborsw x neighborsh x channels), out = (gridw x gridh x channels + movement)

# draw on grid example with video writer thing
# '''
# with VideoWriter('teaser.mp4') as vid:
#     x = np.zeros([len(EMOJI), 64, 64, CHANNEL_N], np.float32)
#     # grow
#     for i in tqdm.trange(200):
#         k = i // 20
#         if i % 20 == 0 and k < len(EMOJI):
#             x[k, 32, 32, 3:] = 1.0
#         vid.add(zoom(tile2d(to_rgb(x), 5), 2))
#         for ca, xk in zip(models, x):
#             xk[:] = ca(xk[None, ...])[0]
# mvp.ipython_display('teaser.mp4', loop=True)
# # update grid during training
# with VideoWriter(out_fn) as vid:
#     for i in tqdm.trange(500):
#         vis = np.hstack(to_rgb(x))
#         vid.add(zoom(vis, 2))
#         for ca, xk in zip(models, x):
#             xk[:] = ca(xk[None, ...])[0]
# # @title Training Progress (Batches)
# frames = sorted(glob.glob('train_log/batches_*.jpg'))
# mvp.ImageSequenceClip(frames, fps=10.0).write_videofile('batches.mp4')
# mvp.ipython_display('batches.mp4')
# '''
