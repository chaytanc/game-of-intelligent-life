import numpy as np
from torch import nn
import torch
from torch.autograd import Function
from torch.autograd.grad_mode import F
from Grid import Grid
from Summary import Summary
import math


class CellConv(nn.Module):
    '''
    ResNet implementation but conv autoencoder style, without fc layer out
    https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/
    '''

    def __init__(self, block_class, layers):
        '''

        :param block_class: ResidualBlock class to use to make layer modules
        :param layers: List of length 4 where each ind is the number of residual
                block_classs in the layer corresponding to its index
        '''
        super(CellConv, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_block_layers(block_class, 64, layers[0], stride=1)
        self.layer1 = self._make_block_layers(block_class, 128, layers[1], stride=2)
        self.layer2 = self._make_block_layers(block_class, 256, layers[2], stride=2)
        self.layer3 = self._make_block_layers(block_class, 512, layers[3], stride=2)
        # Todo make more robust by having game determine this hyperparm rather than relying on only using this architecture
        # XXX note need to apply some sort of fitness prediction normalization
        # XXX no fully connected b/c want grid output, not classes
        # what is planes
        # what does batch norm do again: keeps track of mean and std dev for batch and normalizes inputs to layers

    def _make_block_layers(self, block_class, planes, num_block_classs, stride=1):
        # In order to concatenate the residual with the output we needed to save the downsample (or upsample) and apply to input
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.inplanes, out_channels=planes,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []  # A list of residual blocks
        layers.append(block_class(self.inplanes, planes, stride, downsample))
        self.inplanes = planes  # XXX todo will all layers have same input size as output size?
        for i in range(1, num_block_classs):
            layers.append(block_class(in_channels=self.inplanes,
                                      out_channels=planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0),
                   -1)  # XXX dunno if this is desirable, what shape do we want output? We want same np.size, 3-d structure as state
        # x = self.fc(x)
        return x

    def train_module(self, cell, state, prev_state):
        # Loss and optimizer XXX proabably move these more global
        num_epochs = 20
        batch_size = 16  # XXX todo ???
        learning_rate = 0.01
        # criterion = CA_Loss()
        net = cell.network
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=learning_rate,
            weight_decay=0.001,
            momentum=0.9)
        for epoch in range(num_epochs):
            # Note: no inner for loop here because only doing one frame pred
            # at a time
            # Make a prediction on the previous state and verify if it matches current state
            # XXX --> will have to bootstrap at the start / just run one iteration without training??
            prev_state_pred = net(prev_state)
            # loss = criterion()
            loss = CA_Loss(prev_state_pred, state)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def getNetworkParamVector(self):
        # first_weights = np.array(self.layer0.parameters()).flatten()
        # last_weights = np.array(self.layer3.parameters()).flatten()
        first_params = self.layer0.parameters().__next__().detach().numpy().flatten()
        last_params = self.layer3.parameters().__next__().detach().numpy().flatten()
        # TODO make robust by trimming and padding in order to accommodate diff architectures
        # todo layer0 is in fact a whole module of layers, not just one layer and I could try to account for that
        network_param_vector = np.concatenate([first_params, last_params])
        return network_param_vector

    @staticmethod
    def sigmoid(x):
        sig = 1 / (1 + math.exp(-x))
        return sig

    # XXX todo make color depend on params so that we can actually see something that differentiates the cells
    # and move to cellconv
    def getNetworkColor(self):
        # sum of first layer weights, sum of last layer weights, sum of middle layer weights??
        first_params = self.layer0.parameters().__next__().detach().numpy()
        middle_params0 = self.layer1.parameters().__next__().detach().numpy()
        # middle_params1 = self.layer2.parameters().__next__().detach().numpy()
        last_params = self.layer3.parameters().__next__().detach().numpy()
        color = [self.sigmoid(np.average(first_params)),
                 self.sigmoid(np.average(last_params)),
                 self.sigmoid(np.average(middle_params0))]
        color = np.array(color) * 256
        return color

    '''
    Encodes a model into an RGB value
    '''

    @staticmethod
    def getNetworkASCIIArray(model: nn.Module):
        # summ = summary(model, input_size)
        summ = model.__str__()
        ascii = np.array([ord(c) for c in summ])
        return ascii

    '''
    Encodes a model into an RGB value
    '''

    @staticmethod
    def architectureToRGB(model: nn.Module, input_size=None):
        # project 1d array ascii model into 3 numbers --> this has a nonzero nullspace...
        # not able to recover all info
        summ = Summary(model)
        rgb_channel = np.array((summ.depth, summ.avg_relative_layer_size, summ.total_params))
        return rgb_channel

    '''
    Decodes an RGB value into a model architecture
    '''
    # @staticmethod
    # def RGBtoArchitecture(model: nn.Module):
    #     pass


def CA_Loss(y_pred, y):
    # XXX should be able to use autograd since y_pred was requires_grad
    next_frame_pred = Grid.getColorChannels(y_pred)
    target_frame = Grid.getColorChannels(y)
    fit_preds = Grid.getFitnessChannels(y_pred)
    fit_targets = Grid.getFitnessChannels(y)
    frame_loss = F.binary_cross_entropy(next_frame_pred, target_frame)
    # frame_loss = F.mseloss(next_frame_pred, target_frame)
    fit_loss = F.mseloss(fit_preds, fit_targets)
    losses = [frame_loss, fit_loss]
    norm_loss = np.sum(losses) / len(losses)
    return norm_loss


"""
class CA_Loss(Function):
    '''
    https://stackoverflow.com/questions/65947284/loss-with-custom-backward-function-in-pytorch-exploding-loss-in-simple-mse-exa
    '''

    @staticmethod
    def forward(ctx, *args, **kwargs):
        y_pred = args[0]
        y = args[1]
        next_frame_pred = Grid.getColorChannels(y_pred)
        target_frame = Grid.getColorChannels(y)
        fit_preds = Grid.getFitnessChannels(y_pred)
        fit_targets = Grid.getFitnessChannels(y)
        ctx.save_for_backward(next_frame_pred, target_frame, fit_preds, fit_targets)
        # XXX todo figure out loss wrt bce??
        # https://math.stackexchange.com/questions/2503428/derivative-of-binary-cross-entropy-why-are-my-signs-not-right
        # frame_loss = F.binary_cross_entropy(next_frame_pred, target_frame)
        frame_loss = F.mseloss(next_frame_pred, target_frame)
        fit_loss = F.mseloss(fit_preds, fit_targets)
        losses = [frame_loss, fit_loss]
        norm_loss = np.sum(losses) / len(losses)
        return norm_loss

    @staticmethod
    def backward(ctx, *grad_outputs):
        # y_pred, y = ctx.saved_tensors
        next_frame_pred, target_frame, fit_preds, fit_targets = ctx.saved_tensors
        # XXX need help: why is there no X term in the example I am seeing
        # do I need to take in to consideration activation functions when I am writing these?? or just for BCE cause it
        # often contains sigmoid layer

        # https://stackoverflow.com/questions/65947284/loss-with-custom-backward-function-in-pytorch-exploding-loss-in-simple-mse-exa
        # Why are these not negative
        frame_deriv = 2 * (next_frame_pred - target_frame) / next_frame_pred.shape[0]
        fit_deriv = 2 * (fit_preds - fit_targets) / fit_preds.shape[0]
        return frame_deriv + fit_deriv
"""
