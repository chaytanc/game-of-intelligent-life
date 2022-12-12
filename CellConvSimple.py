import numpy as np
from torch import nn
import torch
from torch.autograd import Function
from torch.autograd.grad_mode import F
from tqdm import tqdm

from Grid import Grid
from Summary import Summary
import math


class CellConv(nn.Module):
    '''
    Simple conv autoencoder implementation but conv autoencoder style, without fc layer out
    '''

    def __init__(self, output_shape=(100, 100, 4), observability='partial'):
        '''

        :param block_class: ResidualBlock class to use to make layer modules
        :param layers: List of length 4 where each ind is the number of residual
                block_classs in the layer corresponding to its index
        :param output_shape: (grid_h, grid_w, channels) of the shape of the output
        :param observability: If partial, cell only receives input from neighbors when predicting next frame
        '''
        super(CellConv, self).__init__()
        self.output_shape = output_shape
        if observability == 'partial':
            kernel_size = 3
            padding = 1
        else:
            kernel_size = 7
            padding = 3

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
            nn.Tanh()
        )


    def deconvolve(self, x):
        desired_w = self.output_shape[1]
        desired_h = self.output_shape[0]
        if x.shape[3] < desired_w or x.shape[2] < desired_h:
            output_shape = (x.shape[0], x.shape[1], desired_w, desired_h)
            up1 = nn.ConvTranspose2d(512, 128, stride=2, kernel_size=3) # (1, 128, 7, 7)
            up2 = nn.ConvTranspose2d(128, 32, stride=3, kernel_size=5) # (1, 32, 23, 23)
            up3 = nn.ConvTranspose2d(32, 8, stride=3, kernel_size=5) # (1, 8, 71, 71)
            up4 = nn.ConvTranspose2d(8, 4, stride=2, kernel_size=2, padding=21)  #(1, 4, 100, 100)
            lrelu = nn.LeakyReLU()
            x = up1(x)
            x = lrelu(x)
            x = up2(x)
            x = lrelu(x)
            x = up3(x)
            x = lrelu(x)
            x = up4(x)
            x = lrelu(x)
        return x
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # x = self.avgpool(x)

        # Get channels to match, then upsample to get desired h, w dims
        x = self.layer4(x)
        # Adding upsampling to get output of convnet into grid shape, since stride > 1 downsampled
        # Note: similar to autoencoder
        # Compare (n, c, h, w) to (h, w, c) provided
        desired_w = self.output_shape[1]
        desired_h = self.output_shape[0]
        if x.shape[3] < desired_w or x.shape[2] < desired_h:
            output_shape = (x.shape[0], x.shape[1], desired_h, desired_w)
            up = nn.Upsample(output_shape) #TODO can play around with method of upsampling ie mode=...
            x = up(x)

        # XXX dunno if this is desirable, what shape do we want output? We want same np.size, 3-d structure as state
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return x

    '''
    Represent the first and last layers of the network as flattened numpy arrays of those parameters, concatenated
    '''
    def getNetworkParamVector(self):
        first_params = self.layer0.parameters().__next__().detach()
        last_params = self.layer3.parameters().__next__().detach()
        first, last = CellConv.firstLastParams(first_params, last_params)
        first = first.detach().numpy().flatten()
        last = last.detach().numpy().flatten()
        network_param_vector = np.concatenate([first, last])
        return network_param_vector

    @staticmethod
    def getParams(layer):
        size = 1
        for dim in layer.shape:
            size *= dim
        return size

    '''
    Reduces the size of first and last layer params to smaller number of params by first reducing channel numbers w 
    1-d convolutions, then simply truncating if that does not work... may need to choose one strat or the other
    '''
    @staticmethod
    def firstLastParams(*args):
        layers_params = []
        for layer_params in args:
            size = CellConv.getParams(layer_params)

            # Reducing channel size until params are manageable. If flattening channels to 1 ch isn't enough,
            # reduce samples in batch by half until under 1500
            last_out = 0
            while size > 1500:
                out = layer_params.shape[1] // 2 if layer_params.shape[1] // 2 != 0 else 1
                if last_out == out:
                    with torch.no_grad():
                        dim0 = layer_params.shape[0] // 2
                        layer_params = layer_params[:dim0, :layer_params.shape[1],
                                       :layer_params.shape[2], :layer_params.shape[3]]
                conv = torch.nn.Conv2d(in_channels=layer_params.shape[1], out_channels=out, kernel_size=1)
                layer_params = conv(layer_params)
                size = CellConv.getParams(layer_params)
                last_out = out

            layers_params.append(layer_params)
        return layers_params

    @staticmethod
    def sigmoid(x):
        sig = 1 / (1 + math.exp(-x))
        return sig

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
    Encodes a model architecture summary into an ascii numpy array
    (not currently used)
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
    Convert from (n, c, h, w) format of network output to (h, w, c) format of the grid
    '''
    @staticmethod
    def reshape_output(output: torch.Tensor, output_shape):
        output = torch.squeeze(output)
        output = torch.reshape(output, output_shape)
        return output

    ''' 
    Can switch between passing in full previous state or only partially observable prev state / neighbors
    '''
    @staticmethod
    def train_module(cell, full_state, prev_state=None):
        # Loss and optimizer XXX proabably move these more global
        num_epochs = 2
        # batch_size = 16  # XXX todo ???
        learning_rate = 0.01
        # criterion = CA_Loss()
        net = cell.network
        #TODO hyperparam tuning
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=learning_rate,
            weight_decay=0.001, #TODO Check weight decay params
            momentum=0.9)
        loss = 1000  # defaults to big
        for epoch in tqdm(range(num_epochs)):
            # Note: no inner for loop here because only doing one frame pred
            # at a time
            # Make a prediction on the previous state and verify if it matches current state
            # feed neighbors for cell and ask for full grid predictions
            net = net.double()
            input = torch.tensor(cell.last_neighbors)
            input = input[None, :, :, :]
            next_full_state_pred = net(input)
            next_full_state_pred = CellConv.reshape_output(next_full_state_pred, full_state.shape)
            # loss = criterion()
            loss = CA_Loss(next_full_state_pred, full_state)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        cell.updateColor()
        return next_full_state_pred

'''
Computes the loss of a cell based on the predicted state of the whole grid (color and fitness) vs actual
'''
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
