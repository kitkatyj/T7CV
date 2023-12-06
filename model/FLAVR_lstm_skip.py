import math
import numpy as np
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_3D import SEGating

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

def joinTensors(X1 , X2 , type="concat"):

    if type == "concat":
        return torch.cat([X1 , X2] , dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1


class Conv_2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if batchnorm:
            self.conv += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)

class upConv3D(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose" , batchnorm=False):

        super().__init__()

        self.upmode = upmode

        if self.upmode=="transpose":
            self.upconv = nn.ModuleList(
                [nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                SEGating(out_ch)
                ]
            )

        else:
            self.upconv = nn.ModuleList(
                [nn.Upsample(mode='trilinear', scale_factor=(1,2,2), align_corners=False),
                nn.Conv3d(in_ch, out_ch , kernel_size=1 , stride=1),
                SEGating(out_ch)
                ]
            )

        if batchnorm:
            self.upconv += [nn.BatchNorm3d(out_ch)]

        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x):

        return self.upconv(x)

class Conv_3d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    SEGating(out_ch)
                    ]

        if batchnorm:
            self.conv += [nn.BatchNorm3d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)

class upConv2D(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose" , batchnorm=False):

        super().__init__()

        self.upmode = upmode

        if self.upmode=="transpose":
            self.upconv = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)]

        else:
            self.upconv = [
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(in_ch, out_ch , kernel_size=1 , stride=1)
            ]

        if batchnorm:
            self.upconv += [nn.BatchNorm2d(out_ch)]

        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x):

        return self.upconv(x)


def joinTensors(X1 , X2 , type="concat"):

    if type == "concat":
        return torch.cat([X1 , X2] , dim=1)
    elif type == "add":
        return X1 + X2
    else:
        return X1


class Conv_2d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=False, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]

        if batchnorm:
            self.conv += [nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)

class upConv3D(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose" , batchnorm=False):

        super().__init__()

        self.upmode = upmode

        if self.upmode=="transpose":
            self.upconv = nn.ModuleList(
                [nn.ConvTranspose3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                SEGating(out_ch)
                ]
            )

        else:
            self.upconv = nn.ModuleList(
                [nn.Upsample(mode='trilinear', scale_factor=(1,2,2), align_corners=False),
                nn.Conv3d(in_ch, out_ch , kernel_size=1 , stride=1),
                SEGating(out_ch)
                ]
            )

        if batchnorm:
            self.upconv += [nn.BatchNorm3d(out_ch)]

        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x):

        return self.upconv(x)

class Conv_3d(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True, batchnorm=False):

        super().__init__()
        self.conv = [nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                    SEGating(out_ch)
                    ]

        if batchnorm:
            self.conv += [nn.BatchNorm3d(out_ch)]

        self.conv = nn.Sequential(*self.conv)

    def forward(self, x):

        return self.conv(x)

class upConv2D(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, upmode="transpose" , batchnorm=False):

        super().__init__()

        self.upmode = upmode

        if self.upmode=="transpose":
            self.upconv = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)]

        else:
            self.upconv = [
                nn.Upsample(mode='bilinear', scale_factor=2, align_corners=False),
                nn.Conv2d(in_ch, out_ch , kernel_size=1 , stride=1)
            ]

        if batchnorm:
            self.upconv += [nn.BatchNorm2d(out_ch)]

        self.upconv = nn.Sequential(*self.upconv)

    def forward(self, x):

        return self.upconv(x)


class UNet_3D_3D_lstm_skip(nn.Module):
    def __init__(self, block , n_inputs, n_outputs, batchnorm=False , joinType="concat" , upmode="transpose"):
        super().__init__()

        # nf = [512 , 256 , 128 , 64]        
        nf = [32 , 16 , 8 , 4]        
        out_channels = 3*n_outputs
        self.joinType = joinType
        self.n_outputs = n_outputs

        growth = 2 if joinType == "concat" else 1
        self.lrelu = nn.LeakyReLU(0.2, True)

        unet_3D = importlib.import_module(".resnet_3D" , "model")
        if n_outputs > 1:
            unet_3D.useBias = True
        self.encoder = getattr(unet_3D , block)(pretrained=False , bn=batchnorm)            

        self.decoder = nn.Sequential(
            Conv_3d(nf[0], nf[1] , kernel_size=3, padding=1, bias=True, batchnorm=batchnorm), # decoder[0]
            upConv3D(nf[1]*growth, nf[2], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode, batchnorm=batchnorm), # decoder[1]
            upConv3D(nf[2]*growth, nf[3], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode, batchnorm=batchnorm), # decoder[2]
            Conv_3d(nf[3]*growth, nf[3] , kernel_size=3, padding=1, bias=True, batchnorm=batchnorm), # decoder[3]
            upConv3D(nf[3]*growth , nf[3], kernel_size=(3,4,4), stride=(1,2,2), padding=(1,1,1) , upmode=upmode, batchnorm=batchnorm) # decoder[4]
        )

        self.feature_fuse = Conv_2d(nf[3]*n_inputs , nf[3] , kernel_size=1 , stride=1, batchnorm=batchnorm)

        self.outconv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(nf[3], out_channels , kernel_size=7 , stride=1, padding=0) 
        )      

        self.conv_lstm_skip_x0 = ConvLSTM(input_dim=4,  hidden_dim=64, kernel_size=(3,3),num_layers=1,batch_first=True, bias=True, return_all_layers=False)   
        self.conv_lstm_skip_x1 = ConvLSTM(input_dim=4,  hidden_dim=64, kernel_size=(3,3),num_layers=1,batch_first=True, bias=True, return_all_layers=False)
        self.conv_lstm_skip_x2 = ConvLSTM(input_dim=4,  hidden_dim=128, kernel_size=(3,3),num_layers=1,batch_first=True, bias=True, return_all_layers=False)
        self.conv_lstm_skip_x3 = ConvLSTM(input_dim=4,  hidden_dim=256, kernel_size=(3,3),num_layers=1,batch_first=True, bias=True, return_all_layers=False)


    def forward(self, images):
        # print("input images type", type(images)) # list
        images = torch.stack(images , dim=2)
        # print("images shape", images.shape) #  torch.Size([2, 3, 4, 256, 256])
        # print("images", len(images)) # len of images list is the batch size
        # print("image tensor", images[0].shape) # torch.Size([3, 4, 256, 256]) -> four context frames (C = 2) -> 2 context inputs from the past and future (C = 2)
        # print("images type", type(images)) # torch.Tensor
        ## Batch mean normalization works slightly better than global mean normalization, thanks to https://github.com/myungsub/CAIN
        mean_ = images.mean(2, keepdim=True).mean(3, keepdim=True).mean(4,keepdim=True)
        images = images-mean_ 
        # print("images", images.shape) # ([2, 3, 4, 256, 256])

        x_0 , x_1 , x_2 , x_3 , x_4 = self.encoder(images) # self.encoder = getattr(unet_3D , block)(pretrained=False , bn=batchnorm)     # unet_3D = importlib.import_module(".resnet_3D" , "model")
        
        # print("x_0",x_0.shape) # torch.Size([2, 64, 4, 128, 128])
        # print("x_1",x_1.shape) # torch.Size([2, 64, 4, 128, 128])
        # print("x_2",x_2.shape) # torch.Size([2, 128, 4, 64, 64])
        # print("x_3",x_3.shape) # torch.Size([2, 256, 4, 32, 32])
        # print("x_4",x_4.shape) # torch.Size([2, 512, 4, 32, 32])

        _, x_0_thru_lstm = self.conv_lstm_skip_x0(x_0)
        x_0 = x_0_thru_lstm[0][0]
        x_0 = x_0.unsqueeze(2)
        x_0 = torch.cat([x_0]*4, dim=2)
        # print("new x_0", x_0.shape)

        _, x_1_thru_lstm = self.conv_lstm_skip_x1(x_1)
        x_1 = x_1_thru_lstm[0][0]
        x_1 = x_1.unsqueeze(2)
        x_1 = torch.cat([x_1]*4, dim=2)
        # print("new x_1", x_1.shape)

        _, x_2_thru_lstm = self.conv_lstm_skip_x2(x_2)
        x_2 = x_2_thru_lstm[0][0]
        x_2 = x_2.unsqueeze(2)
        x_2 = torch.cat([x_2]*4, dim=2)
        # print("new x_2", x_2.shape)

        _, x_3_thru_lstm = self.conv_lstm_skip_x3(x_3)
        x_3 = x_3_thru_lstm[0][0]
        x_3 = x_3.unsqueeze(2)
        x_3 = torch.cat([x_3]*4, dim=2)
        # print("new x_3", x_3.shape)

        dx_3 = self.lrelu(self.decoder[0](x_4))
        # print("leaky relu input shape")
        # print("dx_3", dx_3.shape) # torch.Size([2, 256, 4, 32, 32]) -> same shape as x_3
        dx_3 = joinTensors(dx_3 , x_3 , type=self.joinType)
        # print("dx_3", dx_3.shape) #  torch.Size([2, 512, 4, 32, 32])

        dx_2 = self.lrelu(self.decoder[1](dx_3))
        # print("dx_2", dx_2.shape) # torch.Size([2, 128, 4, 64, 64]) -> same shape as x_2
        dx_2 = joinTensors(dx_2 , x_2 , type=self.joinType)
        # print("dx_2", dx_2.shape) # torch.Size([2, 256, 4, 64, 64])

        dx_1 = self.lrelu(self.decoder[2](dx_2))
        # print("dx_1", dx_1.shape) #  torch.Size([2, 64, 4, 128, 128]) -> same shape as x_1
        dx_1 = joinTensors(dx_1 , x_1 , type=self.joinType)
        # print("dx_1", dx_1.shape) # torch.Size([2, 128, 4, 128, 128])

        dx_0 = self.lrelu(self.decoder[3](dx_1)) 
        # print("dx_0", dx_0.shape) # torch.Size([2, 64, 4, 128, 128]) -> same shape as x_0
        dx_0 = joinTensors(dx_0 , x_0 , type=self.joinType)
        # print("dx_0", dx_0.shape) # torch.Size([2, 128, 4, 128, 128])

        dx_out = self.lrelu(self.decoder[4](dx_0))
        # print("dx_out", dx_out.shape)  # torch.Size([2, 64, 4, 256, 256])
        dx_out = torch.cat(torch.unbind(dx_out , 2) , 1)
        # print("dx_out", dx_out.shape) #  torch.Size([2, 256, 256, 256])

        out = self.lrelu(self.feature_fuse(dx_out))
        # print("out", out.shape) # torch.Size([2, 64, 256, 256])
        out = self.outconv(out)
        # print("out", out.shape) # torch.Size([2, 3, 256, 256])

        out = torch.split(out, dim=1, split_size_or_sections=3)
        mean_ = mean_.squeeze(2)
        out = [o+mean_ for o in out]
        # print("out list length", len(out)) # 1
        # print("out tensor", out[0].shape) # torch.Size([1, 3, 256, 256])
        return out

