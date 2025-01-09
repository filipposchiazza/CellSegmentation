import torch
import torch.nn as nn
import torch.nn.functional as F



class ResidualBlock(nn.Module):

    def __init__(self, 
                 in_channels, 
                 out_channels,
                 num_groups=8,
                 activation_fn=F.silu):
        """Residual block with group normalization.
        
        Parameters:
        -----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        num_groups : int
            Number of groups for group normalization. Default is 8.
        activation_fn : torch activation function
            Activation function to use. Default is F.silu.
        """
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups
        self.activation_fn = activation_fn

        self.group_norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, 
                               out_channels, 
                               kernel_size=3, 
                               padding=1)
        
        self.group_norm2 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, 
                               out_channels, 
                               kernel_size=3, 
                               padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
            
        else:
            self.residual_layer = nn.Conv2d(in_channels, 
                                            out_channels, 
                                            kernel_size=1, 
                                            padding = 0)
            
    
    def forward(self,x):
        
        res = self.residual_layer(x)
        
        x = self.group_norm1(x)
        x = self.activation_fn(x)
        x = self.conv1(x)
        
        x = self.group_norm2(x)
        x = self.activation_fn(x)
        x = self.conv2(x)
        
        return x + res
    


class DownSample(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        """Downsampling layer.
        
        Parameters:
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        """
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels, 
                              kernel_size=3,
                              stride=2,
                              padding=1)
        
    def forward(self, inputs):
        return self.conv(inputs)



class UpSample(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        """Upsampling layer.
        
        Parameters:
        ----------
        in_channels: int
            Number of input channels
        out_channels: int
            Number of output channels
        """
        super(UpSample, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels,
                                         out_channels=out_channels,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1,
                                         output_padding=1)
        
    def forward(self, x):
        return self.deconv(x)




