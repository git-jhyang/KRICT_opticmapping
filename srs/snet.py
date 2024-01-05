from .base_networks import DenseBlock, ConvBlock, DeconvBlock, NetBlock
from torch import nn
import torch

class SpectrumSuperResolutionNetwork(nn.Module):
    def __init__(self, num_channels, base_filter, feat, num_stages, residual=True):
        super(SpectrumSuperResolutionNetwork, self).__init__()
        
        self.num_stages = num_stages
        self.residual = residual
        
        self.embed0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.embed1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)
        
        self.block1 = NetBlock(base_filter, 4, 2, 2, 1)
        self.block2 = NetBlock(base_filter, 8, 4, 4, 2)
        self.block3 = NetBlock(base_filter, 16, 8, 8, 3)
        self.block4 = NetBlock(base_filter, 32, 16, 16, 4)
        self.block5 = NetBlock(base_filter, 64, 32, 32, 5)
        self.block6 = NetBlock(base_filter, 128, 64, 64, 6)
        
        self.output_conv = ConvBlock(base_filter * self.num_stages, 1, 3, 1, 1, activation=None, norm=None)
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv1d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose1d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        h = self.embed0(x)
        h = self.embed1(h)
        
        results = []
        for _ in range(self.num_stages):
            h0 = self.block1(h)
            cat_h = torch.concat([h0, h], dim=1)
            h = self.block2(cat_h)
            cat_h = torch.concat([h, cat_h], dim=1)
            h = self.block3(cat_h)
            cat_h = torch.concat([h, cat_h], dim=1)
            h = self.block4(cat_h)
            cat_h = torch.concat([h, cat_h], dim=1)
            h = self.block5(cat_h)
            cat_h = torch.concat([h, cat_h], dim=1)
            h = self.block6(cat_h)
            
            results.append(h)
        
        result = torch.cat(results, 1)
        out = self.output_conv(result)
        if self.residual:
            out = out + x[:,:1]
        return out

class SpectrumAutoEncoder(nn.Module):
    def __init__(self, num_channels, base_filter, feat, input_length, num_blocks, 
                 latent_dim=32, kernel_size=4, stride=None, padding=None):
        super(SpectrumAutoEncoder, self).__init__()
        self.embed0 = ConvBlock(num_channels, feat, 3, 1, 1, activation='prelu', norm=None)
        self.embed1 = ConvBlock(feat, base_filter, 1, 1, 0, activation='prelu', norm=None)

        if stride is None: stride = kernel_size // 2
        if padding is None: padding = stride // 2

        self.conv_layers = nn.ModuleList([])
        length = input_length
        for i in range(num_blocks):
            _length = (length + 2 * padding - (kernel_size - 1) - 1)/stride + 1
            if _length < 1:
                print(f"Warning: too many blocks for given kernel size (length below 1) - truncate blocks from {num_blocks} to {i}")
                num_blocks = i
                break
            length = int(_length)
            self.conv_layers.append(ConvBlock(base_filter, base_filter, kernel_size, stride, padding, norm='batch'))
        
        self.latent_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(length * base_filter, latent_dim)
        )

        self.deconv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, length * base_filter),
                nn.Unflatten(-1, (base_filter, length))
            )
        ])
        for _ in range(num_blocks):
            self.deconv_layers.append(DeconvBlock(base_filter, base_filter, kernel_size, stride, padding, norm='batch'))

        self.output_layer = ConvBlock(base_filter, 1, 3, 1, 1)

    def forward(self, x):
        h = self.embed0(x)
        h = self.embed1(h)
        for layer in self.conv_layers:
            h = layer(h)
        latent = self.latent_layer(h)
        h = latent
        for layer in self.deconv_layers:
            h = layer(h)
        out = self.output_layer(h)
        return out, latent