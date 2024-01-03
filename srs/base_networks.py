from torch import nn

class DenseBlock(nn.Module):
    def __init__(self, input_size, output_size, bias=True, activation='relu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.fc = nn.Linear(input_size, output_size, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class ConvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, 
                 bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class DeconvBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, 
                 bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = nn.ConvTranspose1d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        
        self.norm = norm
        if self.norm == 'batch':
            self.bn = nn.BatchNorm1d(output_size)
        elif self.norm == 'instance':
            self.bn = nn.InstanceNorm1d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class ZipBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=4, num_stages=1, bias=True, activation='prelu', norm=None):
        super(ZipBlock, self).__init__()
        self.zip_embed = ConvBlock(num_filter*num_stages, num_filter, 3, 1, 1, bias, activation, norm)
        self.zip_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm)
        self.zip_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm)
        self.zip_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm)        

    def forward(self, x):
        x = self.zip_embed(x)
        z0 = self.zip_conv1(x)
        u0 = self.zip_conv2(z0)
        z1 = self.zip_conv3(u0 - x)
        return z1 + z0

class UnzipBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=4, num_stages=1, bias=True, activation='prelu', norm=None):
        super(UnzipBlock, self).__init__()
        self.unzip_embed = ConvBlock(num_filter*num_stages, num_filter, 3, 1, 1, bias, activation, norm)
        self.unzip_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm)
        self.unzip_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm)
        self.unzip_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm)        

    def forward(self, x):    
        x = self.unzip_embed(x)
        u0 = self.unzip_conv1(x)
        z0 = self.unzip_conv2(u0)
        u1 = self.unzip_conv3(z0 - x)
        return u1 + u0

class NetBlock(nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=4, num_stages=1, bias=True, activation='prelu', norm=None):
        super(NetBlock, self).__init__()
        self.block_embed = ConvBlock(num_filter*num_stages, num_filter, 3, 1, 1, bias, activation, norm)
        self.block_conv  = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm)
        self.block_dconv = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias, activation, norm)

    def forward(self, x):
        x = self.block_embed(x)
        c = self.block_conv(x)
        d = self.block_dconv(c)
        return d + x