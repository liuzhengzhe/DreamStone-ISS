
import torch.nn as nn
import torch.nn.functional as F
from im2mesh.layers import ResnetBlockFC
import torch


class generator(nn.Module):
	def __init__(self,  gf_dim):
		super(generator, self).__init__()
		self.gf_dim = gf_dim
		self.linear_1 = nn.Linear(512, self.gf_dim*8, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_7 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_8 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_9 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
		self.linear_10 = nn.Linear(self.gf_dim*4, self.gf_dim*4, bias=True)
		self.linear_11 = nn.Linear(self.gf_dim*4, self.gf_dim*4, bias=True)
		self.linear_12 = nn.Linear(self.gf_dim*4, 256,  bias=True)   
		nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_2.bias,0)
		nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_3.bias,0)
		nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_4.bias,0)
		nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_5.bias,0)
		nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_6.bias,0)

		nn.init.normal_(self.linear_7.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_7.bias,0)
		nn.init.normal_(self.linear_8.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_8.bias,0)
		nn.init.normal_(self.linear_9.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_9.bias,0)
		nn.init.normal_(self.linear_10.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_10.bias,0)
		nn.init.normal_(self.linear_11.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_11.bias,0)
		nn.init.normal_(self.linear_12.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_12.bias,0)
   
   
	def forward(self, clip_feature, is_training=False):

		l1 = self.linear_1(clip_feature)
		l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

		l4 = self.linear_4(l3)
		l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

		l5 = self.linear_5(l4)
		l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

		l6 = self.linear_6(l5)
		l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

		l7 = self.linear_7(l6)
		l7 = F.leaky_relu(l7, negative_slope=0.02, inplace=True)

		l8 = self.linear_8(l7)
		l8 = F.leaky_relu(l8, negative_slope=0.02, inplace=True)

		l9 = self.linear_9(l8)
		l9 = F.leaky_relu(l9, negative_slope=0.02, inplace=True)

		l10 = self.linear_10(l9)
		l10 = F.leaky_relu(l10, negative_slope=0.02, inplace=True)

		l11 = self.linear_11(l10)
		l11 = F.leaky_relu(l11, negative_slope=0.02, inplace=True)

		l12 = self.linear_12(l11)

		return l12


class Decoder(nn.Module):
    ''' Decoder class.

    As discussed in the paper, we implement the OccupancyNetwork
    f and TextureField t in a single network. It consists of 5
    fully-connected ResNet blocks with ReLU activation.

    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of ResNet blocks
        out_dim (int): output dimension (e.g. 1 for only
            occupancy prediction or 4 for occupancy and
            RGB prediction)
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=512, leaky=False, n_blocks=5, out_dim=4):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.out_dim = out_dim

        # Submodules

        #self.fc_pre = generator(64) #nn.Linear(512, 256)
        self.fc_p = nn.Linear(dim, hidden_size)
        self.fc_out = nn.Linear(hidden_size, out_dim)

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

    def forward(self, p, c=None, batchwise=True, only_occupancy=False,
                only_texture=False, **kwargs):

        assert((len(p.shape) == 3) or (len(p.shape) == 2))

        #c=self.fc_pre(c)
        net = self.fc_p(p)
        #print (p, net, 'p net')
        for n in range(self.n_blocks):
            if self.c_dim != 0 and c is not None:
                net_c = self.fc_c[n](c)
                if batchwise:
                    net_c = net_c.unsqueeze(1)
                net = net + net_c

            net = self.blocks[n](net)

        out = self.fc_out(self.actvn(net))
        #print (out, 'out')

        if only_occupancy:
            if len(p.shape) == 3:
                out = out[:, :, 0]
            elif len(p.shape) == 2:
                out = out[:, 0]
        elif only_texture:
            if len(p.shape) == 3:
                out = out[:, :, 1:4]
            elif len(p.shape) == 2:
                out = out[:, 1:4]

        out = out.squeeze(-1)
        #print ('decoder', torch.unique(out))
        return out
