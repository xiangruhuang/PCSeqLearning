import sys
from torch import nn
import torch
from torch_cluster import grid_cluster
import numpy as np
from torch_scatter import scatter

from .kpconv_layers import KPConvLayer
from pcdet.models.model_utils.grid_sampling import GridSampling3D
from .basic_blocks import MLP

class BaseModule(nn.Module):

    @property
    def num_params(self):
        """This property is used to return the number of trainable parameters for a given layer
           It is useful for debugging and reproducibility.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self._num_params = sum([np.prod(p.size()) for p in model_parameters])
        return self._num_params


class SimpleBlock(BaseModule):
    """
    simple layer with KPConv convolution -> activation -> BN
    we can perform a stride version (just change the query and the neighbors)
    """

    CONV_TYPE = "partial_dense" 
    RIGID_DENSITY = 2.5

    def __init__(
        self,
        down_conv_nn=None,
        grid_size=None,
        prev_grid_size=None,
        sigma=1.0,
        max_num_neighbors=16,
        num_kernel_points=15,
        num_act_kernel_points=15,
        activation=nn.LeakyReLU(negative_slope=0.1),
        bn_momentum=0.02,
        bn=nn.BatchNorm1d,
        deformable=False,
        add_one=False,
        has_bottleneck=None,
        neighbor_finder=None,
        **kwargs,
    ):
        super(SimpleBlock, self).__init__()
        assert len(down_conv_nn) == 2
        num_inputs, num_outputs = down_conv_nn
        density_parameter = self.RIGID_DENSITY
        self.kp_conv = KPConvLayer(
            num_inputs, num_outputs,
            kernel_influence_dist=prev_grid_size * sigma,
            num_kernel_points=num_kernel_points,
            num_act_kernel_points=num_act_kernel_points,
            add_one=add_one, **kwargs
        )
        search_radius = density_parameter * sigma * prev_grid_size
        self.radius = search_radius
        self.num_neighbors = max_num_neighbors
        self.neighbor_finder = neighbor_finder

        if bn:
            self.bn = bn(num_outputs, momentum=bn_momentum)
        else:
            self.bn = None
        self.activation = activation

        is_strided = prev_grid_size != grid_size
        if is_strided:
            self.sampler = GridSampling3D(grid_size)
        else:
            self.sampler = None

    def forward(self, pos, x, vis_dict=None, **kwargs):
        """
        Args:
            pos [N, 4] first dimension is batch index
            x [N, D] point-wise features
            vis_dict for visualization only
            **kwargs ignored args

        Returns:
            query_pos
            x_out
            edge_indices
        """

        if self.sampler:
            query_pos = self.sampler(pos)
            if vis_dict is not None:
                vis_dict['pos'].append(query_pos)
        else:
            query_pos = pos
        
        if 'edge_indices' in kwargs:
            kwargs.pop('edge_indices')
        edge_indices = self.neighbor_finder(
                           pos, query_pos,
                           self.radius, self.num_neighbors
                       )

        x = self.kp_conv(pos[:, 1:], query_pos[:, 1:],
                         edge_indices, x)
        if self.bn:
            x = self.bn(x)
        x = self.activation(x)

        return dict(
            pos=query_pos,
            x=x, 
            edge_indices=edge_indices,
            vis_dict=vis_dict,
            **kwargs
        )

    def extra_repr(self):
        return "Num parameters: {}; {}; {}".format(self.num_params, self.sampler,
                                                   self.neighbor_finder)


class ResnetBBlock(BaseModule):
    """ Resnet block with optional bottleneck activated by default
    Arguments:
        down_conv_nn (len of 2 or 3) :
                        sizes of input, intermediate, output.
                        If length == 2 then intermediate =  num_outputs // 4
        radius : radius of the conv kernel
        sigma :
        density_parameter : density parameter for the kernel
        max_num_neighbors : maximum number of neighboors for the neighboor search
        activation : activation function
        has_bottleneck: wether to use the bottleneck or not
        bn_momentum
        bn : batch norm (can be None -> no batch norm)
        grid_size : size of the grid,
        prev_grid_size : size of the grid at previous step.
                        In case of a strided block, this is different than grid_size
    """

    CONV_TYPE = "partial_dense" 

    def __init__(
        self,
        down_conv_nn=None,
        grid_size=None,
        prev_grid_size=None,
        sigma=1,
        max_num_neighbors=16,
        num_kernel_points=15,
        num_act_kernel_points=15,
        activation=torch.nn.LeakyReLU(negative_slope=0.1),
        has_bottleneck=True,
        bn_momentum=0.02,
        bn=nn.BatchNorm1d,
        deformable=False,
        add_one=False,
        neighbor_finder=None,
        **kwargs,
    ):
        super(ResnetBBlock, self).__init__()
        assert len(down_conv_nn) == 2 or len(down_conv_nn) == 3, "down_conv_nn should be of size 2 or 3"
        if len(down_conv_nn) == 2:
            num_inputs, num_outputs = down_conv_nn
            d_2 = num_outputs // 4
        else:
            num_inputs, d_2, num_outputs = down_conv_nn
        self.is_strided = prev_grid_size != grid_size
        self.has_bottleneck = has_bottleneck

        # Main branch
        if self.has_bottleneck:
            kp_size = [d_2, d_2]
        else:
            kp_size = [num_inputs, num_outputs]

        self.simple_block = SimpleBlock(
            down_conv_nn=kp_size,
            grid_size=grid_size,
            prev_grid_size=prev_grid_size,
            sigma=sigma,
            num_kernel_points=num_kernel_points,
            num_act_kernel_points=num_act_kernel_points,
            max_num_neighbors=max_num_neighbors,
            activation=activation,
            bn_momentum=bn_momentum,
            bn=bn,
            deformable=deformable,
            add_one=add_one,
            neighbor_finder=neighbor_finder,
            **kwargs,
        )

        if self.has_bottleneck:
            if bn:
                self.unary_1 = nn.Sequential(
                    nn.Linear(num_inputs, d_2, bias=False), bn(d_2, momentum=bn_momentum), activation
                )
                self.unary_2 = nn.Sequential(
                    nn.Linear(d_2, num_outputs, bias=False), bn(num_outputs, momentum=bn_momentum), activation
                )
            else:
                self.unary_1 = nn.Sequential(nn.Linear(num_inputs, d_2, bias=False), activation)
                self.unary_2 = nn.Sequential(nn.Linear(d_2, num_outputs, bias=False), activation)

        # Shortcut
        if num_inputs != num_outputs:
            if bn:
                self.shortcut_op = nn.Sequential(
                    nn.Linear(num_inputs, num_outputs, bias=False),
                    bn(num_outputs, momentum=bn_momentum)
                )
            else:
                self.shortcut_op = nn.Linear(num_inputs, num_outputs, bias=False)
        else:
            self.shortcut_op = nn.Identity()

        # Final activation
        self.activation = activation

    def forward(self, pos, x, **kwargs):
        """
            data: x, pos, batch_idx and idx_neighbor when the neighboors of each point in pos have already been computed
        """
        # Main branch_dict
        x_skip = x
        if self.has_bottleneck:
            x = self.unary_1(x)
        batch_dict = self.simple_block(pos=pos, x=x, **kwargs)
        new_batch_dict = {}; new_batch_dict.update(batch_dict)
        query_pos = new_batch_dict['pos']
        x = new_batch_dict['x']
        edge_indices = new_batch_dict['edge_indices']
        if self.has_bottleneck:
            x = self.unary_2(x)

        # Shortcut
        if self.is_strided:
            e_ref, e_query = edge_indices 
            #x_skip = torch.cat([x_skip, torch.zeros_like(x_skip[:1, :])], axis=0)  # Shadow feature
            edge_features = x_skip[e_ref]
            x_skip = scatter(edge_features, e_query, dim=0,
                             dim_size=query_pos.shape[0], reduce='max')

        x_skip = self.shortcut_op(x_skip)
        x += x_skip

        new_batch_dict.update(
            dict(
                 pos=query_pos,
                 x=x,
                 edge_indices=edge_indices
            )
        )
        return new_batch_dict

    @property
    def sampler(self):
        return self.kp_conv.sampler

    @property
    def neighbor_finder(self):
        return self.kp_conv.neighbor_finder

    def extra_repr(self):
        return "Num parameters: %i" % self.num_params


class KPDualBlock(BaseModule):
    """ Dual KPConv block (usually strided + non strided)

    Arguments: Accepted kwargs
        block_names: Name of the blocks to be used as part of this dual block
        down_conv_nn: Size of the convs e.g. [64,128],
        grid_size: Size of the grid for each block,
        prev_grid_size: Size of the grid in the previous KPConv
        has_bottleneck: Wether a block should implement the bottleneck
        max_num_neighbors: Max number of neighboors for the radius search,
        deformable: Is deformable,
        add_one: Add one as a feature,
    """

    def __init__(
        self,
        block_names=None,
        down_conv_nn=None,
        grid_size=None,
        prev_grid_size=None,
        has_bottleneck=None,
        max_num_neighbors=None,
        deformable=False,
        add_one=False,
        neighbor_finder=None,
        **kwargs,
    ):
        super(KPDualBlock, self).__init__()

        assert len(block_names) == len(down_conv_nn)
        self.blocks = torch.nn.ModuleList()
        for i, class_name in enumerate(block_names):
            # Constructing extra keyword arguments
            block_kwargs = {}
            for key, arg in kwargs.items():
                block_kwargs[key] = arg[i] if isinstance(arg, list) else arg

            # Building the block
            kpcls = getattr(sys.modules[__name__], class_name)
            block = kpcls(
                down_conv_nn=down_conv_nn[i],
                grid_size=grid_size[i],
                prev_grid_size=prev_grid_size[i],
                max_num_neighbors=max_num_neighbors[i],
                deformable=deformable[i] if isinstance(deformable, list) else deformable,
                add_one=add_one[i] if isinstance(add_one, list) else add_one,
                has_bottleneck=has_bottleneck[i],
                neighbor_finder=neighbor_finder,
                **block_kwargs,
            )
            self.blocks.append(block)

    def forward(self, data_dict, **kwargs):
        for block in self.blocks:
            data_dict = block(**data_dict)
        return data_dict

    @property
    def sampler(self):
        return [b.sampler for b in self.blocks]

    @property
    def neighbor_finder(self):
        return [b.neighbor_finder for b in self.blocks]

    def extra_repr(self):
        return "Num parameters: %i" % self.num_params


class FPBlockUp(BaseModule):
    def __init__(self, up_conv_nn, neighbor_finder, up_k, grid_size, **kwargs):
        super().__init__()
        self.neighbor_finder = neighbor_finder
        bn_momentum = kwargs.get("bn_momentum", 0.1)
        self.up_k = up_k
        self.nn = MLP(up_conv_nn, bn_momentum=bn_momentum, bias=False)
        self.radius = grid_size*np.sqrt(3)

    def forward(self, batch_dict, batch_dict_skip):
        pos = batch_dict['pos']
        x = batch_dict['x']
        new_batch_dict = {}; new_batch_dict.update(batch_dict)

        query_pos = batch_dict_skip['pos']
        x_skip = batch_dict_skip['x']

        edge_indices = self.neighbor_finder(
                           pos, query_pos,
                           self.radius, 1,
                           sort_by_dist=True)
        e_ref, e_query = edge_indices

        assert e_query.shape[0] == query_pos.shape[0]
        assert (e_query == torch.arange(query_pos.shape[0]).to(e_query)).all()
        
        x = torch.cat([x[e_ref], x_skip], dim=-1)
        if self.nn:
            x = self.nn(x)

        new_batch_dict.update(
            dict(
                pos=query_pos,
                x=x,
            )
        )

        return new_batch_dict
