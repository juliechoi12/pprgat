from typing import Literal

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_sparse import SparseTensor

from layers.gat_conv_ppr import GATConvPPR
from layers.gatv2_conv_ppr import GATv2ConvPPR


class NetPPRGAT(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=2,
        heads=3,
        # conv: Literal["GATConvPPR", "GATv2ConvPPR"] = "GATConvPPR",
        # edge_index=None,
        dropout=0.6,
        v2=False,
    ):
        super().__init__()
        self.lin1 = Linear(in_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        Conv = GATConvPPR if not v2 else GATv2ConvPPR
        self.prop = Conv(
            hidden_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            # edge_index=edge_index,
        )
        # self.ppr_matrix = ppr_matrix
        self.dropout = dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, ppr_vals, return_attention_weights=True):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        if isinstance(edge_index, SparseTensor):
            x, edge_index1 = self.prop(
                x,
                edge_index,
                ppr_vals=ppr_vals,
                return_attention_weights=return_attention_weights,
            )
            return F.log_softmax(x, dim=-1), edge_index1, edge_index1
        else:
            x, (edge_index1, alpha1) = self.prop(
                x,
                edge_index,
                ppr_vals=ppr_vals,
                return_attention_weights=return_attention_weights,
            )
            return (
                F.log_softmax(x, dim=-1),
                (edge_index1, alpha1),
                (edge_index1, alpha1),
            )


class NetPPRGATDouble(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=2,
        heads=3,
        conv: Literal["GATConvPPR", "GATv2ConvPPR"] = "GATConvPPR",
        # edge_index=None,
        dropout=0.6,
        v2=False,
    ):
        super().__init__()
        self.dropout = dropout
        Conv = GATConvPPR if not v2 else GATv2ConvPPR
        self.conv1 = Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            # edge_index=edge_index,
        )
        self.conv2 = Conv(
            in_channels=heads * hidden_channels,
            out_channels=out_channels,
            heads=heads,
            concat=False,
            dropout=dropout,
            # edge_index=edge_index,
        )
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, return_attention_weights=True, **kwargs):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, (edge_index1, alpha1) = self.conv1(
            x, edge_index, return_attention_weights=return_attention_weights, **kwargs
        )
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, (edge_index2, alpha2) = self.conv2(
            x, edge_index, return_attention_weights=return_attention_weights, **kwargs
        )
        return F.log_softmax(x, dim=-1), (edge_index1, alpha1), (edge_index2, alpha2)
