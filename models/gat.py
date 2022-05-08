import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv


class NetGATSimple(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=8,
        heads=4,
        heads2=1,
        dropout=0.6,
        v2=False,
    ):
        super().__init__()
        Conv = GATConv if not v2 else GATv2Conv

        self.conv1 = Conv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = Conv(
            hidden_channels * heads,
            out_channels,
            heads=heads2,
            concat=False,
            dropout=dropout,
        )
        self.dropout = dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, x, edge_index, return_attention_weights=True):
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(
            x,
            p=self.dropout,
            training=self.training,
        )
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)
