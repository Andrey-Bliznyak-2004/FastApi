import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_max_pool

class DGCNN_seg(nn.Module):
    def __init__(self, in_channels=3, out_channels=4, k=16): 
        super().__init__()
        self.k = k

        # DGCNN слои
        self.conv1 = EdgeConv(nn=nn.Sequential(
            nn.Linear(2*in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        ), aggr='max')
        #
        self.conv2 = EdgeConv(nn=nn.Sequential(
            nn.Linear(2*64, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        ), aggr='max')

        self.conv3 = EdgeConv(nn=nn.Sequential(
            nn.Linear(2*128, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        ), aggr='max')

        # Объединение признаков
        total_features = 64 + 128 + 256

        # Классификационная голова (для каждой точки)
        self.mlp = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, out_channels)
        )

    # Функция для построения k-NN графа
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h1 = self.conv1(x, edge_index)
        h2 = self.conv2(h1, edge_index)
        h3 = self.conv3(h2, edge_index)

        # Конкатенируем все промежуточные представления
        h = torch.cat([h1, h2, h3], dim=1) 

        out = self.mlp(h)                 
        return out