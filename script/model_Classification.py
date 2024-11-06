import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.nn import MLP, PointTransformerConv, fps, global_mean_pool, knn, knn_graph

#Fig4a in Paper
class TransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear_in = nn.Linear(in_channels, in_channels) #위쪽 Linear in Paper_Fig4a
        self.linear_out = nn.Linear(out_channels, out_channels) #아래쪽 Linear in Paper_Fig4a
        
        #theta in Paper_Eqn(4)
        self.pos_nn = MLP([2, 64, out_channels], #single_hidden_FC_Layer <--- 2는 pos_dim
                          norm=None, #batch normalization 등의 정규화 방식 설정 (여기서는 안 함)
                          plain_last=False #만약 True라면 마지막 층에는 비선형활성화함수 및 드랍아웃/배치정규화 등을 전혀 적용 안 함 (여기서는 함) 
                          )
        
        #gamma in Paper_Eqn(3)
        self.attn_nn = MLP([out_channels, 64, out_channels], norm=None,
                           plain_last=False)

        self.transformer = PointTransformerConv(in_channels, #<--- x_dim
                                                out_channels, #<--- y_dim
                                                pos_nn=self.pos_nn, #theta in Paper_Eqn(4)
                                                attn_nn=self.attn_nn #gamma in Paper_Eqn(3)
                                                #Paper_Eqn(3)의 phi, psi, alpha는 단순한 linear_transform으로, source_code에 알아서 구현돼있다.
                                                )

    def forward(self, x, pos, edge_index):
        x = self.linear_in(x).relu()
        x = self.transformer(x, pos, edge_index)
        x = self.linear_out(x).relu()
        return x
    

#Fig4b in Paper
class TransitionDown(nn.Module):
    """Samples the input point cloud by a ratio percentage to reduce
    cardinality and uses an mlp to augment features dimensionnality.
    """
    def __init__(self, in_channels, out_channels, ratio=1/3, k=6): #1/4
        super().__init__()
        self.k = k
        self.ratio = ratio
        self.mlp = MLP([in_channels, out_channels], plain_last=False)

    def forward(self, x, pos, batch):
        # FPS sampling
        id_clusters = fps(pos, ratio=self.ratio, batch=batch)
        # compute for each cluster the k nearest points
        sub_batch = batch[id_clusters] if batch is not None else None

        # beware of self loop
        id_k_neighbor = knn(pos, pos[id_clusters], k=self.k, batch_x=batch,
                            batch_y=sub_batch)

        # transformation of features through a simple MLP
        x = self.mlp(x)

        # Max pool onto each cluster the features from knn in points
        x_out = scatter(x[id_k_neighbor[1]], id_k_neighbor[0], dim=0,
                        dim_size=id_clusters.size(0), reduce='max')

        # keep only the clusters and their max-pooled features
        sub_pos, out = pos[id_clusters], x_out
        return out, sub_pos, sub_batch


class Model_Classification(nn.Module):
    def __init__(self, in_channels, out_channels, dim_model, k=16):
        #in_channels : x_dim [int] (ex. in_channels=0 for ModelNet10_Classification_Task)
        #out_channels : D_out in Paper_Fig3 [int] (ex. out_channels=10 for ModelNet10_Classification_Task)
        #dim_models : the dimensions of each hidden layer [the list of integers]
        #               ---> For example, according to Paper_Fig3, dim_models will be [32, 64, 128, 256, 512]
        #k : the value of K for KNN
        super().__init__()
        self.k = k

        # dummy feature is created if there is none given
        # 만약 정말 classification만 할 것이라면 position_vector만 있어도 충분 --> 굳이 x_vector가 필요없음
        # 즉, 일단 in_channels=0 및 (밑의 forward에서의) x인수를 None으로 두면 알아서 아무 의미없는 값 (one_vector)을 할당한다.
        in_channels = max(in_channels, 1)

        # first block
        self.mlp_input = MLP([in_channels, dim_model[0]], plain_last=False)

        self.transformer_input = TransformerBlock(in_channels=dim_model[0],
                                                  out_channels=dim_model[0])
        # backbone layers
        self.transformers_down = nn.ModuleList()
        self.transition_down = nn.ModuleList()

        for i in range(len(dim_model) - 1):
            # Add Transition Down block followed by a Transformer block
            self.transition_down.append(
                TransitionDown(in_channels=dim_model[i],
                               out_channels=dim_model[i + 1], k=self.k))

            self.transformers_down.append(
                TransformerBlock(in_channels=dim_model[i + 1],
                                 out_channels=dim_model[i + 1]))

        # class score computation <--> from (1,512) to (1,D_out) in Paper_Fig3
        self.mlp_output = MLP([dim_model[-1], 64, out_channels], norm=None)

    def forward(self, x, pos, batch=None):

        # add dummy features in case there is none
        if x is None:
            x = torch.ones((pos.shape[0], 1), device=pos.get_device())

        # first block
        x = self.mlp_input(x)
        edge_index = knn_graph(pos, k=self.k, batch=batch)
        x = self.transformer_input(x, pos, edge_index)

        # backbone
        for i in range(len(self.transformers_down)):
            x, pos, batch = self.transition_down[i](x, pos, batch=batch)

            edge_index = knn_graph(pos, k=self.k, batch=batch)
            x = self.transformers_down[i](x, pos, edge_index)

        # GlobalAveragePooling
        x = global_mean_pool(x, batch)
        
        # Class score
        out = self.mlp_output(x)
        
        return F.log_softmax(out, dim=-1)