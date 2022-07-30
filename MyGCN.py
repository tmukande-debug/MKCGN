import networkx as nx
import numpy
import torch as t
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
import math
import numpy as  np
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_dim=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  #[item_num,path_num,1]--mean(0)-->[path_num,1]
        beta = t.softmax(w,dim=0)    #[path_num,1] score
        beta = beta.expand((z.shape[0],)+beta.shape) #[item_num,path_num,1]
        return (beta*z).sum(1)       #[item_num,path_num,hidden_dim]--sum(1)-->[item_num,hidden_dim]

def message_func(edges):
    return {'m' : edges.src['n_f'] + edges.data['e_f']}

class MODEL(nn.Module):
    def __init__(self, user_graph_len,item_graph_len,args, userNum, itemNum, hide_dim, maxTime, interactionNum=5, layer=[16,16]):
        super(MODEL, self).__init__()
        self.user_graph_num=user_graph_len
        self.item_graph_num=item_graph_len
        self.act = t.nn.PReLU()
        self.userNum = userNum
        self.itemNum = itemNum
        self.hide_dim = hide_dim
        self.layer = [hide_dim] + layer
        self.embedding_dict = self.init_weight(userNum, itemNum*interactionNum, hide_dim,itemNum)
        self.in_size = numpy.sum(self.layer)
        self.userMetaLayers = nn.ModuleList()
        self.itemMetaLayers = nn.ModuleList()
        for _ in range(self.user_graph_num):
            userLayers = nn.ModuleList()
            for i in range(0, len(self.layer) - 1):
                userLayers.append(GraphConv(self.layer[i], self.layer[i + 1], bias=False, activation=self.act))
            self.userMetaLayers.append(userLayers)

        self.itemMetaLayers = nn.ModuleList()
        for _ in range(self.item_graph_num):
            itemLayers = nn.ModuleList()
            for i in range(0, len(self.layer) - 1):
                itemLayers.append(GraphConv(self.layer[i], self.layer[i + 1], bias=False, activation=self.act))
            self.itemMetaLayers.append(itemLayers)
        self.semanticItemAttention = SemanticAttention(self.in_size)
        self.semanticUserAttention = SemanticAttention(self.in_size)
        self.args = args
        slope = self.args.slope
        # GCN activation is leakyReLU
        self.act = t.nn.LeakyReLU(negative_slope=slope)
        if self.args.fuse == "weight":
            self.w = nn.Parameter(t.Tensor(itemNum, interactionNum, 1))
            init.xavier_uniform_(self.w)

        self.t_e = TimeEncoding(self.hide_dim, maxTime)
        self.layers = nn.ModuleList()
        for i in range(0, len(self.layer)-1):
                self.layers.append(GCNLayer(self.layer[i], self.layer[i+1], weight=True, bias=False, activation=self.act))
    
    def init_weight(self, userNum, itemNum, hide_dim,itemNum2):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(t.empty(userNum, hide_dim))),
            'item_emb': nn.Parameter(initializer(t.empty(itemNum, hide_dim))),
            'item_emb2': nn.Parameter(initializer(t.empty(itemNum2, hide_dim))),
        })
        return embedding_dict
    
    def forward(self, graph,usergraph,itemgraph, time_seq, out_dim, rClass=5, isTrain=True):
        norm=1
        self.semanticUserEmbeddings = []
        self.semanticItemEmbeddings = []
        all_user_embeddings = [self.embedding_dict['user_emb']]
        all_item_embeddings = [self.embedding_dict['item_emb']]
        if len(self.layers) == 0:
            item_embedding = self.embedding_dict['item_emb'].view(-1, rClass, out_dim)
            ret_item_embedding = t.div(t.sum(item_embedding, dim=1), rClass)
            return self.embedding_dict['user_emb'], ret_item_embedding
        edge_feat = self.t_e(time_seq)
        pathNum, blockNum = np.shape(self.userMetaLayers)
        for i in range(pathNum):
            self.all_user_embeddings = [self.embedding_dict['user_emb']]

            layers = self.userMetaLayers[i]
            for j in range(blockNum):
                layer = layers[j]
                if j == 0:
                    userEmbeddings = layer(usergraph[i], self.embedding_dict['user_emb'])
                else:
                    userEmbeddings = layer(usergraph[i], userEmbeddings)

                if norm == 1:
                    norm_embeddings = F.normalize(userEmbeddings, p=2, dim=1)
                    self.all_user_embeddings += [norm_embeddings]
                else:
                    self.all_user_embeddings += [userEmbeddings]
            self.userEmbedding = t.cat(self.all_user_embeddings, 1)  # 24 [8,8,8]
            self.semanticUserEmbeddings.append(self.userEmbedding)
        pathNum, blockNum = np.shape(self.itemMetaLayers)
        for i in range(pathNum):
            self.all_item_embeddings = [self.embedding_dict['item_emb2']]

            layers = self.itemMetaLayers[i]
            for j in range(blockNum):
                layer = layers[j]
                if j == 0:
                    itemEmbeddings = layer(itemgraph[i], self.embedding_dict['item_emb2'])
                else:
                    itemEmbeddings = layer(itemgraph[i], itemEmbeddings)

                if norm == 1:
                    norm_embeddings = F.normalize(itemEmbeddings, p=2, dim=1)
                    self.all_item_embeddings += [norm_embeddings]
                else:
                    self.all_item_embeddings += [itemEmbeddings]
            self.itemEmbedding = t.cat(self.all_item_embeddings, 1)
            self.semanticItemEmbeddings.append(self.itemEmbedding)

        for i, layer in enumerate(self.layers):
            if i == 0:
                embeddings = layer(graph, self.embedding_dict['user_emb'], self.embedding_dict['item_emb'], edge_feat)
            else:
                embeddings = layer(graph, embeddings[: self.userNum], embeddings[self.userNum:], edge_feat)

            norm_embeddings = F.normalize(embeddings, p=2, dim=1)
            all_user_embeddings += [norm_embeddings[: self.userNum]]
            all_item_embeddings += [norm_embeddings[self.userNum:]]

        user_embedding = t.cat(all_user_embeddings, 1)
        item_embedding = t.cat(all_item_embeddings, 1)

        if rClass == 1:
            return user_embedding, item_embedding
        item_embedding = item_embedding.view(-1, rClass, out_dim)
        if self.args.fuse == "mean":
            ret_item_embedding = t.div(t.sum(item_embedding, dim=1), rClass)
        elif self.args.fuse == "weight":
            weight = t.softmax(self.w, dim=1)
            item_embedding = item_embedding * weight
            ret_item_embedding = t.sum(item_embedding, dim=1)

        self.semanticUserEmbeddings.append(user_embedding)
        self.semanticItemEmbeddings.append(ret_item_embedding)
        self.semanticItemEmbeddings = t.stack(self.semanticItemEmbeddings, dim=1)
        self.semanticUserEmbeddings = t.stack(self.semanticUserEmbeddings, dim=1)
        ret_item_embedding = self.semanticItemAttention(self.semanticItemEmbeddings)
        user_embedding=self.semanticUserAttention(self.semanticUserEmbeddings)
        return user_embedding, ret_item_embedding


class GCNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 weight=True,
                 bias=False,
                 activation=None):
        super(GCNLayer, self).__init__()
        self.bias = bias
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.weight = weight
        if self.weight:
            self.u_w = nn.Parameter(t.Tensor(in_feats, out_feats))
            self.v_w = nn.Parameter(t.Tensor(in_feats, out_feats))
            # self.e_w = nn.Parameter(t.Tensor(in_feats, out_feats))
            init.xavier_uniform_(self.u_w)
            init.xavier_uniform_(self.v_w)
            # init.xavier_uniform_(self.e_w)
        self._activation = activation

    # def forward(self, graph, feat):
    def forward(self, graph, u_f, v_f, e_f):
        with graph.local_scope():
            if self.weight:
                u_f = t.mm(u_f, self.u_w)
                v_f = t.mm(v_f, self.v_w)
                # e_f = t.mm(e_f, self.e_w)
            node_f = t.cat([u_f, v_f], dim=0)
            # D^-1/2
            # degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            degs = graph.out_degrees().to(u_f.device).float().clamp(min=1)
            #norm = t.pow(degs, -0.5).view(-1, 1)
            norm = t.pow(degs, -0.5).view(-1, 1)
            # norm = norm.view(-1,1)
            # shp = norm.shape + (1,) * (feat.dim() - 1)
            # norm = t.reshape(norm, shp)

            node_f = node_f * norm

            graph.ndata['n_f'] = node_f
            graph.edata['e_f'] = e_f
            graph.update_all(message_func=message_func, reduce_func=fn.sum(msg='m', out='n_f'))

            rst = graph.ndata['n_f']

            degs = graph.in_degrees().to(u_f.device).float().clamp(min=1)
            norm = t.pow(degs, -0.5).view(-1, 1)
            # shp = norm.shape + (1,) * (feat.dim() - 1)
            # norm = t.reshape(norm, shp)
            rst = rst * norm

            if self._activation is not None:
                rst = self._activation(rst)

            return rst
            
class TimeEncoding(nn.Module):
    def __init__(self, n_hid, max_len = 240, dropout = 0.2):
        super(TimeEncoding, self).__init__()
        position = t.arange(0., max_len).unsqueeze(1)
        # ref self-attention
        div_term = 1 / (10000 ** (t.arange(0., n_hid * 2, 2.)) / n_hid / 2)
        
        # initializer = nn.init.xavier_uniform_
        # self.emb = nn.Parameter(initializer(t.empty(max_len, n_hid)))
        self.emb = nn.Embedding(max_len, n_hid * 2)
        self.emb.weight.data[:, 0::2] = t.sin(position * div_term) / math.sqrt(n_hid)
        self.emb.weight.data[:, 1::2] = t.cos(position * div_term) / math.sqrt(n_hid)
        self.emb.weight.requires_grad = False
        # 0 is useless, 1 is self->self
        self.emb.weight.data[0] = t.zeros_like(self.emb.weight.data[-1])
        self.emb.weight.data[1] = t.zeros_like(self.emb.weight.data[-1])
        self.lin = nn.Linear(n_hid * 2, n_hid)

    def forward(self, time):
        return self.lin(self.emb(time))
    #调用方法
    #1.先初始化 t_e=TimeEncoding(隐藏层)#这个隐藏层随便你自己定义，一般的话是和你原本模型的相同
    #2.t_embaddings=t_e(时间信息) #这个时间信息是tensor