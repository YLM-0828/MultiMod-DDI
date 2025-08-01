import dgl
import torch
import torch.nn as nn
from dgl import mean_nodes
from transformers import BertModel, BertPreTrainedModel, RobertaModel, AlbertModel
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import numpy as np
import math

from Informer_block import ProbAttention, FullAttention, AttentionLayer
from MultiFocalLoss import MultiFocalLoss
from rgcn_model import RGCN
from ddi_task.initialization_utils import initialize_model
from tool import TextGuidedGate, CrossGraphAttention

torch.autograd.set_detect_anomaly(True)

PRETRAINED_MODEL_MAP = {
    'biobert': BertModel,
    'scibert': BertModel,
    'roberta': RobertaModel,
    'albert': AlbertModel
}



class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, args, N_fingerprints, dim, layer_hidden, layer_output, mode, activation):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.args = args
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_hidden)])
        if layer_output != 0:
            self.W_output = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                           for _ in range(layer_output)])
            self.W_output_ = nn.ModuleList([nn.Linear(dim, dim)
                                            for _ in range(layer_output)])
        self.layer_hidden = layer_hidden
        self.layer_output = layer_output
        self.mode = mode
        activations = {'relu':nn.ReLU(), 'elu':nn.ELU(), 'leakyrelu':nn.LeakyReLU(), 'prelu':nn.PReLU(),
                       'relu6':nn.ReLU6, 'rrelu':nn.RReLU(), 'selu':nn.SELU(), 'celu':nn.CELU(), 'gelu':GELU()}
        self.activation = activations[activation]

    def pad(self, matrices):
        """Pad the list of matrices
        with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C],
        we obtain a new matrix [A00, 0B0, 00C],
        where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        zeros = torch.FloatTensor(np.zeros((M, N))).to(device)

        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            pad_matrices = zeros
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n

        return pad_matrices

    def update(self, matrix, vectors, layer):
        hidden_vectors = self.activation(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.matmul(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def gnn(self, inputs):

        """Cat or pad each input data for batch processing."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fingerprints = [torch.tensor(x, dtype=torch.long).to(device) for x in inputs[:,0,]]
        adjacencies = [torch.tensor(x, dtype=torch.long).to(device) for x in inputs[:,1,]]
        molecular_sizes = [torch.tensor(x, dtype=torch.long).to(device) for x in inputs[:,2,]]
        masks = [torch.tensor(x, dtype=torch.float).to(device) for x in inputs[:,3,]]
        masks = torch.cat(masks).unsqueeze(-1)


        #fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints)
        adjacencies = self.pad(adjacencies)

        """GNN layer (update the fingerprint_file vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for l in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, l)
            fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.

        """Molecular vector by sum or mean of the fingerprint_file vectors."""
        if self.mode == 'sum':
            molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        elif self.mode == 'mean':
            molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        if self.layer_output != 0:
            for l in self.W_output_:
                molecular_vectors = self.activation(l(molecular_vectors))

        """Mask invalid SMILES vectors"""
        molecular_vectors *= masks

        return molecular_vectors

    def mlp(self, vectors1, vectors2):
        vectors = torch.cat((vectors1, vectors2), 1)
        if self.layer_output != 0:
            for l in self.W_output:
                vectors = torch.relu(l(vectors))
        return vectors


# class MolecularGraphNeuralNetwork(nn.Module):
#     def __init__(self, args, N_fingerprints, dim, layer_hidden, layer_output, mode, activation, attention_type='prob'):
#         super(MolecularGraphNeuralNetwork, self).__init__()
#         self.args = args
#         self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
#         self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_hidden)])
#         if layer_output != 0:
#             self.W_output = nn.ModuleList([nn.Linear(2 * dim, 2 * dim) for _ in range(layer_output)])
#             self.W_output_ = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_output)])
#         self.layer_hidden = layer_hidden
#         self.layer_output = layer_output
#         self.mode = mode
#         activations = {'relu': nn.ReLU(inplace=False), 'elu': nn.ELU(), 'leakyrelu': nn.LeakyReLU(),
#                        'prelu': nn.PReLU(),
#                        'relu6': nn.ReLU6(), 'rrelu': nn.RReLU(), 'selu': nn.SELU(), 'celu': nn.CELU(),
#                        'gelu': nn.GELU()}
#         self.activation = activations[activation]
#
#         # 初始化注意力层
#         if attention_type == 'prob':
#             attention = ProbAttention(mask_flag=True, factor=5, scale=None, attention_dropout=0.1)
#         elif attention_type == 'full':
#             attention = FullAttention(mask_flag=True, factor=5, scale=None, attention_dropout=0.1)
#         else:
#             raise ValueError("Invalid attention type")
#         self.attention_layer = AttentionLayer(attention, dim, n_heads=1)
#
#     def pad(self, matrices):
#         """Pad the list of matrices with a pad_value (e.g., 0) for batch processing."""
#         shapes = [m.shape for m in matrices]
#         M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         zeros = torch.FloatTensor(np.zeros((M, N))).to(device)
#
#         i, j = 0, 0
#         for k, matrix in enumerate(matrices):
#             pad_matrices = zeros
#             m, n = shapes[k]
#             pad_matrices[i:i + m, j:j + n] = matrix
#             i += m
#             j += n
#
#         return pad_matrices
#
#     def update(self, matrix, vectors, layer):
#         # 确保 vectors 形状是 (B * L, D)
#         B = 1
#         L = vectors.shape[0]
#         D = vectors.shape[-1]
#
#         #print(f"vectors shape before reshape: {vectors.shape}")  # 调试信息
#
#         # 将 vectors 调整为 (B, L, D)
#         vectors = vectors.view(B, L, D)
#
#         #print(f"vectors shape after reshape: {vectors.shape}")  # 调试信息
#
#         # 线性变换和激活函数
#         hidden_vectors = self.activation(self.W_fingerprint[layer](vectors))
#
#         #print(f"hidden_vectors shape: {hidden_vectors.shape}")  # 调试信息
#
#         B, L, D = hidden_vectors.shape  # 现在应该不会报错
#
#         attn_mask = None  # 根据需要可以添加掩码
#
#         keys = hidden_vectors
#         values = hidden_vectors
#
#         attention_outputs = self.attention_layer(hidden_vectors, keys, values, attn_mask)
#
#
#         updated_vectors = hidden_vectors + torch.matmul(matrix.unsqueeze(-2), attention_outputs).squeeze(-2)
#
#         updated_vectors = updated_vectors.view(B * L, D)
#
#         return updated_vectors
#
#     def sum(self, vectors, axis):
#         sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
#         return torch.stack(sum_vectors)
#
#     def mean(self, vectors, axis):
#         mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
#         return torch.stack(mean_vectors)
#
#     def gnn(self, inputs):
#         """Cat or pad each input data for batch processing."""
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         fingerprints = [torch.tensor(x, dtype=torch.long).to(device) for x in inputs[:, 0]]
#         adjacencies = [torch.tensor(x, dtype=torch.float).to(device) for x in inputs[:, 1]]
#         molecular_sizes = [torch.tensor(x, dtype=torch.long).to(device) for x in inputs[:, 2]]
#         masks = [torch.tensor(x, dtype=torch.float).to(device) for x in inputs[:, 3]]
#         masks = torch.cat(masks).unsqueeze(-1)
#
#         fingerprints = torch.cat(fingerprints)
#         adjacencies = self.pad(adjacencies)
#
#         """GNN layer (update the fingerprint vectors)."""
#         fingerprint_vectors = self.embed_fingerprint(fingerprints)
#         for l in range(self.layer_hidden):
#             hs = self.update(adjacencies, fingerprint_vectors, l)
#             fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
#
#         """Molecular vector by sum or mean of the fingerprint vectors."""
#         if self.mode == 'sum':
#             molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
#         elif self.mode == 'mean':
#             molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)
#
#         if self.layer_output != 0:
#             for l in self.W_output_:
#                 molecular_vectors = self.activation(l(molecular_vectors))
#
#         """Mask invalid SMILES vectors"""
#         molecular_vectors *= masks
#
#         return molecular_vectors
#
#     def mlp(self, vectors1, vectors2):
#         vectors = torch.cat((vectors1, vectors2), 1)
#         if self.layer_output != 0:
#             for l in self.W_output:
#                 vectors = torch.relu(l(vectors))
#         return vectors
#

import torch
import torch.nn as nn

class GraphClassifier(nn.Module): #RGCN
    def __init__(self, args, relation2id):
        super().__init__()

        self.params = args  # args 包含模型的各种超参数和配置。
        self.relation2id = relation2id  # relation2id 是一个字典，用于映射关系类型到唯一的ID
        self.dropout = nn.Dropout(p=0.3)  # dropout 层用于防止过拟合。
        self.relu = nn.ReLU()  # relu 是激活函数。
        self.train_rels = args.train_rels  # train_rels 和 relations 分别表示训练时使用的关系数量和总的关系数量。
        self.relations = args.num_rels  #
        self.gnn = RGCN(args)  # gnn 是一个图神经网络（GNN），具体实现为 RGCN（Relational Graph Convolutional Network）。

        self.mp_layer1 = nn.Linear(args.feat_dim, args.emb_dim)
        self.mp_layer2 = nn.Linear(args.emb_dim, args.emb_dim)  # mp_layer1 和 mp_layer2 是两个线性层，用于特征变换

        # 根据 args.add_ht_emb 和 args.add_sb_emb 的设置，选择不同的全连接层来处理图的嵌入表示。
        # add_ht_emb表示是否添加头节点和尾节点的嵌入。
        # add_sb_emb 表示是否添加其他特征嵌入。
        if args.add_ht_emb and args.add_sb_emb:
            self.fc_layer = nn.Linear(3 * (1 + args.num_gcn_layers) * args.emb_dim, args.graph_dim)
        elif self.params.add_ht_emb:
            self.fc_layer = nn.Linear(2 * (1 + args.num_gcn_layers) * args.emb_dim, args.graph_dim)
        else:
            self.fc_layer = nn.Linear(args.num_gcn_layers * args.emb_dim, args.graph_dim)

    def drug_feat(self, emb):
        self.drugfeat = emb  # 这个方法用于存储药物的特征表示。

    #输入数据 data 是一个图对象。
    #使用 gnn 对图进行编码，得到节点的嵌入表示。
    # mean_nodes 函数计算图中所有节点嵌入的均值，作为图的整体表示 g_out。

    def forward(self, data):
        g = data
        g.ndata['h'] = self.gnn(g)
        g_out = mean_nodes(g, 'repr')

        # head's embedding 找到图中标识为头节点（id 为 1）和尾节点（id 为 2）的节点，并提取它们的嵌入表示。
        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)
        head_embs = g.ndata['repr'][head_ids]

        # tail's embedding
        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]

        # 根据 add_ht_emb 和 add_sb_emb 的设置，将图的整体表示、头节点嵌入和尾节点嵌入拼接在一起，形成最终的图表示 g_rep。
        if self.params.add_ht_emb and self.params.add_sb_emb:
            g_rep = torch.cat([g_out.view(-1, (1 + self.params.num_gcn_layers) * self.params.emb_dim),
                               head_embs.view(-1, (1 + self.params.num_gcn_layers) * self.params.emb_dim),
                               tail_embs.view(-1, (1 + self.params.num_gcn_layers) * self.params.emb_dim),
                               ], dim=1)

        elif self.params.add_ht_emb:
            g_rep = torch.cat([
                head_embs.view(-1, (1 + self.params.num_gcn_layers) * self.params.emb_dim),
                tail_embs.view(-1, (1 + self.params.num_gcn_layers) * self.params.emb_dim)
            ], dim=1)
        else:
            g_rep = g_out.view(-1, self.params.num_gcn_layers * self.params.emb_dim)

        # 使用 dropout 层对拼接后的图表示 g_rep 进行处理，以防止过拟合。
        # 通过全连接层 fc_layer 将处理后的图表示转换为最终的输出。
        g_rep = self.fc_layer(self.dropout(g_rep))

        # 返回最终的图表示 g_rep。
        return g_rep

def use_transform(vecs, kernel, bias):
    return torch.mm((vecs + bias), kernel)

class ddi_Bert(BertPreTrainedModel):
    def __init__(self, config, gnn_config, args):
        super(ddi_Bert, self).__init__(config)

        self.args = args
        self.num_labels = config.num_labels
        self.bert = PRETRAINED_MODEL_MAP[args.model_type](config=config)
        self.use_sub = args.use_sub
        self.middle_layer_size = args.middle_layer_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)
        self.dropout = nn.Dropout(args.dropout_rate)
        activations = {'relu': nn.ReLU(inplace=False), 'elu': nn.ELU(), 'leakyrelu': nn.LeakyReLU(),
                       'prelu': nn.PReLU(),
                       'relu6': nn.ReLU6, 'rrelu': nn.RReLU(), 'selu': nn.SELU(), 'celu': nn.CELU(), 'gelu': GELU()}
        self.activation = activations['leakyrelu']

        if args.use_sub:
            self.mlp = nn.Linear(config.hidden_size + args.graph_dim, config.hidden_size)
            if args.middle_layer_size == 0:
                self.classifier = nn.Linear(config.hidden_size + args.graph_dim, config.num_labels)
            else:
                self.middle_classifier = nn.Linear(config.hidden_size, args.middle_layer_size)
                self.classifier = nn.Linear(args.middle_layer_size, config.num_labels)

        if args.use_sub:
            self.graph_classifier = initialize_model(args, GraphClassifier, args.load_model)


            # only use the BERT model
        if self.args.model == "only_bert":
            self.label_classifier = FCLayer(config.hidden_size, config.num_labels, args.dropout_rate, use_activation=False)

        # use  BERT + Interaction attention vector
        if self.args.model == "bert_int":
            self.label_classifier = FCLayer(config.hidden_size * 2, config.num_labels, args.dropout_rate, use_activation=False)

        # use  BERT + molecular
        if self.args.model == "bert_mol":
            self.gnn = MolecularGraphNeuralNetwork(args, gnn_config.N_fingerprints, gnn_config.dim,
                                                   gnn_config.layer_hidden,
                                                   gnn_config.layer_output, gnn_config.mode, gnn_config.activation)
            self.label_classifier = FCLayer(config.hidden_size + 2 * gnn_config.dim, config.num_labels,
                                            args.dropout_rate,
                                            use_activation=False)

        # use  BERT + Interaction attention vector + molecular
        if self.args.model == "bert_int_mol":
            self.gnn = MolecularGraphNeuralNetwork(args, gnn_config.N_fingerprints, gnn_config.dim, gnn_config.layer_hidden,
                                                   gnn_config.layer_output, gnn_config.mode, gnn_config.activation)
            self.label_classifier = FCLayer(config.hidden_size*2 + 2*gnn_config.dim, config.num_labels, args.dropout_rate,
                                            use_activation=False)

        # use  BERT + Interaction attention + Entities attention  + molecular
        if self.args.model == "bert_int_ent_mol":
            self.gnn = MolecularGraphNeuralNetwork(args, gnn_config.N_fingerprints, gnn_config.dim,
                                                       gnn_config.layer_hidden,
                                                       gnn_config.layer_output, gnn_config.mode, gnn_config.activation)
            self.label_classifier = FCLayer(config.hidden_size * 3 + 2 * gnn_config.dim, config.num_labels,
                                                args.dropout_rate,
                                                use_activation=False)



        # 使用 BERT + 图分类器
        if self.args.model == "bert_graph":
            if args.use_sub:
                self.graph_classifier = initialize_model(args, GraphClassifier, args.load_model)
            self.label_classifier = FCLayer(config.hidden_size + args.graph_dim, config.num_labels,
                                            args.dropout_rate,
                                            use_activation=False)

        # 使用 BERT + Interaction attention vector + 图分类器
        if self.args.model == "bert_int_graph":
            if args.use_sub:
                self.graph_classifier = initialize_model(args, GraphClassifier, args.load_model)
            self.label_classifier = FCLayer(config.hidden_size * 2 + args.graph_dim, config.num_labels,
                                            args.dropout_rate,
                                            use_activation=False)

        # 使用 BERT + molecular + 图分类器
        if self.args.model == "bert_mol_graph":
            self.gnn = MolecularGraphNeuralNetwork(args, gnn_config.N_fingerprints, gnn_config.dim,
                                                   gnn_config.layer_hidden,
                                                   gnn_config.layer_output, gnn_config.mode, gnn_config.activation)
            if args.use_sub:
                self.graph_classifier = initialize_model(args, GraphClassifier, args.load_model)
            self.label_classifier = FCLayer(config.hidden_size + 2 * gnn_config.dim + args.graph_dim, config.num_labels,
                                            args.dropout_rate,
                                            use_activation=False)

        # 使用 BERT + Interaction attention vector + molecular + 图分类器
        if self.args.model == "bert_int_mol_graph":
            self.gnn = MolecularGraphNeuralNetwork(args, gnn_config.N_fingerprints, gnn_config.dim,
                                                   gnn_config.layer_hidden,
                                                   gnn_config.layer_output, gnn_config.mode, gnn_config.activation)
            if args.use_sub:
                self.graph_classifier = initialize_model(args, GraphClassifier, args.load_model)
            self.label_classifier = FCLayer(args.bert_hidden_size*2, config.num_labels,args.dropout_rate, use_activation=False)
            # self.label_classifier = FCLayer(config.hidden_size * 2 + 2 * gnn_config.dim + args.graph_dim,
            #                                 config.num_labels,
            #                                 args.dropout_rate,
            #                                 use_activation=False)
            # 初始化注意力与门控模块
            self.cross_attention = CrossGraphAttention(dim=args.emb_dim)
            text_dim = args.bert_hidden_size  # 文本特征维度
            fusion_dim = args.emb_dim * 2  # 融合特征维度
            self.text_guided_gate = TextGuidedGate(text_dim, fusion_dim)
            # 第三阶段：残差路径与LayerNorm
            self.layer_norm = nn.LayerNorm(text_dim)  # 根据融合特征维度设置


    def load_model(self):
        self.graph_classifier.load_state_dict(torch.load("my_resnet.pth"))

    @staticmethod
    # Averaging treatment
    def average(hidden_output, list):
        list_unsqueeze = list.unsqueeze(1)
        length_tensor = (list != 0).sum(dim=1).unsqueeze(1)
        sum_vector = torch.bmm(list_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids,
                labels,
                int_list,
                ent_list,
                fingerprint_index,
                fingerprint_data,
                subgraph_index,
                subgraph
                ):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            )

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        pooled_output = self.fc_layer(pooled_output)

        # only BERT model used
        if self.args.model == "only_bert":
            logits = self.label_classifier(pooled_output)
            outputs = (logits,) + outputs[2:]

        # use BERT model and Interaction vector
        if self.args.model == 'bert_int':
            int = self.average(sequence_output, int_list)
            int = self.fc_layer(int)
            concat = torch.cat([int, pooled_output, ], dim=-1)
            logits = self.label_classifier(concat)
            outputs = (logits,) + outputs[2:]

        # use  BERT + Molecualr
        if self.args.model == "bert_mol":
            fingerprint = fingerprint_data[fingerprint_index.cpu()]
            if fingerprint.ndim == 3:
                fingerprint1 = fingerprint[:,0,]
                fingerprint2 = fingerprint[:,1,]
            else:
                fingerprint = np.expand_dims(fingerprint, 0)
                fingerprint1 = fingerprint[:,0,]
                fingerprint2 = fingerprint[:,1,]
            gnn_output1 = self.gnn.gnn(fingerprint1)
            gnn_output2 = self.gnn.gnn(fingerprint2)
            gnn_output = torch.cat((gnn_output1, gnn_output2), -1)

            concat = torch.cat((pooled_output, gnn_output), -1)

            logits = self.label_classifier(concat)
            outputs = (logits,) + outputs[2:]

        # use  BERT + Intearction vector + Molecular
        if self.args.model == "bert_int_mol":
                fingerprint = fingerprint_data[fingerprint_index.cpu()]
                if fingerprint.ndim == 3:
                    fingerprint1 = fingerprint[:, 0, ]
                    fingerprint2 = fingerprint[:, 1, ]
                else:
                    fingerprint = np.expand_dims(fingerprint, 0)
                    fingerprint1 = fingerprint[:, 0, ]
                    fingerprint2 = fingerprint[:, 1, ]
                gnn_output1 = self.gnn.gnn(fingerprint1)
                gnn_output2 = self.gnn.gnn(fingerprint2)
                gnn_output = torch.cat((gnn_output1, gnn_output2), -1)

                int = self.average(sequence_output, int_list)
                int = self.fc_layer(int)

                concat = torch.cat((pooled_output, gnn_output, int), -1)

                logits = self.label_classifier(concat)

                outputs = (logits,) + outputs[2:]

        # use  BERT + Intearction attention + Entities attention +  Molecular
        if self.args.model == "bert_int_ent_mol":
            #print(fingerprint_data.item())
            fingerprint = fingerprint_data[fingerprint_index.cpu()]
            if fingerprint.ndim == 3:
                fingerprint1 = fingerprint[:,0,]
                fingerprint2 = fingerprint[:,1,]
            else:
                fingerprint = np.expand_dims(fingerprint, 0)
                fingerprint1 = fingerprint[:,0,]
                fingerprint2 = fingerprint[:,1,]
            gnn_output1 = self.gnn.gnn(fingerprint1)
            gnn_output2 = self.gnn.gnn(fingerprint2)
            gnn_output = torch.cat((gnn_output1, gnn_output2), -1)

            int = self.average(sequence_output, int_list)
            int = self.fc_layer(int)

            ent = self.average(sequence_output, ent_list)
            ent = self.fc_layer(ent)

            concat = torch.cat((pooled_output, int, ent, gnn_output), -1)

            logits = self.label_classifier(concat)

            outputs = (logits,) + outputs[2:]

            # 使用 BERT 模型和图分类器
        if self.args.model == 'bert_graph':
            if self.use_sub:
                self.graph_classifier.train()
                g_dgl_pos, r_labels_pos, targets_pos = [], [], []
                for idx in subgraph_index:
                    g_dgl_pos_, r_labels_pos_, targets_pos_ = subgraph[idx]
                    g_dgl_pos.append(g_dgl_pos_)
                    r_labels_pos.append(r_labels_pos_)
                    targets_pos.append(targets_pos_)
                g_dgl_pos = dgl.batch(g_dgl_pos).to(self.args.device)
                subgraph_batch = ((g_dgl_pos, r_labels_pos), targets_pos)
                data_pos, r_labels_pos, targets_pos = self.args.move_batch_to_device(subgraph_batch, self.args.device)
                sub_output = self.graph_classifier(data_pos)

                concat = torch.cat([pooled_output, sub_output], -1)
                logits = self.label_classifier(concat)
                outputs = (logits,) + outputs[2:]

        # 使用 BERT 模型、Interaction vector 和图分类器
        if self.args.model == 'bert_int_graph':
            int = self.average(sequence_output, int_list)
            int = self.fc_layer(int)

            if self.use_sub:
                self.graph_classifier.train()
                g_dgl_pos, r_labels_pos, targets_pos = [], [], []
                for idx in subgraph_index:
                    g_dgl_pos_, r_labels_pos_, targets_pos_ = subgraph[idx]
                    g_dgl_pos.append(g_dgl_pos_)
                    r_labels_pos.append(r_labels_pos_)
                    targets_pos.append(targets_pos_)
                g_dgl_pos = dgl.batch(g_dgl_pos).to(self.args.device)
                subgraph_batch = ((g_dgl_pos, r_labels_pos), targets_pos)
                data_pos, r_labels_pos, targets_pos = self.args.move_batch_to_device(subgraph_batch, self.args.device)
                sub_output = self.graph_classifier(data_pos)

                concat = torch.cat([pooled_output, int, sub_output], dim=-1)
                logits = self.label_classifier(concat)
                outputs = (logits,) + outputs[2:]

        # 使用 BERT 模型、分子图和图分类器
        if self.args.model == 'bert_mol_graph':
            # print(fingerprint_data.item())
            fingerprint = fingerprint_data[fingerprint_index.cpu()]
            if fingerprint.ndim == 3:
                fingerprint1 = fingerprint[:, 0, ]
                fingerprint2 = fingerprint[:, 1, ]
            else:
                fingerprint = np.expand_dims(fingerprint, 0)
                fingerprint1 = fingerprint[:, 0, ]
                fingerprint2 = fingerprint[:, 1, ]
            gnn_output1 = self.gnn.gnn(fingerprint1)
            gnn_output2 = self.gnn.gnn(fingerprint2)
            gnn_output = torch.cat((gnn_output1, gnn_output2), -1)

            if self.use_sub:
                self.graph_classifier.train()
                g_dgl_pos, r_labels_pos, targets_pos = [], [], []
                for idx in subgraph_index:
                    g_dgl_pos_, r_labels_pos_, targets_pos_ = subgraph[idx]
                    g_dgl_pos.append(g_dgl_pos_)
                    r_labels_pos.append(r_labels_pos_)
                    targets_pos.append(targets_pos_)
                g_dgl_pos = dgl.batch(g_dgl_pos).to(self.args.device)
                subgraph_batch = ((g_dgl_pos, r_labels_pos), targets_pos)
                data_pos, r_labels_pos, targets_pos = self.args.move_batch_to_device(subgraph_batch, self.args.device)
                sub_output = self.graph_classifier(data_pos)

                concat = torch.cat([pooled_output, gnn_output, sub_output], dim=-1)
                logits = self.label_classifier(concat)
                outputs = (logits,) + outputs[2:]

        # 使用 BERT 模型、Interaction vector、分子图和图分类器
        if self.args.model == 'bert_int_mol_graph':
            fingerprint = fingerprint_data[fingerprint_index.cpu()]
            #fingerprint = fingerprint_data[fingerprint_index.cpu().numpy()]
            if fingerprint.ndim == 3:
                fingerprint1 = fingerprint[:, 0, ]
                fingerprint2 = fingerprint[:, 1, ]
            else:
                fingerprint = np.expand_dims(fingerprint, 0)
                fingerprint1 = fingerprint[:, 0, ]
                fingerprint2 = fingerprint[:, 1, ]
            gnn_output1 = self.gnn.gnn(fingerprint1)
            gnn_output2 = self.gnn.gnn(fingerprint2)
            gnn_output = torch.cat((gnn_output1, gnn_output2), -1)


            if self.use_sub:
                self.graph_classifier.train()
                g_dgl_pos, r_labels_pos, targets_pos = [], [], []
                for idx in subgraph_index:
                    g_dgl_pos_, r_labels_pos_, targets_pos_ = subgraph[idx]
                    g_dgl_pos.append(g_dgl_pos_)
                    r_labels_pos.append(r_labels_pos_)
                    targets_pos.append(targets_pos_)
                g_dgl_pos = dgl.batch(g_dgl_pos).to(self.args.device)
                subgraph_batch = ((g_dgl_pos, r_labels_pos), targets_pos)
                data_pos, r_labels_pos, targets_pos = self.args.move_batch_to_device(subgraph_batch, self.args.device)
                sub_output = self.graph_classifier(data_pos)

                # 第一阶段：子图-分子交叉注意力
                fused_graph_chem = self.cross_attention(sub_output, gnn_output)
                # 第二阶段：文本引导融合
                gated_fusion = self.text_guided_gate(pooled_output, fused_graph_chem)

                # 第三阶段：残差路径与LayerNorm
                # 收集原始模态特征（示例：分子特征、子图特征、文本特征）
                original_feats = torch.cat([gnn_output, sub_output, pooled_output], dim=-1)
                # 残差连接
                residual_connection = gated_fusion + original_feats[:, :gated_fusion.shape[-1]]
                # LayerNorm稳定训练
                final_fused_feat = self.layer_norm(residual_connection)

                # 后续处理
                int = self.average(sequence_output, int_list)
                int = self.fc_layer(int)
                concat = torch.cat([final_fused_feat, int], dim=-1)
                logits = self.label_classifier(concat)
                outputs = (logits,) + outputs[2:]

        # if labels is not None:
        #     if self.num_labels == 1:
        #         loss_fct = nn.MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = nn.CrossEntropyLoss()
        #         print(f"当前的标签是：{labels.view(-1)}")
        #         print(f"当前的数量是：{self.num_labels}")
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #     outputs = (loss,) + outputs

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1),labels.view(-1))
            else:
                loss_fct = MultiFocalLoss(self.num_labels, [0.8, 0.07, 0.08, 0.04, 0.01])
                loss = loss_fct(logits.view(-1,self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs #将计算得到的损失loss添加到输出结果的前面


        return outputs  # (loss), logits, (hidden_states), (attentions)


    def zero_init_params(self):
        self.update_cnt = 0
        for x in self.parameters():
            x.data *= 0

    def accumulate_params(self, model):
        self.update_cnt += 1
        for x, y in zip(self.parameters(), model.parameters()):
            x.data += y.data

    def restore_params(self):
        for x in self.parameters():
            x.data *= self.update_cnt





