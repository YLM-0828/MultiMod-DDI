import os

import torch

from graph_utils import move_batch_to_device_dgl, move_batch_to_device_dgl_ddi2

from datasets import generate_subgraph_datasets

os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0, 1, 2, 3]))
import argparse
import sys

sys.path.append('../')
sys.path.append('./')
from utils import MODEL_CLASSES, MODEL_PATH_MAP, init_logger
from trainer import Trainer
from load_data import load_and_cache_examples

torch.autograd.set_detect_anomaly(True)


def main(args):

    init_logger()
    # if args.do_train:
    #     train_dataset, train_graph_dataset = load_and_cache_examples(args, mode='train', pos="total_pos", neg="total_neg")
    # if args.do_eval:
    #     dev_dataset, dev_graph_dataset = load_and_cache_examples(args, mode='dev', pos="total_pos", neg="total_neg")
    # if args.do_test:
    #     test_dataset, test_graph_dataset = load_and_cache_examples(args, mode="test", pos="total_pos", neg="total_neg")
    train_dataset, dev_dataset, test_dataset, subgraph = load_and_cache_examples(args, pos="total_pos", neg="total_neg")
    trainer = Trainer(args, train_dataset=train_dataset, dev_dataset=dev_dataset, test_dataset=test_dataset)


    if args.do_train:
        trainer.train(subgraph)

    # if args.do_eval:
    #     #trainer.load_model()
    #     trainer.evaluate(subgraph)

    if args.do_test:
        # trainer.load_model()
        trainer.load_best_model()
        trainer.predict(subgraph)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_filename", default=None, type=str, required=True, help="train file name")

    parser.add_argument("--dev_filename", default=None, type=str, required=True, help=" dev file name")

    parser.add_argument("--test_filename", default=None, type=str, required=True, help=" test file name")

    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")

    parser.add_argument("--label_file", default=None, type=str, required=True, help="Label file")

    parser.add_argument("--model_dir", default=None, type=str, required=True, help="Path to model")

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")

    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--model", type=str, required=True,
                        help="which model to use: only_bert , bert_int, bert_int_mol, bert_int_ent_mol")

    parser.add_argument("--do_train", action="store_true", help="whether do train.")

    parser.add_argument("--do_eval", action="store_true", help="whether do dev.")

    parser.add_argument("--do_test", action="store_true", help="whether do test.")

    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")

    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")

    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")

    parser.add_argument("--num_train_epochs", default=20.0, type=float,
                        help="Total number of trainingtop epochs to perform.")

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--middle_layer_size', type=int, default=0, help="Dimention of middle layer")


    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")

    parser.add_argument('--logging_steps', type=int, default=800, help="Log every X updates steps.")

    parser.add_argument('--save_steps', type=int, default=800, help="Save checkpoint every X updates steps.")

    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")

    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument('--tpu', action='store_true',
                        help="Whether to run on the TPU defined in the environment variables")

    parser.add_argument('--tpu_ip_address', type=str, default='',
                        help="TPU IP address if none are set in the environment variables")

    parser.add_argument('--tpu_name', type=str, default='',
                        help="TPU name if none are set in the environment variables")

    parser.add_argument('--xrt_tpu_config', type=str, default='',
                        help="XRT TPU config if none are set in the environment variables")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")

    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")

    parser.add_argument('--parameter_averaging', action='store_true', help="Whether to use parameter averaging")

    parser.add_argument('--use_Under_sampling_and_over_sampling', default=True,
                        help="Whether to use Under_sampling_and_over_sampling ")

    # For Molecular Structure
    parser.add_argument('--fingerprint_dir', default=None, required=True, type=str,
                        help="The path to fingerprint_file .npy files")

    parser.add_argument('--molecular_vector_size', default=50, type=int, help="Dimention of molecular embeddings.")

    parser.add_argument('--gnn_layer_hidden', default=5, type=int, help="The number of hidden layer")

    parser.add_argument('--gnn_layer_output', default=1, type=int, help="The number of output layer")

    parser.add_argument('--gnn_mode', default='sum', type=str, help="The method of aggregating atom vectors")

    parser.add_argument('--gnn_activation', type=str, default='leakyrelu', help="GNN activation function")

    parser.add_argument('--kfold', '-k', type=int, default=1, help='The fold of the cross validation')

    parser.add_argument('--bert_hidden_size', type=int, default=768,
                        help='BERT模型隐藏层维度，BERT-base默认768，BERT-large默认1024')

    # For 子图特征
    parser.add_argument('--use_sub', action='store_true', help='Whether to use sub graph features')
    parser.add_argument('--work_dir', default=None, type=str, help="The address of the project's location")
    parser.add_argument('--graph_dim', '-g_dim', type=int, default=50, help='The dim of the subgraph features')
    parser.add_argument('--experiment_name', '-e', type=str, default='default1',
                        help='A folder with this name would be created to dump saved models and log files')
    parser.add_argument('--load_model', action='store_true', help='Load existing model?')
    parser.add_argument('--total_file', type=str, default='total', help='Name of file containing total triplets')
    parser.add_argument('--train_file', '-tf', type=str, default='train',
                        help='Name of file containing training triplets')
    parser.add_argument('--valid_file', '-vf', type=str, default='dev',
                        help='Name of file containing validation triplets')
    parser.add_argument('--test_file', '-ttf', type=str, default='test',
                        help='Name of file containing validation triplets')
    parser.add_argument('--dataset', '-d', type=str, default='drugbank',help="Dataset String")
    parser.add_argument('--max_links', type=int, default=250000,
                        help="Set maximum number of train links (to fit into memory)")
    parser.add_argument('--feat', '-f', type=str, default='morgan',
                        help='the type of the feature we use in molecule modeling')
    parser.add_argument('--feat_dim', type=int, default=8, help='the dimension of the feature')  #2048  8
    parser.add_argument('--emb_dim', "-dim", type=int, default=50, help="Entity embedding size")
    parser.add_argument('--add_traspose_rels', '-tr', type=bool, default=False,
                        help='Whether to append adj matrix list with symmetric relations')
    parser.add_argument('--num_neg_samples_per_link', '-neg', type=int, default=0,
                        help="Number of negative examples to sample per positive link")
    parser.add_argument('--use_kge_embeddings', "-kge", type=bool, default=False,
                        help='Whether to use pretrained KGE embeddings')
    parser.add_argument('--kge_model', type=str, default='TransE',
                        help="Which KGE model to load entity embeddings from")
    parser.add_argument('--enclosing_sub_graph', '-en', type=bool, default=True,
                        help='whether to only consider enclosing subgraph')
    parser.add_argument('--hop', type=int, default=2, help="Enclosing subgraph hop number")
    parser.add_argument('--constrained_neg_prob', '-cn', type=float, default=0.0,
                        help='with what probability to sample constrained heads/tails while neg sampling')
    parser.add_argument('--max_nodes_per_hop', '-max_h', type=list, default=[20, 20, 20],
                        help='if > 0, upper bound the # nodes per hop by subsampling')
    parser.add_argument('--num_bases', '-b', type=int, default=4,
                        help='Number of basis functions to use for GCN weights')
    parser.add_argument('--attn_rel_emb_dim', '-ar_dim', type=int, default=32,
                        help='Relation embedding size for attention')
    parser.add_argument('--num_gcn_layers', '-l', type=int, default=2, help='Number of GCN layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate in GNN layers')
    parser.add_argument('--edge_dropout', type=float, default=0.4, help='Dropout rate in edges of the subgraphs')
    parser.add_argument('--gnn_agg_type', '-a', type=str, choices=['sum', 'mlp', 'gru'], default='sum',
                        help='what type of aggregation to do in gnn msg passing')
    parser.add_argument('--add_ht_emb', '-ht', type=bool, default=True,
                        help='whether to concatenate head/tail embedding with pooled graph representation')
    parser.add_argument('--add_sb_emb', '-sb', type=bool, default=True,
                        help='whether to concatenate subgraph embedding with pooled graph representation')
    parser.add_argument('--has_attn', '-attn', type=bool, default=True, help='whether to have attn in model or not')
    parser.add_argument('--has_kg', '-kg', type=bool, default=True, help='whether to have kg in model or not')
    parser.add_argument('--add_transe_emb', type=bool, default=True,
                        help='whether to have knowledge graph embedding in model or not')
    parser.add_argument('--gamma', type=float, default=0.2, help='The threshold for attention')

    parser.add_argument('--pretrained_dir', type=str, help="The path to pre-trained model dir")
    parser.add_argument('--freeze_pretrained_parameters', action='store_true',
                        help="Whether to freeze parameters pretrained on database")

    args = parser.parse_args()
    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]

    args.file_paths = {
        'total': os.path.join(args.work_dir, 'data/{}/{}.txt'.format(args.dataset, args.total_file)),
    }
    args.move_batch_to_device = move_batch_to_device_dgl if args.dataset == 'drugbank' else move_batch_to_device_dgl_ddi2
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "The folder {} already has an output file. Please use overwrite_output_dir if you want to overwrite the output".format(
                args.output_dir))

    args.db_path = os.path.join(args.work_dir,f'data/{args.dataset}/subgraphs_en_{args.enclosing_sub_graph}_neg_{args.num_neg_samples_per_link}_hop_{args.hop}')

    #args.db_path = r"E:\pythonProjects\IMSE-mainME2\ddi_task\share_code\data\drugbank\subgraphs_en_True_neg_0_hop_3"
    if not os.path.isdir(args.db_path):
        generate_subgraph_datasets(args, splits=['total'])
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    # args.device = 'cpu'

    main(args)
