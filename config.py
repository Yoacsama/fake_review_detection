import torch
from pytorch_pretrained import BertModel, BertTokenizer


class Config_BERT():
    """
    配置参数
    """

    def __init__(self):
        # 模型的名称
        self.model_name = 'Bert'
        # 数据集
        self.data_path = 'source/data/stance_dataA.csv'
        # dataset
        self.datasetpkl = 'source/data/stance_dataA.pkl'
        # index
        self.index = 'source/data/index_stance_dataA.pkl'
        # 类别名单
        self.class_list = [x.strip() for x in open('source/data/stance_class.txt', encoding="utf-8").readlines()]
        # 模型保存路径
        self.save_path = 'source/saved_model/stance_dataA' + self.model_name
        self.ROC_path = 'source/画图数据/stance/'
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 若超过1000 batch效果还没有提升，提前结束训练
        self.require_improvement = 1000
        # 类别数
        self.num_classes = len(self.class_list)
        # epoch数
        self.num_epochs = 30
        # batch_size
        self.batch_size = 32
        # 每句话处理的长度(短填，长切）
        self.pad_size = 64
        # 学习率
        self.learning_rate = 1e-5
        # bert预训练模型位置
        self.bert_path = 'bert_pretrain'
        # bert切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # bert隐层层个数
        self.hidden_size = 768


class Config_BERTCNN():
    """配置参数"""
    def __init__(self):
        # 模型名称
        self.model_name = "BERTCNN"
        # 数据集
        self.data_path = 'source/data/data10000.csv'
        # dataset
        self.datasetpkl = 'source/data/dataset10000.pkl'
        # 类别名单
        self.class_list = [x.strip() for x in open('source/data/class.txt').readlines()]
        # 模型保存路径
        self.save_path = 'source/saved_dict/' + self.model_name + '.ckpt'
        # 运行设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 若超过1000 batch效果还没有提升，提前结束训练
        self.require_improvement = 1000
        # 类别数量
        self.num_classes = len(self.class_list)
        # epoch数
        self.num_epochs = 3
        # batch_size
        self.batch_size = 64
        # 每句话处理的长度(短填，长切）
        self.pad_size = 32
        # 学习率
        self.learning_rate = 1e-5
        # 预训练位置
        self.bert_path = 'bert_pretrain'
        # bert的 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # Bert的隐藏层数量
        self.hidden_size = 768
        # 卷积核尺寸
        self.filter_sizes = (2, 3, 4)
        # 卷积核数量
        self.num_filters = 256
        # dropout
        self.dropout = 0.5


class Config_BERTLSTM():
    """配置参数"""
    def __init__(self):
        # 模型名称
        self.model_name = "BERTLSTM"
        # 数据集
        self.data_path = 'source/data/data10000.csv'
        # dataset
        self.datasetpkl = 'source/data/dataset10000.pkl'
        # 类别名单
        self.class_list = [x.strip() for x in open('source/data/class.txt').readlines()]
        # 模型保存路径
        self.save_path = 'source/saved_dict/' + self.model_name + '.ckpt'
        # 运行设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 若超过1000 batch效果还没有提升，提前结束训练
        self.require_improvement = 1000
        # 类别数量
        self.num_classes = len(self.class_list)
        # epoch数
        self.num_epochs = 3
        # batch_size
        self.batch_size = 64
        # 每句话处理的长度(短填，长切）
        self.pad_size = 32
        # 学习率
        self.learning_rate = 1e-5
        # 预训练位置
        self.bert_path = 'bert_pretrain'
        # bert的 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # Bert的隐藏层数量
        self.hidden_size = 768
        # RNN隐层层数量
        self.rnn_hidden = 256
        # rnn数量
        self.num_layers = 2
        # dropout
        self.dropout = 0.5


class Config_BERTMIX():
    """配置参数"""
    def __init__(self):
        # 模型名称
        self.model_name = "BERTMIX"
        # 数据集
        self.data_path = 'source/data/stance_dataA.csv'
        # dataset
        self.datasetpkl = 'source/data/stance_dataA.pkl'
        # index
        self.index = 'source/data/index_stance_dataA.pkl'
        # 类别名单
        self.class_list = [x.strip() for x in open('source/data/stance_class.txt', encoding="utf-8").readlines()]
        # 模型保存路径
        self.save_path = 'source/saved_model/stance_dataA' + self.model_name
        self.ROC_path = 'source/画图数据/stance/'
        # 运行设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 若超过1000 batch效果还没有提升，提前结束训练
        self.require_improvement = 1000
        # 类别数量
        self.num_classes = len(self.class_list)
        # epoch数
        self.num_epochs = 20
        # batch_size
        self.batch_size = 32
        # 每句话处理的长度(短填，长切）
        self.pad_size = 32
        # 学习率
        self.learning_rate = 1e-5
        # 预训练位置
        self.bert_path = 'bert_pretrain'
        # bert的 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # Bert的隐藏层数量
        self.hidden_size = 768
        # 卷积核尺寸
        self.filter_sizes = (2, 3, 4)
        # 卷积核数量
        self.num_filters = 256
        # RNN隐层层数量
        self.rnn_hidden = 256
        # rnn数量
        self.num_layers = 2
        # dropout
        self.dropout = 0.5