from tqdm import tqdm
import torch
import time
from datetime import timedelta
import pickle as pkl
import os
from random import shuffle
import numpy as np
from sklearn.model_selection import KFold
PAD, CLS = '[PAD]', '[CLS]'


def load_dataset(config):
    """
    返回结果 5个list ids, label, ids_len, mask, others
    :param file_path:
    :param seq_len:
    :return: content[list ids, label, ids_len, mask, others]
    """
    contents = []
    # 读取数据文件
    with open(config.data_path, 'r', encoding='utf-8') as f:
        f_item = f.readlines()
        # 每一条数据操作
        for line in tqdm(f_item[1:-1]):

            line = line.strip()  # 去除每条数据后的换行符
            if not line:
                continue
            k = line.split(',')  # 切分文本与标签
            content = k[0]
            label = k[-1]
            others = k[1:-1]
            others = [float(x) for x in others]
            token = config.tokenizer.tokenize(content)  # 使用pytorch_pretrained中定义的切词器对文本经行分词
            token = [CLS] + token  # 让token标准格式化
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)  # 获得token的ids（每个字的编号）

            pad_size = config.pad_size  # 分词后一条标准的文本数据长度

            if pad_size:
                # 如果文本数据长度小于标准长度，补[0]达到标准长度，mask、ids都要补
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))  # mask形式为：文本长度个[1]，标准长度-文本长度个[0]
                    token_ids = token_ids + ([0] * (pad_size - len(token)))  # ids形式同上
                # 如果文本数据长度大于标准长度，截取成为标准长度
                else:
                    mask = [1] * pad_size  # mask全为[1]
                    token_ids = token_ids[:pad_size]  # ids只取前标准长度个
                    seq_len = pad_size  # 将句子长度重新定义成标准长度
            contents.append((token_ids, int(label), seq_len, mask, others))  # 将上述四个必要的结果塞入已初始化的content列表
    return contents


# def build_dataset(config):
#     """
#     返回值 train, dev ,test
#     :param config:
#     :return:
#     """
#     if os.path.exists(config.datasetpkl):  # 如果存在数据集，直接使用，不再创建
#         dataset = pkl.load(open(config.datasetpkl, 'rb'))
#         train = dataset['train']
#         dev = dataset['dev']
#         test = dataset['test']
#     else:  # 如果不存在数据集，调用load_dataset创建数据集
#         dataset_pre = load_dataset(config)
#         shuffle(dataset_pre)
#         train = dataset_pre[0:6999]
#         dev = dataset_pre[7000:8499]
#         test = dataset_pre[8500:]
#         dataset = {'train': train, 'dev': dev, 'test': test}
#         pkl.dump(dataset, open(config.datasetpkl, 'wb'))
#     return train, dev, test
def build_dataset(config, fold):
    """
    返回值 train, dev ,test
    :param fold:
    :param config:
    :return:
    """
    if os.path.exists(config.datasetpkl):  # 如果存在数据集，直接使用，不再创建
        dataset = pkl.load(open(config.datasetpkl, 'rb'))
        dataset = np.array(dataset, dtype=object)
        train, test = K_Fold_spilt(10, fold, dataset, config)
        train.tolist()
        test.tolist()
    else:  # 如果不存在数据集，调用load_dataset创建数据集
        dataset_pre = load_dataset(config)
        shuffle(dataset_pre)
        pkl.dump(dataset_pre, open(config.datasetpkl, 'wb'))
        train, test = build_dataset(config, fold)
    return train, test


# 输入数据推荐使用numpy数组，使用list格式输入会报错
def K_Fold_spilt(K, fold, data, config):
    """
    :param config:
    :param K: 要把数据集分成的份数。如十次十折取K=10
    :param fold: 要取第几折的数据。如要取第5折则 flod=5
    :param data: 需要分块的数据
    :return: 对应折的训练集、测试集
    """
    split_list = []
    if os.path.exists(config.index):
        split_list = pkl.load(open(config.index, 'rb'))
        train_index, test_index = split_list[2 * fold], split_list[2 * fold + 1]
        train, test = data[train_index], data[test_index]
    else:
        kf = KFold(n_splits=K)
        for train_index, test_index in kf.split(data):
            split_list.append(train_index.tolist())
            split_list.append(test_index.tolist())
        pkl.dump(split_list, open(config.index, 'wb'))
        train, test = K_Fold_spilt(K, fold, data, config)
    return train, test  # 已经分好块的数据集


class DatasetIterator(object):
    def __init__(self, dataset, batch_size, device):
        self.batch_size = batch_size  # 每个batch的大小：config.batch_size
        self.dataset = dataset  # 已经处理好的数据集train，dev，test
        self.n_batches = len(dataset) // batch_size  # batch的数量（将数据集切分为若干个batch）
        self.residue = False  # 记录batch数量是否为整数
        if len(dataset) % self.n_batches != 0:
            self.residue = True
        self.index = 0  # 标记位（当前是第几个batch）
        self.device = device

    def _to_tensor(self, datas):  # 将Bert所需要的三个数据与label转换成tensor，并返回
        x = torch.LongTensor([item[0] for item in datas]).to(self.device)  # 样本数据ids
        y = torch.LongTensor([item[1] for item in datas]).to(self.device)  # 标签数据label

        seq_len = torch.LongTensor([item[2] for item in datas]).to(self.device)  # 每一个序列的真实长度
        mask = torch.LongTensor([item[3] for item in datas]).to(self.device)
        others = torch.FloatTensor([item[4] for item in datas]).to(self.device)

        return (x, seq_len, mask, others), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:  # 整除情况，同时标记位等于batch总数（即标记位移到最后一位）
            batches = self.dataset[self.index * self.batch_size: len(self.dataset)]  # 从数据集截取batch操作，分batch的具体操作
            self.index += 1  # 标记位后移
            batches = self._to_tensor(batches)  # 将数据转换成tensor形式
            return batches

        elif self.index > self.n_batches:  # 已经处理完成全部的数据
            self.index = 0
            raise StopIteration  # 结束迭代器

        else:  # 标记位未达到最后一位时
            batches = self.dataset[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:  # 根据整除情况，确定十分增加batch的数量
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):  # 创建迭代器，并返回
    iter = DatasetIterator(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """
    获取已经使用的时间
    :param start_time:
    :return:
    """
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
