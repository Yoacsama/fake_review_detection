import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer
import torch.nn.functional as F

"""
BERT模型
"""


class Model_BERT(nn.Module):

    def __init__(self, config):
        super(Model_BERT, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # x [ids, seq_len, mask]
        context = x[0]  # 对应输入的句子 shape[64,32]
        mask = x[2]  # 对padding部分进行mask shape[64,32]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=True)  # shape [128,768]
        out = self.fc(pooled)  # shape [128,2]
        return out


"""
BERTCNN模型
"""


class Model_BERTCNN(nn.Module):
    def __init__(self, config):
        super(Model_BERTCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
             [nn.Conv2d(in_channels=1, out_channels=config.num_filters, kernel_size=(k, config.hidden_size)) for k in config.filter_sizes]
        )

        self.droptout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x)
        x = x.squeeze(3)
        size = x.size(2)
        x = F.max_pool1d(x, size)
        x = x.squeeze(2)
        return x  # [64, 256]

    def forward(self, x):
        # x [ids, seq_len, mask]
        context = x[0]  # 对应输入的句子 shape[64,32]
        mask = x[2]  # 对padding部分进行mask shape[64,32]
        encoder_out, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)  # [128,768]
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv)for conv in self.convs], 1)  # [128, 3*256]
        out = self.droptout(out)
        out = self.fc(out)
        return out


"""
BERTLSTM模型
"""


class Model_BERTLSTM(nn.Module):

    def __init__(self, config):
        super(Model_BERTLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers, batch_first=True, dropout=config.dropout, bidirectional=True)

        self.dropout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(config.rnn_hidden * 2, config.num_classes)

    def forward(self, x):
        # x [ids, seq_len, mask]
        context = x[0]  # 对应输入的句子 shape[64,32]
        mask = x[2]  # 对padding部分进行mask shape[64,32]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out, _ = self.lstm(encoder_out)
        out = self.dropout(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


"""
BERT+BILSTM+CNN 并联模型
"""


class Model_BERTMIX(nn.Module):

    def __init__(self, config):
        super(Model_BERTMIX, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
             [nn.Conv2d(in_channels=1, out_channels=config.num_filters, kernel_size=(k, config.hidden_size)) for k in config.filter_sizes]
        )
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers, batch_first=True,
                            dropout=config.dropout, bidirectional=True)
        self.droptout = nn.Dropout(config.dropout)

        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes)+config.rnn_hidden * 2+14, config.num_classes)

    def conv_and_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x)
        x = x.squeeze(3)
        size = x.size(2)
        x = F.max_pool1d(x, size)
        x = x.squeeze(2)
        return x  # [64, 256]

    def forward(self, x):
        # x [ids, seq_len, mask]
        context = x[0]  # 对应输入的句子 shape[64,32]
        mask = x[2]  # 对padding部分进行mask shape[64,32]
        encoder_out, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)  # [128,768]
        Out_BILSTM, _ = self.lstm(encoder_out)
        Out_BILSTM = Out_BILSTM[:, -1, :]
        Out_CNN = encoder_out.unsqueeze(1)
        Out_CNN = torch.cat([self.conv_and_pool(Out_CNN, conv)for conv in self.convs], 1)  # [128, 3*256]
        out = torch.cat((Out_BILSTM, Out_CNN), 1)
        out = torch.cat((out, x[3]), 1)
        out = self.droptout(out)
        out = self.fc(out)
        return out


class BFA(nn.Module):

    def __init__(self, config):
        super(BFA, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
             [nn.Conv2d(in_channels=1, out_channels=config.num_filters, kernel_size=(k, config.hidden_size)) for k in config.filter_sizes]
        )
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers, batch_first=True,
                            dropout=config.dropout, bidirectional=True)
        self.feature_net1 = nn.Linear(1, 64)
        self.droptout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes)+config.rnn_hidden * 2 + 64, config.num_classes)

    def conv_and_pool(self, x, conv):
        x = conv(x)
        x = F.relu(x)
        x = x.squeeze(3)
        size = x.size(2)
        x = F.max_pool1d(x, size)
        x = x.squeeze(2)
        return x  # [64, 256]

    def forward(self, x):
        # x [ids, seq_len, mask]
        context = x[0]  # 对应输入的句子 shape[64,32]
        mask = x[2]  # 对padding部分进行mask shape[64,32]
        encoder_out, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)  # [128,768]
        Out_BILSTM, _ = self.lstm(encoder_out)
        Out_BILSTM = Out_BILSTM[:, -1, :]
        Out_CNN = encoder_out.unsqueeze(1)
        Out_CNN = torch.cat([self.conv_and_pool(Out_CNN, conv)for conv in self.convs], 1)  # [128, 3*256]
        Out_Feature = self.feature_net1(x[3])
        out = torch.cat((Out_BILSTM, Out_CNN), 1)
        out = torch.cat((out, Out_Feature), 1)
        out = self.droptout(out)
        out = self.fc(out)
        return out


