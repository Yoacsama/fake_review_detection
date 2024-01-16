import numpy as np
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
from sklearn import metrics
import time
from pytorch_pretrained.optimization import BertAdam
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm


def train(config, model, train_iter, test_iter, K):
    """
    模型训练方法
    :param config:
    :param model:
    :param train_iter:
    :param test_iter:
    :return:
    """
    start_time = time.time()
    # 启动 BatchNormalization 和 dropout
    model.train()
    # 拿到所有model中的参数
    param_optimizer = list(model.named_parameters())
    # 不需要衰减的参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        # n为参数标签，p为参数值。对p(参数值经行操作)，如果任意一个no_decay中的标签都不在n里，p衰减（即将该衰减的衰减，不该衰减的不衰减）
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(params=optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * config.num_epochs)

    total_batch = 0  # 记录进行多少batch
    test_best_loss = float('inf')  # 记录校验集合最好的loss
    train_best_loss = float('inf')
    train_good_acc = float('inf')
    last_improve = 0  # 记录上次校验集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升，停止训练
    # true = []
    # pred_y = []
    # true = np.array(true)
    # pred_y = np.array(pred_y)
    model.train()
    for epoch in range(config.num_epochs):
        # print('Epoch [{}/{}]:'.format(epoch + 1, config.num_epochs))
        loop = tqdm(train_iter, leave=False)
        loop.set_description('Epoch [{}/{}]'.format(epoch, config.num_epochs))
        for trains, labels in loop:
            total_batch = total_batch + 1
            outputs = model(trains)
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward(retain_graph=False)
            optimizer.step()
            # if i == 0:
            #     true = labels.data.cpu().numpy()
            #     pred_y = torch.max(outputs.data, 1)[0].cpu().numpy()
            # else:
            #     true = np.append(true, labels.data.cpu().numpy())  # 读取label数据
            #     pred_y = np.append(pred_y, torch.max(outputs.data, 1)[0].cpu().numpy())
            if total_batch % 5 == 0:  # 每多少次输出在训练集和校验集上的效果
                true_temp = labels.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true_temp, predict)  # 比较得出训练集上的准确率
                train_loss = loss.item()
                test_acc, test_loss = evaluate(config, model, test_iter, K)  # 得到校验集上的准确率与loss
                if test_loss < test_best_loss:  # 找到最好的一次训练成功,更新dev_best_loss并保存模型
                    test_best_loss = test_loss
                    train_best_loss = train_loss
                    train_good_acc = train_acc
                    torch.save(model.state_dict(), config.save_path + str(K) + '.ckpt')
                    improve = '√'
                    last_improve = total_batch  # 记录下降过的batch数
                else:
                    improve = ''
                time_dif = utils.get_time_dif(start_time)
                # msg = 'Iter:{0:>6}, Train Loss:{1:>5.2}, Train Acc:{2:>6.2%}, Test Loss{3:>5.2}, Test Acc{4:>6.2%}, Time:{5} {6}'
                # print(msg.format(total_batch, loss.item(), train_acc, test_loss, test_acc, time_dif, improve))
                msg_loss = '{0:>5.2}'
                loop.set_postfix(loss=msg_loss.format(test_loss), acc=test_acc, best_loss=msg_loss.format(test_best_loss), last_update=total_batch - last_improve)
                # print("   true:", true_temp[:20])
                # print("predict:", predict[:20])
                model.train()
            if total_batch - last_improve > config.require_improvement:
                # 在验证集合上loss超过1000batch没有下降，结束训练
                print('在校验数据集合上已经很长时间没有提升了，模型自动停止训练')
                flag = True
                break

        if flag:
            break
    # fpr, tpr, thresholds = metrics.roc_curve(true, pred_y, pos_label=1)
    # auc = roc_auc_score(true, pred_y)
    # 精确率，召回率，阈值
    # precision, recall, thresholds_PR = precision_recall_curve(true, pred_y)
    # np.savetxt(config.ROC_path + str(K) + "precision.txt", precision)
    # np.savetxt(config.ROC_path + str(K) + "recall.txt", recall)
    # np.savetxt(config.ROC_path + str(K) + "thresholds_PR.txt", thresholds_PR)
    # np.savetxt(config.ROC_path + str(K) + "pred_y.txt", pred_y)
    # np.savetxt(config.ROC_path + str(K) + "true.txt", true)
    # np.savetxt(config.ROC_path + str(K) + "fpr.txt", fpr)
    # np.savetxt(config.ROC_path + str(K) + "tpr.txt", tpr)
    # np.savetxt(config.ROC_path + str(K) + "thresholds.txt", thresholds)
    # with open(config.ROC_path + "auc.txt", 'a') as f:
    #     f.write("fold{0}: ".format(K) + str(auc))
    test_loss, test_acc = test(config, model, test_iter, K)
    score = [train_best_loss, train_good_acc, test_loss, test_acc]
    return score


def evaluate(config, model, dev_iter, fold, test=False):
    """
    :param test:
    :param config:
    :param model:
    :param dev_iter:
    :param fold:
    :return:
    """
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    predict_test_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in dev_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total = loss_total + loss
            labels = labels.data.cpu().numpy()
            predict_test = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict = torch.max(outputs.data, 1)[0].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_test_all = np.append(predict_test_all, predict_test)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_test_all)
    if test:
        report = metrics.classification_report(labels_all, predict_test_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_test_all)
        np.savetxt(config.ROC_path + str(fold) + "pred_y.txt", predict_all)
        np.savetxt(config.ROC_path + str(fold) + "true.txt", labels_all)
        return acc, loss_total / len(dev_iter), report, confusion

    return acc, loss_total / len(dev_iter)


# def test(config, model, test_iter):
#     """
#     模型测试
#     :param config:
#     :param model:
#     :param test_iter:
#     :return:
#     """
#     model.load_state_dict(torch.load(config.save_path))
#     model.eval()
#     start_time = time.time()
#     test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
#     msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}'
#     print(msg.format(test_loss, test_acc))
#     print("Precision, Recall and F1-Score")
#     print(test_report)
#     print("Confusion Maxtrix")
#     print(test_confusion)
#     time_dif = utils.get_time_dif(start_time)
#     print("使用时间：", time_dif)


def test(config, model, test_iter, k):
    """
    模型测试
    :param config:
    :param model:
    :param test_iter:
    :return:
    """
    model.load_state_dict(torch.load(config.save_path+ str(k) + '.ckpt'))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, k, test=True)
    msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score")
    print(test_report)
    print("Confusion Matrix")
    print(test_confusion)
    time_dif = utils.get_time_dif(start_time)
    print("使用时间：", time_dif)
    return test_loss, test_acc
