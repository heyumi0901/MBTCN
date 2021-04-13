import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append("../../")

from model import TCN
import numpy as np
import argparse
from datapro import readFile
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam
from torch import nn
from MAT import plot_confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import colorsys
import random
import os

def get_n_hls_colors(num):
    hls_colors = []
    i = 0
    step = 360.0 / num
    while i < 360:
        h = i
        s = 90 + random.random() * 10
        l = 50 + random.random() * 10
        _hlsc = [h / 360.0, l / 100.0, s / 100.0]
        hls_colors.append(_hlsc)
        i += step

    return hls_colors


def ncolors(num):
    rgb_colors = []
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors.append([r, g, b])

    return rgb_colors








def getcluster():
    c = []

    c1 = [0,1,2,3,4,5,6,7,8,20,22,23,24,25,26,27,41,42,43,50]
    c2 = [19,45,51]
    c3 = [9,10,11,12,13,21,28,29,30,31,32,33,34,35,46,47]
    c4 = [14,15,16,17,18,36,37,38,39,40,44,48,49]

    c.append(c1)
    c.append(c2)
    c.append(c3)
    c.append(c4)
    return c
parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=24, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=500,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=5,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-4)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1121,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

torch.manual_seed(args.seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
winlength = 20
root = './data/mnist'
batch_size = args.batch_size
n_classes = 18
input_channels = 52
seq_length = int( input_channels*winlength/ input_channels)
epochs = args.epochs
steps = 0
c =getcluster()

train_data,train_label,train_loader =readFile(IS_TRAIN='Train', batch_size=args.batch_size, window=True,windowl=winlength)
test_data, test_label,test_loader =readFile(IS_TRAIN='Test', batch_size=args.batch_size, window=True,windowl=winlength)

channel_sizes = [args.nhid] * args.levels

kernel_size = args.ksize
model = TCN(len(c[0]),len(c[1]),len(c[2]),len(c[3]), n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)

if args.cuda:
    model.cuda()

lr = args.lr
opti = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
lr_s = StepLR(opti, step_size=10, gamma=0.99)


class NMTCritierion(nn.Module):
    """
    TODO:
    1. Add label smoothing
    """

    def __init__(self, label_smoothing=0.1):
        super(NMTCritierion, self).__init__()
        self.label_smoothing = label_smoothing
        self.LogSoftmax = nn.LogSoftmax()

        if label_smoothing > 0:
            self.criterion = nn.KLDivLoss(reduction='sum' )
        else:
            self.criterion = nn.NLLLoss(reduction='sum' , ignore_index=100000)
        self.confidence = 1.0 - label_smoothing

    def _smooth_label(self, num_tokens):
        # When label smoothing is turned on,
        # KL-divergence between q_{smoothed ground truth prob.}(w)
        # and p_{prob. computed by model}(w) is minimized.
        # If label smoothing value is set to zero, the loss
        # is equivalent to NLLLoss or CrossEntropyLoss.
        # All non-true labels are uniformly set to low-confidence.
        one_hot = torch.randn(1, num_tokens)
        one_hot.fill_(self.label_smoothing / (num_tokens - 1))
        return one_hot

    def _bottle(self, v):
        return v.view(-1, v.size(2))

    def forward(self, dec_outs, labels):
        scores = F.log_softmax(dec_outs,dim=1)
        num_tokens = scores.size(-1)

        # conduct label_smoothing module
        gtruth = labels.view(-1)
        if self.confidence < 1:
            tdata = gtruth.detach()
            one_hot = self._smooth_label(num_tokens)  # Do label smoothing, shape is [M]
            if labels.is_cuda:
                one_hot = one_hot.cuda()
            tmp_ = one_hot.repeat(gtruth.size(0), 1)  # [N, M]
            tmp_.scatter_(1, tdata.unsqueeze(1), self.confidence)  # after tdata.unsqueeze(1) , tdata shape is [N,1]
            gtruth = tmp_.detach()
        loss = self.criterion(scores, gtruth)
        return loss

def train(ep):
    RHO = 0.5
    rho = torch.FloatTensor([RHO for _ in range(100)]).unsqueeze(0)
    global steps
    train_loss = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        lr_s.step()

        data_raw = data.reshape(-1,winlength,52)
        if args.cuda: data, target = data.cuda(), target.cuda()
        data_0 = data_raw[:,:, c[0]].permute(0,2,1).cuda()
        data_1 = data_raw[:,:, c[1]].permute(0,2,1).cuda()
        data_2 = data_raw[:,:, c[2]].permute(0,2,1).cuda()
        data_3 = data_raw[:,:, c[3]].permute(0,2,1).cuda()


        cross_E = nn.CrossEntropyLoss()
        LOS = NMTCritierion()
        opti.zero_grad()
        output = model(data_0,data_1,data_2,data_3)

        loss = LOS(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opti.step()
        train_loss += loss
        steps += seq_length
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tSteps: {}'.format(
                ep, batch_idx * batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.item()/args.log_interval, steps))
            train_loss = 0

def classpred(num_class, target, pred):
    num_pred = [0]*num_class
    num_target = [0]*num_class

    for i in range(target.shape[0]):
        if target[i] == pred[i]:
            num_pred[target[i]]+=1
        num_target[target[i]]+=1
    return num_pred, num_target

train_acc = []
test_acc = []
def test():
    model.eval()
    num_class = n_classes
    test_loss_t= 0
    target_total = [0]*num_class
    pred_total = [0]*num_class
    pred_per = [0]*num_class
    correct = 0
    test_loss =0
    correct_l = 0
    with torch.no_grad():

        data, target = torch.tensor(test_data).cuda().float(), torch.tensor(test_label).long().cuda()

        data_raw = data.reshape(-1, winlength, 52)
        data_0 = data_raw[:,:, c[0]].permute(0,2,1).cuda()
        data_1 = data_raw[:,:, c[1]].permute(0,2,1).cuda()
        data_2 = data_raw[:,:, c[2]].permute(0,2,1).cuda()
        data_3 = data_raw[:,:, c[3]].permute(0,2,1).cuda()

        inner_o , output = model.testforward(data_0,data_1,data_2,data_3)
        target_cpu = target.detach().cpu().numpy()
        target_cpu = np.where(target_cpu > 2,target_cpu+1,target_cpu)
        target_cpu = np.where(target_cpu > 8, target_cpu + 1, target_cpu)
        target_cpu = np.where(target_cpu > 14, target_cpu + 1, target_cpu)



        inner_o = inner_o.detach().cpu().numpy()




        # plt.figure(figsize=(12, 6))
        # plt.figure()
        # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=target_cpu.flatten(), label=, cmap='tab20')
        # plt.legend()
        # plt.show()

        # a = output.detach().cpu()
        # X_tsne = data_tsne = TSNE(n_components=2, learning_rate=200.0, perplexity=30).fit_transform(
        #     a)
        # plt.figure(figsize=(12, 6))
        # # plt.subplot(121)
        # plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=target_cpu.flatten(), cmap='Set2')
        # plt.colorbar()
        # plt.show()
        target_hot = torch.zeros(data_1.shape[0], n_classes).scatter_(1, target.detach().cpu().reshape(-1, 1), 1)
        cross_E = nn.CrossEntropyLoss()
        test_loss = cross_E(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        pred_num, tar_num = classpred(num_class, target, pred)
        # pred_total = list(map(lambda x: x[0] + x[1], zip(pred_total, pred_num)))
        # target_total = list(map(lambda x: x[0] + x[1], zip(target_total, tar_num)))

        correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        acc_t =  float(correct) / float(len(test_loader.dataset))
        test_acc.append(acc_t)
        pred_per = list(map(lambda x: x[0] / x[1], zip(pred_num, tar_num)))


        data, target = torch.tensor(train_data).float().cuda(), torch.tensor(train_label).long().cuda()
        data_raw = data.reshape(-1, winlength, 52)
        data_0 = data_raw[:,:, c[0]].permute(0,2,1).cuda()
        data_1 = data_raw[:,:, c[1]].permute(0,2,1).cuda()
        data_2 = data_raw[:,:, c[2]].permute(0,2,1).cuda()
        data_3 = data_raw[:,:, c[3]].permute(0,2,1).cuda()
        # data_4 = data_raw[:,:, c[4]].permute(0,2,1).cuda()
        # data_0 = data_raw[:,:, c[0]].cuda()
        # data_1 = data_raw[:,:, c[1]].cuda()
        # data_2 = data_raw[:,:, c[2]].cuda()
        # data_3 = data_raw[:,:, c[3]].cuda()
        # data_4 = data_raw[:,:, c[4]].cuda()
        # data = data.view(-1, input_channels, seq_length)
        # if args.permute:
        #     data = data[:, :, permute]
        # data, target = Variable(data, volatile=True), Variable(target)
        output_t = model.forward(data_0,data_1,data_2,data_3)
        target_hot_t = torch.zeros(data_1.shape[0], n_classes).scatter_(1, target.detach().cpu().reshape(-1, 1), 1).cuda().long()
        test_loss_t = cross_E(output_t, target)

        pred = output_t.data.max(1, keepdim=True)[1]
        correct_l = pred.eq(target.data.view_as(pred)).cpu().sum()
        acc_train =  float(correct_l) / float(len(train_loader.dataset))
        train_acc.append(acc_train)
        # test_loss_t/= len(train_loader.dataset)
        # test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
             100*correct / len(test_loader.dataset)))
        print('every test class precent:')
        print(pred_per)
        print('\nTRAIN set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss_t, correct_l, len(train_loader.dataset),
             100.*correct_l / len(train_loader.dataset)))

        if  epoch==20:
            X_tsne = TSNE(n_components=2, learning_rate=100).fit_transform(inner_o)
            # category_to_color = {0: 'lightgreen', 1: 'lawngreen', 2: 'limegreen', 4: 'darkgreen',5: 'darkorange',
            #                      6: 'darkslategrey', 7: 'teal', 8: 'coral', 10: 'gold', 11: 'firebrick', 12: 'sandybrown',
            #                      13: 'deepskyblue', 14: 'indigo', 16: 'crimson', 17: 'royalblue', 18: 'turquoise',
            #                      19: 'orchid', 20: 'hotpink'}
            # category_to_label = {0: '0', 1: '1', 2: '2', 4: '4',5: '5',
            #                      6: '6', 7: '7', 8: '8', 10: '10', 11: '11', 12: '12',
            #                      13: '13', 14: '14', 16: '16', 17: '17', 18: '18',
            #                      19: '19', 20: '20'}

            fig, ax = plt.subplots(figsize=(10, 8))
            N = 7  # number of colors to extract from each of the base_cmaps below

            base_cmaps = ['tab20', 'tab20b', 'tab20c']
            #
            # n_base = len(base_cmaps)
            # # we go from 0.2 to 0.8 below to avoid having several whites and blacks in the resulting cmaps
            colors = np.concatenate([plt.get_cmap(name)(np.linspace(0.2, 0.8, N)) for name in base_cmaps])
            cmap = matplotlib.colors.ListedColormap(colors)
            size = 81
            sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=size, c=target_cpu.flatten(), cmap=cmap, edgecolors='none')

            lp = lambda i: plt.plot([], color=sc.cmap(sc.norm(i)), ms=np.sqrt(size), mec="none",
                                    label="Fault {:g}".format(i), ls="", marker="o")[0]
            handles = [lp(i) for i in np.unique(target_cpu.flatten())]
            plt.legend(handles=handles, bbox_to_anchor=(1.05, 1.0))

            fig.subplots_adjust(right=0.8)
            plt.show()
            plot_confusion_matrix(np.argmax(target_hot, axis=1), np.argmax(output.detach().cpu(), axis=1))
            # x = list(range(1,epoch+1))
            # plt.figure(figsize=(5, 5))
            # plt.plot(x, test_acc, color='darkorange', markerfacecolor='darkorange',marker='o',label='Validation accuracy')
            # plt.plot(x, train_acc, color='cornflowerblue',  markerfacecolor='cornflowerblue',marker='o', label='Training accuracy')
            # # plt.grid()
            # plt.xticks(x)
            # plt.legend()
            # plt.xlabel('Epochs')
            # plt.ylabel('Accuracy')
            # plt.show()
        return test_loss



if __name__ == "__main__":

    if not os.path.exists('./pretrain_model_MBTCN_LB.pk'):
        for epoch in range(1, 21):
            train(epoch)
            # if epoch %10 ==0:
            test()
            # if epoch % 1 == 0:
            #     lr /= 2
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr
        torch.save(model.state_dict(), './pretrain_model_MBTCN_LB.pk')
    else:
        epoch = 20
        model.load_state_dict(torch.load('./pretrain_model_MBTCN_LB.pk'))
        test()