#malevis
import os
import sys
import time
import argparse
import gc
import random
import copy
from scipy.sparse import data
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torchvision.datasets as datasets
import shutil
from torchvision.transforms import ToPILImage
from collections import Counter
from resnet import *
from loss import *
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

start_time = time.time()

# """
#输出保存到本地
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

current_directory = os.path.dirname(__file__)
root_1 = current_directory + "/log/res32_mle_50_n.log"
root_2 = current_directory + "/log/res32_mle_50_n.log_file"


sys.stdout = Logger(root_1, sys.stdout)
sys.stderr = Logger(root_2, sys.stderr)
# """

parser = argparse.ArgumentParser(description='Malevis Classification')
parser.add_argument('--dataset', default='Malevis', type=str,
                    help='dataset (Malimg or Malevis[default])')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_classes', type=int, default=26)
parser.add_argument('--num_meta', type=int, default=10,
                    help='The number of meta data for each class.')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--imb_factor', type=float, default=0.02)
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--split', type=int, default=1000)
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    help='print frequency (default: 10)')
# 参数
parser.add_argument('--alpha', default=0.75, type=float, help='[0.25, 0.5, 0.75, 1.0,1.5]')
parser.add_argument('--beta', default=1.0, type=float, help='[0.25, 0.5, 0.75, 1.0,1.5]')
parser.add_argument('--head', default=5, type=int, help='[10, 20, 30, 40]')

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--save_name', default='name', type=str)
parser.add_argument('--idx', default='0', type=str)
parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
parser.add_argument('--rand_number', default=42, type=int, help='fix random number for data sampling')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model', default='resnet50', type=str,
                    help='Model to be trained. '
                         'Select from resnet{18, 34, 50, 101, 152} / resnext{50_32x4d, 101_32x8d} / '
                         'densenet{121, 169, 201, 265}')
parser.add_argument('--pre-train', default='', type=str, metavar='PATH',
                    help='path to the pre-train model (default: none)')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

args = parser.parse_args()
for arg in vars(args):
    print("{}={}".format(arg, getattr(args, arg)))
# 清除pytorch无用缓存
gc.collect()
torch.cuda.empty_cache()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
kwargs = {'num_workers': 0, 'pin_memory': False}
use_cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)  ##
#   cudnn.benchmark = False             ##
#   torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if use_cuda else "cpu")
best_prec1 = 0
best_prec1_train = 0
best_Precision1 = 0
best_Precision1_mi = 0
best_Recall1 = 0
best_Recall1_mi = 0
best_F11 = 0
best_F11_mi = 0
kg = torch.zeros(26, 26)
feature_mean = torch.zeros(26, 64)  # 64
print(feature_mean.shape)


# Data loading code
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.ToPILImage(),
    transforms.RandomCrop(300),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
])
# 读取数据集
# 数据集根目录
current_directory = os.path.dirname(__file__)
dataset_root = current_directory + "/malevis_train_val_300x300"

# 创建ImageFolder对象
train_dataset = ImageFolder(root=dataset_root + "/train_50", transform=transform_train)
val_dataset = ImageFolder(root=dataset_root + "/val", transform=transform_val)

# 统计每个类别的图像数量
class_counts = Counter(train_dataset.targets)

# 存储每个类别的图像数量
img_num_list = []
for class_idx in sorted(class_counts.keys()):
    class_name = train_dataset.classes[class_idx]
    count = class_counts[class_idx]
    img_num_list.append(count)

print('img_num_list:')
print(img_num_list)
print(len(img_num_list))
# args.img_num_list = img_num_list

#Val

# 统计每个类别的图像数量
class_counts_val = Counter(val_dataset.targets)

# 存储每个类别的图像数量
img_num_list_val = []
for class_idx_val in sorted(class_counts_val.keys()):
    class_name_val = val_dataset.classes[class_idx_val]
    count_val = class_counts_val[class_idx_val]
    img_num_list_val.append(count_val)

# 打印每个类别的图像数量
print('img_num_list_val:')
print(img_num_list_val)
print(len(img_num_list_val))

train_sampler = None

imbalanced_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
# 获取一个批次的图像数据
images, labels = next(iter(imbalanced_train_loader))

# 查看图像数据的形状
batch_size, channels, height, width = images.shape

print("图像维度：{} x {} x {} x {}".format(batch_size, channels, height, width))  # 100 3*224*224

# """
# 计算用于样本加权的权重，并将其存储在 weights 张量中。这些权重可以在训练过程中应用于损失函数，用于解决类别不平衡的问题。
rho = 0.9999
effective_num = 1.0 - np.power(rho, img_num_list)
per_cls_weights = (1.0 - rho) / np.array(effective_num)
per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(img_num_list)
per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
weights = torch.tensor(per_cls_weights).float()
print('weights')
print(weights)


def main():
    global args, best_prec1, best_prec1_train, best_Recall1, best_Recall1_mi, best_Precision1, best_Precision1_mi, best_F11, best_F11_mi, kg, feature_mean
    args = parser.parse_args()

    model = build_model()

    feature_num = 64  # 特征数量

    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                checkpoint = torch.load(args.resume)
                # loc = 'cuda:{}'.format(args.gpu)
                # checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            print(args.start_epoch)

            best_prec1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                print('ok')
                # best_acc1 = best_acc1.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer_a.load_state_dict(checkpoint['optimizer'])
            criterion = checkpoint['criterion']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.deterministic = True
    if not cudnn.deterministic:
        exit()
    cudnn.benchmark = False

    # loss.py
    criterion = FSDA_CE(feature_num, 26, cls_num_list=img_num_list,
                         max_m=0.5, s=30)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer_a, epoch)

        alpha = args.alpha * float(epoch) / float(args.epochs)
        beta = args.beta * float(epoch) / float(args.epochs)

        if epoch == 100:
            run_time_sec = time.time() - start_time
            hours = run_time_sec // 3600
            minutes = (run_time_sec % 3600) // 60
            seconds = run_time_sec % 60
            output_str = f"{hours}时{minutes}分{seconds}秒"
            print("不加FSDA模块的程序运行时间：", output_str)
            print("不加FSDA模块的best评价指标：")
            print('Best accuracy:', best_prec1, 'Best Recall:', best_Recall1,
                  'Best Precision:', best_Precision1, 'Best F1:', best_F11)
            print('微平均: Best Recall:', best_Recall1_mi,
                  'Best Precision:', best_Precision1_mi, 'Best F1:', best_F11_mi)
            best_prec1 = 0
            best_Recall1 = 0
            best_Precision1 = 0
            best_F11 = 0
            best_Recall1_mi = 0
            best_Precision1_mi = 0
            best_F11_mi = 0

        if epoch < 100:
            train(imbalanced_train_loader, model, optimizer_a, epoch)
            prec1_train, _, _, _, _, _, _, preds_train, labels_train, cf_normalized = validate(imbalanced_train_loader, model,
                                                                             nn.CrossEntropyLoss().cuda(), epoch)

            is_best_train = prec1_train > best_prec1_train
            if is_best_train:
                # print(cf_normalized)
                print('Change cf.')
                kg = cf_normalized
                # torch.save(cf_normalized,'cifar100_im100_kg.pkl')
            best_prec1_train = max(prec1_train, best_prec1_train)

        else:
            if epoch == 100:  # 160
                # obtain kg and prototype
                kg = torch.tensor(kg).cuda()
                kg = kg.to(torch.float32).cuda()
                feature_mean = get_feature_mean(imbalanced_train_loader, model, len(img_num_list))
                feature_mean = feature_mean.to(torch.float32).cuda()
                # print(feature_mean.shape)

                # use kg to get reasoning prototype
                out_new = torch.matmul(kg, feature_mean)
                out_new = out_new - feature_mean

            if True:
                print('Freezing feature weights except for self attention weights (if exist).')
                for param_name, param in model.named_parameters():
                    if 'linear' not in param_name:
                        param.requires_grad = False
                    print('  | ', param_name, param.requires_grad)

            train_FSDA(imbalanced_train_loader, model, optimizer_a, epoch, criterion, alpha, kg, beta, out_new,
                        feature_mean, args)
        prec1, Recall1_mi, Precision1_mi, F11_mi, Recall1, Precision1, F11, preds, labels, cf_normalized = validate(test_loader, model, nn.CrossEntropyLoss().cuda(), epoch)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        best_Recall1 = max(Recall1, best_Recall1)
        best_Precision1 = max(Precision1, best_Precision1)
        best_F11 = max(F11, best_F11)
        best_Recall1_mi = max(Recall1_mi, best_Recall1_mi)
        best_Precision1_mi = max(Precision1_mi, best_Precision1_mi)
        best_F11_mi = max(F11_mi, best_F11_mi)

        print('Best accuracy: ', best_prec1)

    print("加上FSDA模块的best评价指标:")
    print('Best accuracy:', best_prec1, 'Best Recall:', best_Recall1, 'Best Precision:', best_Precision1, 'Best F1:', best_F11)
    print('微平均: Best Recall:', best_Recall1_mi,
          'Best Precision:', best_Precision1_mi, 'Best F1:', best_F11_mi)


def train(train_loader, model, optimizer_a, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)
        _, y_f = model(input_var, target_var, 0, 0, 0)
        del _
        cost_w = F.cross_entropy(y_f, target_var, reduce=False)
        l_f = torch.mean(cost_w)
        prec_train = accuracy(y_f.data, target_var.data, topk=(1,))[0]

        losses.update(l_f.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        optimizer_a.zero_grad()
        l_f.backward()
        optimizer_a.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses, top1=top1))


def train_FSDA(train_loader, model, optimizer_a, epoch, criterion, alpha, kg, beta, out_new, feature_mean, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    model.train()
    kg = kg.cuda()

    for i, (input, target) in enumerate(train_loader):

        input_var = to_var(input, requires_grad=False)
        target_var = to_var(target, requires_grad=False)

        cv = criterion.get_cv()
        cv_var = to_var(cv)
        # reasoning prototype and CoVariance
        features, predicts = model(input_var, target_var, out_new, True, alpha)

        cls_loss = criterion(model.linear, features, predicts, target_var, alpha, weights, cv_var, "update", kg,
                             out_new, feature_mean, beta, args.head)

        prec_train = accuracy(predicts.data, target_var.data, topk=(1,))[0]

        losses.update(cls_loss.item(), input.size(0))
        top1.update(prec_train.item(), input.size(0))

        optimizer_a.zero_grad()
        cls_loss.backward()
        optimizer_a.step()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(train_loader),
                loss=losses, top1=top1))


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.eval()

    true_labels = []
    preds = []
    all_y_true = []
    all_y_pred = []

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        # print("This is a test.")
        with torch.no_grad():
            _, output = model(input_var, target_var, 0, 0, 0)

        output_numpy = output.data.cpu().numpy()
        preds_output = list(output_numpy.argmax(axis=1))

        true_labels += list(target_var.data.cpu().numpy())
        preds += preds_output

        prec1 = accuracy(output.data, target, topk=(1,))[0]

        all_y_true.append(target.cpu())
        all_y_pred.append(preds_output)

        top1.update(prec1.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    # 将列表中的预测结果和目标标签合并为一个numpy数组
    y_true = np.concatenate(all_y_true)
    y_pred = np.concatenate(all_y_pred)

    # 计算召回率
    recall = recall_score(y_true, y_pred, average='macro') * 100 # 宏平均
    recall_mi = recall_score(y_true, y_pred, average='micro') * 100 #微平均
    # 计算精确率
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    precision_mi = precision_score(y_true, y_pred, average='micro', zero_division=0) * 100
    # 计算F1值
    f1 = f1_score(y_true, y_pred, average='macro') * 100
    f1_mi = f1_score(y_true, y_pred, average='micro') * 100

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    cf = confusion_matrix(true_labels, preds).astype(float)  # 69,69
    cf_normalized = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
    cf_normalized = np.round(cf_normalized, 2)

    # torch.save(cf_normalized,'kg_cifar100_im200.pkl')
    return top1.avg, recall_mi, precision_mi, f1_mi, recall, precision, f1, preds, true_labels, cf_normalized


def build_model():
    # model = ResNet32(args.dataset == 'cifar10' and 10 or 100)
    model = ResNet32(26)
    print("The model is res32.")

    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True
        print(1)
    return model


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    if epoch < 5:
        lr = args.lr * float(epoch + 1) / 5
    else:
        lr = args.lr * ((0.1 ** int(epoch >= 40)) * (0.1 ** int(epoch >= 60)) * (0.1 ** int(epoch >= 80)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(args, state, is_best):
    path = 'checkpoint/' + args.idx + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    filename = path + args.save_name + '_ckpt.pth.tar'
    if is_best:
        torch.save(state, filename)


def get_feature_mean(imbalanced_train_loader, model, class_num):
    model.eval()
    feature_mean_end = torch.zeros(class_num, 64)
    with torch.no_grad():
        for i, (input, target) in enumerate(imbalanced_train_loader):
            target = target.cuda()
            input = input.cuda()
            input_var = to_var(input, requires_grad=False)
            target_var = to_var(target, requires_grad=False)

            features, output = model(input_var, target_var, 0, 0, 0)
            features = features.cpu().data.numpy()

            for out, label in zip(features, target):
                feature_mean_end[label] = feature_mean_end[label] + out

        img_num_list_tensor = torch.tensor(img_num_list).unsqueeze(1)

        feature_mean_end = torch.div(feature_mean_end, img_num_list_tensor)

        return feature_mean_end


if __name__ == '__main__':
    main()

    end_time = time.time()
    run_time_sec = end_time - start_time
    hours = run_time_sec // 3600
    minutes = (run_time_sec % 3600) // 60
    seconds = run_time_sec % 60
    output_str = f"{hours}时{minutes}分{seconds}秒"
    print("程序运行时间：", output_str)

# """