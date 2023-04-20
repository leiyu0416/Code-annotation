# Code-annotation

Before starting a Pytorch training project, we need to initialize some things.

Firstly, initialize the weights of the model.These values determine the importance of each input signal when transmitted to the next layer. During the training process, the model continuously adjusts its weights to better fit the data.

Secondly, initialize the bias of the model.During the training process, the model will continuously adjust the bias to better fit the data. Initializing bias can help the model converge faster. If initialization is not carried out, the initial value of the bias may be too large or too small, resulting in poor performance of the model in the early stages of training. By properly initializing the bias, the model can better fit the data in the early stages of training.

Finally, initialize the batch normalization.Initializing the batch normalization layer can help the model converge faster.

Basic steps for preparing optimization algorithms in PyTorch

 1.Define the model and move it to the appropriate device (such as GPU).
 2.Select a loss function to calculate the difference between the model output and the target.
 3.Select an optimizer, such as torch. optim. SGD or torch. optim. Adam, and pass the model parameters to it.
 4.Define a learning rate scheduler for dynamically adjusting the learning rate.
 5.Loading dataset: When training the model, it is necessary to load the training and validation sets, and perform preprocessing, data augmentation, and other operations on the data.
 6.Defining a training cycle: When training a model, it is necessary to define a training cycle.

### Code-annotation

```import argparse
import time
import json
import os

from tqdm import tqdm
from models import *
# from efficientnet_pytorch import EfficientNet
from torch import nn
from torch import optim
# from torch.optim.lr_scheduler import *
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from tools import warmup_lr


# 初始化参数
def get_args():
    """在下面初始化你的参数.
    """
    parser = argparse.ArgumentParser(description='基于Pytorch实现的分类任务')
    parser.add_argument('--config', type=str, help='Path to the configuration file')
    # exp
    parser.add_argument('--time_exp_start', type=str,
                        default=time.strftime('%m-%d-%H-%M', time.localtime(time.time())))
    parser.add_argument('--train_dir', type=str, default='data/train')
    parser.add_argument('--val_dir', type=str, default='data/val')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--save_station', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--is_mps', type=bool, default=False)
    parser.add_argument('--is_cuda', type=bool, default=False)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--test_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.001)

    # dataset
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--data_mean', type=tuple, default=[.5, .5, .5])
    parser.add_argument('--data_std', type=tuple, default=[.5, .5, .5])

    # model
    parser.add_argument('--model', type=str, default='ResNet18',
                        choices=[
                            'ResNet18',
                            'ResNet34',
                            'ResNet50',
                            'ResNet18RandomEncoder',
                        ])
    # scheduler
    parser.add_argument('--warmup_epoch', type=int, default=1)
    # 通过json记录参数配置
    args = parser.parse_args()
    args.directory = 'dictionary/%s/Hi%s/' % (args.model, args.time_exp_start)
    log_file = os.path.join(args.directory, 'log.json')
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
    with open(log_file, 'w') as log:
        json.dump(vars(args), log, indent=4)
    # 返回参数集
    return args
class Worker:
    def __init__(self, args):
        self.opt = args
        # 判定设备
        self.device = torch.device('cuda:0' if args.is_cuda else 'cpu') # 用于设置 PyTorch 计算所使用的设备，用于判断是否使用cuda,选择在GPU或CPU上运行模型
                                                                        # torch.device用于指定张量在哪个设备上运行
                                                                        # 'cuda:0'表示在第一个GPU上运行,'cpu'表示在 CPU 上运行
        kwargs = {
            'num_workers': args.num_workers,  # num_workers是一个参数，它指定了用于数据加载的子进程数。这个参数可以加快数据加载的速度。
                                              # args.num_workers是一个整数，表示要使用的子进程数。
            'pin_memory': True,
        } if args.is_cuda else {}
        # 载入数据
        train_dataset = datasets.ImageFolder(
            args.train_dir,  # 表示训练数据所在的目录，这个参数可以用来指定训练数据的位置，以便在训练模型时加载数据。
                             # args是一个命名空间，它包含了从命令行传递给程序的参数。
                             # train_dir是其中的一个参数，表示训练数据所在的目录。
        transform=transforms.Compose([
                transforms.RandomResizedCrop(256),  # 它用于对图像进行随机裁剪和缩放。表示裁剪后图像的大小为256。
                transforms.ToTensor()  # 是一个数据预处理方法，它用于将图像数据转换为PyTorch张量。
                                       # 将图像数据从PIL图像或NumPy数组转换为PyTorch张量，这样可以让模型更容易处理数据。
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
        )
        val_dataset = datasets.ImageFolder(
            args.val_dir,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(256),
                transforms.ToTensor()
                # transforms.Normalize(opt.data_mean, opt.data_std)
            ])
        )
        self.train_loader = DataLoader(
            dataset=train_dataset,  # 是DataLoader类的一个参数，指定了要使用的数据集。
            batch_size=args.batch_size,  # 是DataLoader类的一个参数，指定了每个批次的数据量
                                         # args.batch_size是一个整数，表示每个批次的数据量。

            shuffle=True,  # 是DataLoader类的一个参数，表示在每个epoch开始时打乱数据顺序
                           # 这样可以防止模型在训练过程中过拟合数据
            **kwargs
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=args.test_batch_size,
            shuffle=False,
            **kwargs
        )
        # 挑选神经网络、参数初始化
        net = None
        if args.model == 'ResNet18':
            net = ResNet18(num_cls=args.num_classes)
        elif args.model == 'ResNet34':
            net = ResNet34(num_cls=args.num_classes)
        elif args.model == 'ResNet50':
            net = ResNet50(num_cls=args.num_classes)
        elif args.model == 'ResNet18RandomEncoder':
            net = ResNet18RandomEncoder(num_cls=args.num_classes)
        assert net is not None

        self.model = net.to(self.device)  # 用于将模型移动到指定的设备上。这样，根据硬件配置选择在GPU或CPU上运行模型。
                                          # self.model是一个模型对象
                                          # net是一个模型类的实例，to(self.device)方法用于将模型移动到 self.device 指定的设备上。
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),  # 在训练模型时更新参数
                                      # self.model.parameters()是一个方法，它返回模型的所有参数。
                                      # self.model 是一个模型对象，parameters()方法返回一个迭代器，它包含了模型的所有参数。
            lr=args.lr  # 是优化器的一个参数，它指定了学习率。
                        # args.lr是一个浮点数，表示学习率。
        )
        # 损失函数
        self.loss_function = nn.CrossEntropyLoss()  # 用于创建一个损失函数对象，它用于计算模型的损失，在训练模型时，最小化损失函数的值，以提高模型的性能。
                                                    # self.loss_function是一个损失函数对象，nn.CrossEntropyLoss()是一个损失函数类。
        # warm up 学习率调整部分
        self.per_epoch_size = len(train_dataset) // args.batch_size
        self.warmup_step = args.warmup_epoch * self.per_epoch_size
        self.max_iter = args.epochs * self.per_epoch_size
        self.global_step = 0

    def train(self, epoch):
        self.model.train()
        bar = tqdm(enumerate(self.train_loader))
        for batch_idx, (data, target) in bar:
            self.global_step += 1
            data, target = data.to(self.device), target.to(self.device)

            # 训练中...
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss_function(output, target)
            loss.backward()
            self.optimizer.step()
            lr = warmup_lr.adjust_learning_rate_cosine(
                self.optimizer, global_step=self.global_step,
                learning_rate_base=self.opt.lr,
                total_steps=self.max_iter,
                warmup_steps=self.warmup_step
            )

            # 更新进度条
            bar.set_description(
                'train epoch {} >> [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr: {:.6f} >> '.format(
                    epoch,
                    batch_idx * len(data),
                    len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader),
                    loss.item(),
                    lr
                )
            )
        bar.close()

    def val(self):
        self.model.eval()
        validating_loss = 0
        num_correct = 0
        with torch.no_grad():
            bar = tqdm(self.val_loader)
            for data, target in bar:
                # 测试中...
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                validating_loss += self.loss_function(output, target).item()  # 累加 batch loss
                pred = output.argmax(dim=1, keepdim=True)  # 获取最大概率神经元下标
                num_correct += pred.eq(target.view_as(pred)).sum().item()
            bar.close()

        # 打印验证结果
        validating_loss /= len(self.val_loader)
        print('val >> Average loss: {:.4f}, Accuracy: {}/{} ({:.03f}%)\n'.format(
            validating_loss,
            num_correct,
            len(self.val_loader.dataset),
            100. * num_correct / len(self.val_loader.dataset))
        )

        # 返回重要信息，用于生成模型保存命名
        return 100. * num_correct / len(self.val_loader.dataset), validating_loss


if __name__ == '__main__':
    # 初始化
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed(0)
    args = get_args()
    worker = Worker(args=args)  # 用于创建一个 Worker 类的实例。
                                # worker是一个Worker类的实例，Worker(args=args)是一个类构造函数，它用于创建Worker类的实例。

    # 训练与验证
    for epoch in range(1, args.epochs + 1):
        worker.train(epoch)
        val_acc, val_loss = worker.val()
        if epoch > args.save_station:
            save_dir = args.directory + '%s-epochs-%d-model-val-acc-%.3f-loss-%.6f.pt' \
                       % (args.model, epoch, val_acc, val_loss)
            torch.save(worker.model, save_dir)
