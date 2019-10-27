import torch
import os
import torch.nn as nn
import torchvision.models as models
from models.vgg import VGG
from utils.compute_average import AverageMeter
from models.transform_model import integrated_kernel

class DPP(object):
    def __init__(self, args):
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.lr = args.lr
        self.epochs = args.epochs
        self.save_dir = './' + args.save_dir #later change
        if(os.path.exists(self.save_dir) == False):
            os.mkdir(self.save_dir)
            
        if(args.model == 'vgg16'):
            self.model = VGG('VGG16',0)
            self.optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, self.model.parameters()), lr = self.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            self.model = torch.nn.DataParallel(self.model)
            self.model.cuda()
        elif(args.model == 'dpp_vgg16'):
            self.model = integrated_kernel(args)
            self.optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, self.model.parameters()), lr = self.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
        #Parallel 
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('The number of parametrs of models is', num_params) 

        if(args.save_load):
            location = args.save_location
            print("locaton",location)
            checkpoint = torch.load(location)
            self.model.load_state_dict(checkpoint['state_dict'])
            
    def train(self, train_loader, test_loader, graph):
        #Declaration Model
        self.model.train()
        best_prec = 0
        losses = AverageMeter()
        top1 = AverageMeter()
        for epoch in range(self.epochs):
            #Test Accuarcy
            #self.adjust_learning_rate(epoch)
            for k, (inputs, target) in enumerate(train_loader):
                target = target.cuda(async=True)
                input_var = inputs.cuda()
                target_var = target
                output = self.model(input_var)
                loss = self.criterion(output, target_var)
                #Compute gradient and Do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                #Measure accuracy and record loss
                prec1 = self.accuracy(output.data, target)[0]
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
            
            graph.train_loss(losses.avg, epoch,'train_loss')
            graph.train_acc(top1.avg, epoch,'train_acc')
            prec = self.test(test_loader, epoch, graph)
            if(prec > best_prec):
                print("Acc", prec)
                best_prec = prec
                self.save_checkpoint({
                    'best_prec1':best_prec,
                    'state_dict':self.model.state_dict(),
                    }, filename=os.path.join(self.save_dir, 'checkpoint_{}.tar'.format(epoch)))
        
    def test(self, test_loader, epoch, test_graph):
        self.model.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        for k, (inputs, target) in enumerate(test_loader):
            target = target.cuda()
            inputs = inputs.cuda()
            #Calculate each model
            #Compute gradient and Do SGD step
            output = self.model(inputs)
            loss = self.criterion(output, target)
            #Measure accuracy and record loss
            prec1 = self.accuracy(output.data, target)[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
        test_graph.test_loss(losses.avg, epoch, 'test_loss')
        test_graph.test_acc(top1.avg, epoch, 'test_acc')
        return top1.avg

    def accuracy(self, output, target, topk=(1,)):
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
    def adjust_learning_rate(self, epoch):
        self.lr = self.lr * (0.1 ** (epoch // 90))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
