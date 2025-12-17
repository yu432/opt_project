import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import argparse
import time
# ====================== SEED ======================

def set_seed(seed: int = 42):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ====================== Model Definitions ======================

class BasicBlockWRN(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlockWRN, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    
    def forward(self, x):
        if not self.equalInOut:
            x = self.bn1(x)
            x = self.relu1(x)
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=16, widen_factor=4, num_classes=100, dropRate=0.0):
        super(WideResNet, self).__init__()
        
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        if (depth - 4) % 6 != 0:
            raise RuntimeError(f"Depth = {depth} does not follow 6*n + 4, n \in N")
        n = (depth - 4) / 6
        block = BasicBlockWRN
        
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.avg_pool2d(out, 8, 1, 0)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

class BasicBlockResNet(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetCIFAR(nn.Module):
    def __init__(self, depth=50, num_classes=100):
        super(ResNetCIFAR, self).__init__()
        self.in_planes = 64
        
        blocks_layers = {
            18: (BasicBlockResNet, [2, 2, 2, 2]),
            34: (BasicBlockResNet, [3, 4, 6, 3]),
            50: (Bottleneck, [3, 4, 6, 3]),
            101: (Bottleneck, [3, 4, 23, 3]),
            152: (Bottleneck, [3, 8, 36, 3]),
        }
        
        if depth not in blocks_layers:
            raise RuntimeError(f"Model depth {depth} not supported for Resnet.")
        
        block, num_blocks = blocks_layers[depth]
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# ====================== Optimizer Definitions ======================

def smooth_crossentropy(predictions, targets, smoothing=0.1):
    n_classes = predictions.size(-1)
    one_hot = torch.zeros_like(predictions).scatter_(1, targets.unsqueeze(1), 1)
    one_hot = one_hot * (1 - smoothing) + smoothing / n_classes
    log_prb = F.log_softmax(predictions, dim=1)
    loss = -(one_hot * log_prb).sum(dim=1)
    return loss

class SGD(optim.SGD):
    def __init__(self, params, lr=0.5, momentum=0.9, weight_decay=0.0005):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

class SAM(torch.optim.Optimizer):
    def __init__(self, params, lr=0.5, momentum=0.9, weight_decay=0.0005, rho=0.3):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, rho=rho)
        super(SAM, self).__init__(params, defaults)
        
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['momentum_buffer'] = torch.zeros_like(p).detach()
    
    @torch.no_grad()
    def move_up(self):
        norm = self._grad_norm()
        
        for group in self.param_groups:
            rho = group['rho']
            scale = rho / (norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w.detach().clone()
    
    @torch.no_grad()
    def move_back(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
    
    def step(self, model, inputs, targets):
        predictions = model(inputs)
        loss = smooth_crossentropy(predictions, targets)
        loss.mean().backward()
        
        self.move_up()
        self.zero_grad()
        
        predictions = model(inputs)
        loss = smooth_crossentropy(predictions, targets)
        loss.mean().backward()
        
        self.move_back()
        
        self.SGD_step()
        self.zero_grad()
        
        return loss, predictions
    
    def _grad_norm(self):
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2) for group in self.param_groups for p in group["params"] if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    @torch.no_grad()
    def SGD_step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                d_p = p.grad
                if group['weight_decay'] != 0:
                    d_p = d_p.add(p, alpha=group['weight_decay'])
                
                if group['momentum'] != 0:
                    buf = self.state[p]['momentum_buffer']
                    buf.mul_(group['momentum']).add_(d_p)
                    d_p = buf
                
                p.add_(d_p, alpha=-group['lr'])

class MSAM(torch.optim.Optimizer):
    def __init__(self, params, lr=0.5, momentum=0.9, weight_decay=0.0005, rho=3):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, rho=rho)
        super(MSAM, self).__init__(params, defaults)
        
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['momentum_buffer'] = torch.zeros_like(p).detach()
            group["inverse_norm_buffer"] = 0
    
    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        params_with_grad = []
        d_p_list = []
        momentum_buffer_list = []
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        rho = group['rho']
        lr = group['lr']
        
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                momentum_buffer_list.append(self.state[p]['momentum_buffer'])
        
        for i, param in enumerate(params_with_grad):
            d_p = d_p_list[i]
            if weight_decay != 0:
                d_p.add_(param, alpha=weight_decay)
            
            buf = momentum_buffer_list[i]
            param.add_(buf, alpha=rho*group["inverse_norm_buffer"])
            
            buf.mul_(momentum).add_(d_p)
            param.add_(buf, alpha=-lr)
        
        ascent_norm = torch.norm(
            torch.stack([buf.norm(p=2) for buf in momentum_buffer_list]),
            p=2
        )
        group["inverse_norm_buffer"] = 1/(ascent_norm+1e-12)
        
        for i, param in enumerate(params_with_grad):
            param.sub_(momentum_buffer_list[i], alpha=rho*group["inverse_norm_buffer"])
        
        for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
            self.state[p]['momentum_buffer'] = momentum_buffer
    
    @torch.no_grad()
    def move_up_to_momentumAscent(self):
        for group in self.param_groups:
            for p in group['params']:
                if "momentum_buffer" in self.state[p]:
                    p.sub_(self.state[p]["momentum_buffer"], alpha=group["rho"]*group["inverse_norm_buffer"])
    
    @torch.no_grad()
    def move_back_from_momentumAscent(self):
        for group in self.param_groups:
            for p in group['params']:
                if "momentum_buffer" in self.state[p]:
                    p.add_(self.state[p]["momentum_buffer"], alpha=group["rho"]*group["inverse_norm_buffer"])


class RAND_MSAM(torch.optim.Optimizer):
    def __init__(self, params, lr=0.5, momentum=0.9, weight_decay=0.0005, rho=3):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, rho=rho)
        super(MSAM, self).__init__(params, defaults)
        
        for group in self.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    state['momentum_buffer'] = torch.zeros_like(p).detach()
            group["inverse_norm_buffer"] = 0
    
    @torch.no_grad()
    def step(self):
        group = self.param_groups[0]
        params_with_grad = []
        d_p_list = []
        momentum_buffer_list = []
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        rho = group['rho']
        lr = group['lr']
        
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                momentum_buffer_list.append(self.state[p]['momentum_buffer'])
        
        for i, param in enumerate(params_with_grad):
            d_p = d_p_list[i]
            if weight_decay != 0:
                d_p.add_(param, alpha=weight_decay)
            
            buf = momentum_buffer_list[i]
            param.add_(buf, alpha=rho*group["inverse_norm_buffer"])
            param.add_(torch.randn_like(param) * 1e-5) #added as noise
            
            buf.mul_(momentum).add_(d_p)
            param.add_(buf, alpha=-lr)
        
        ascent_norm = torch.norm(
            torch.stack([buf.norm(p=2) for buf in momentum_buffer_list]),
            p=2
        )
        group["inverse_norm_buffer"] = 1/(ascent_norm+1e-12)
        
        for i, param in enumerate(params_with_grad):
            param.sub_(momentum_buffer_list[i], alpha=rho*group["inverse_norm_buffer"])
        
        for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
            self.state[p]['momentum_buffer'] = momentum_buffer
    
    @torch.no_grad()
    def move_up_to_momentumAscent(self):
        for group in self.param_groups:
            for p in group['params']:
                if "momentum_buffer" in self.state[p]:
                    p.sub_(self.state[p]["momentum_buffer"], alpha=group["rho"]*group["inverse_norm_buffer"])
    
    @torch.no_grad()
    def move_back_from_momentumAscent(self):
        for group in self.param_groups:
            for p in group['params']:
                if "momentum_buffer" in self.state[p]:
                    p.add_(self.state[p]["momentum_buffer"], alpha=group["rho"]*group["inverse_norm_buffer"])

# ====================== Data Loading ======================

def get_cifar100_loaders(batch_size=256, data_dir='~/.datasets'):
    data_dir = os.path.expanduser(data_dir)
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])
    
    trainset = torchvision.datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=transform_test)
    
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=6)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=6)
    
    return trainloader, testloader

# ====================== Learning Rate Scheduler ======================

class CosineWithWarmup:
    def __init__(self, optimizer, epochs=200, warmup_epochs=1, lr=0.5):
        self.optimizer = optimizer
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.base_lr = lr
        
    def step(self, epoch, step=None):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch / self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)
            lr = 0.5 * self.base_lr * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

# ====================== Training Loop ======================

def train_epoch(model, trainloader, optimizer, scheduler, epoch, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        if isinstance(optimizer, SAM):
            scheduler.step(epoch-1, (batch_idx+1)/len(trainloader))
            loss, predictions = optimizer.step(model, inputs, targets)
        elif isinstance(optimizer, MSAM):
            scheduler.step(epoch-1, (batch_idx+1)/len(trainloader))
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            loss.mean().backward()
            optimizer.step()
            optimizer.zero_grad()
        elif isinstance(optimizer, RAND_MSAM):
            scheduler.step(epoch-1, (batch_idx+1)/len(trainloader))
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            loss.mean().backward()
            optimizer.step()
            optimizer.zero_grad()
        else:  # SGD
            scheduler.step(epoch-1, (batch_idx+1)/len(trainloader))
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            loss.mean().backward()
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.mean().item()
        _, predicted = predictions.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return total_loss/len(trainloader), 100.*correct/total

def test(model, testloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets)
            
            total_loss += loss.mean().item()
            _, predicted = predictions.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return total_loss/len(testloader), 100.*correct/total

def main():
    parser = argparse.ArgumentParser(description='CIFAR100 Training')
    parser.add_argument('--model', type=str, default='WRN', choices=['WRN', 'ResNet50'],
                       help='Model architecture: WRN or ResNet50')
    parser.add_argument('--optimizer', type=str, default='SGD', 
                       choices=['SGD', 'SAM', 'MSAM', 'RAND_MSAM'],
                       help='Optimizer: SGD, SAM, MSAM or RAND_MSAM')
    parser.add_argument('--rho', type=float, default=0.3,
                       help='Rho parameter for SAM/MSAM/RAND_MSAM (default: 0.3 for SAM, 3 for MSAM/RAND_MSAM)')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--data_dir', type=str, default='~/.datasets',
                       help='Data directory (default: ~/.datasets)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    weight_decay_ = None
    # Set hyperparameters based on model
    if args.model == 'WRN':
        model = WideResNet(depth=16, widen_factor=4, num_classes=100, dropRate=0.0)
        lr = 0.5
        weight_decay_ = 5e-4
    else:  # ResNet50
        model = ResNetCIFAR(depth=50, num_classes=100)
        lr = 0.1
        weight_decay_ = 1e-3
    
    model = model.to(device)
    
    # Set optimizer with appropriate rho
    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay_)
        rho = None
    elif args.optimizer == 'SAM':
        optimizer = SAM(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay_, rho=args.rho)
        rho = args.rho
    elif args.optimizer == 'MSAM':
        optimizer = MSAM(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay_, rho=args.rho)
        rho = args.rho
    elif args.optimizer == 'RAND_MSAM':
        optimizer = RAND_MSAM(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay_, rho=args.rho)
        rho = args.rho
    
    # Data loaders
    trainloader, testloader = get_cifar100_loaders(batch_size=args.batch_size, data_dir=args.data_dir)
    
    # Learning rate scheduler
    scheduler = CosineWithWarmup(optimizer, epochs=args.epochs, warmup_epochs=1, lr=lr)
    
    print(f'Training {args.model} with {args.optimizer} optimizer for {args.epochs} epochs')
    if rho is not None:
        print(f'Rho parameter: {rho}')
    
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        if isinstance(optimizer, MSAM) or isinstance(optimizer, RAND_MSAM):
            optimizer.move_up_to_momentumAscent()
        
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, scheduler, epoch, device)
        
        if isinstance(optimizer, MSAM) or isinstance(optimizer, RAND_MSAM):
            optimizer.move_back_from_momentumAscent()
        
        test_loss, test_acc = test(model, testloader, device)
        
        elapsed = time.time() - start_time
        
        print(f'Epoch {epoch:3d}/{args.epochs} | '
              f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.2f}% | '
              f'Time: {elapsed:.1f}s')
        
        if test_acc > best_acc:
            best_acc = test_acc
    
    print(f'\nBest test accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    SEED = 42
    set_seed(SEED)
    main()
