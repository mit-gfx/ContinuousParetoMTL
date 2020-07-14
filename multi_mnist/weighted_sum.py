import random
from pathlib import Path

from termcolor import colored

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

from pareto.metrics import topk_accuracy
from pareto.datasets import MultiMNIST
from pareto.networks import MultiLeNet
from pareto.utils import evenly_dist_weights


@torch.no_grad()
def evaluate(network, dataloader, device, closures, header=''):
    num_samples = 0
    losses = np.zeros(2)
    top1s = np.zeros(2)
    network.train(False)
    for images, labels in dataloader:
        batch_size = len(images)
        num_samples += batch_size
        images = images.to(device)
        labels = labels.to(device)
        logits = network(images)
        losses_batch = [c(network, logits, labels).item() for c in closures]
        losses += batch_size * np.array(losses_batch)
        top1s[0] += batch_size * topk_accuracy(logits[0], labels[:, 0], k=1)
        top1s[1] += batch_size * topk_accuracy(logits[1], labels[:, 1], k=1)
    losses /= num_samples
    top1s /= num_samples

    loss_msg = '[{}]'.format('/'.join([f'{loss:.6f}' for loss in losses]))
    top1_msg = '[{}]'.format('/'.join([f'{top1 * 100.0:.2f}%' for top1 in top1s]))
    msgs = [
        f'{header}:' if header else '',
        'loss', colored(loss_msg, 'yellow'),
        'top@1', colored(top1_msg, 'yellow')
    ]
    print(' '.join(msgs))
    return losses, top1s


def train(pref, ckpt_name):

    # prepare hyper-parameters

    seed = 42

    cuda_enabled = True
    cuda_deterministic = False

    batch_size = 256
    num_workers = 2

    lr = 0.01
    momentum = 0.9
    weight_decay = 0.0

    num_epochs = 30


    # prepare path

    root_path = Path(__file__).resolve().parent
    dataset_path = root_path / 'MultiMNIST'
    ckpt_path = root_path / 'weighted_sum'

    root_path.mkdir(parents=True, exist_ok=True)
    dataset_path.mkdir(parents=True, exist_ok=True)
    ckpt_path.mkdir(parents=True, exist_ok=True)


    # fix random seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_enabled and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


    # prepare device

    if cuda_enabled and torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        device = torch.device('cuda')
        if cuda_deterministic:
            cudnn.benchmark = False
            cudnn.deterministic = True
        else:
            cudnn.benchmark = True
    else:
        device = torch.device('cpu')


    # prepare dataset

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainset = MultiMNIST(dataset_path, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = MultiMNIST(dataset_path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    # prepare network

    network = MultiLeNet()
    network.to(device)


    # prepare losses

    criterion = F.cross_entropy
    closures = [lambda n, l, t: criterion(l[0], t[:, 0]), lambda n, l, t: criterion(l[1], t[:, 1])]


    # prepare optimizer

    optimizer = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = CosineAnnealingLR(optimizer, num_epochs * len(trainloader))


    # save initial state

    if not (ckpt_path / 'random.pth').is_file():
        random_ckpt = {
            'state_dict': network.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict()
        }
        torch.save(random_ckpt, ckpt_path / 'random.pth')
    random_ckpt = torch.load(ckpt_path / 'random.pth', map_location='cpu')

    network.load_state_dict(random_ckpt['state_dict'])
    optimizer.load_state_dict(random_ckpt['optimizer'])
    lr_scheduler.load_state_dict(random_ckpt['lr_scheduler'])


    # first evaluation

    evaluate(network, testloader, device, closures, f'{ckpt_name}')


    # training

    num_steps = len(trainloader)
    for epoch in range(1, num_epochs + 1):

        network.train(True)
        trainiter = iter(trainloader)
        for _ in range(1, num_steps + 1):

            images, labels = next(trainiter)
            images = images.to(device)
            labels = labels.to(device)
            logits = network(images)
            losses = [c(network, logits, labels) for c in closures]
            loss = sum(w * l for w, l in zip(pref, losses))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        losses, tops = evaluate(network, testloader, device, closures, f'{ckpt_name}: {epoch}/{num_epochs}')


    # saving

    ckpt = {
        'state_dict': network.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'preference': pref,
    }
    record = {'losses': losses, 'tops': tops}
    ckpt['record'] = record
    torch.save(ckpt, ckpt_path / f'{ckpt_name}.pth')


def weighted_sum(num_prefs=5):
    prefs = evenly_dist_weights(num_prefs + 2, 2)
    for i, pref in enumerate(prefs):
        train(pref, str(i))


if __name__ == '__main__':
    weighted_sum(5)
