from typing import Iterable
from torch import Tensor


__all__ = ['topk_accuracies', 'topk_accuracy']


def topk_accuracies(
        output: Tensor,
        label: Tensor,
        ks: Iterable[int] = (1,),
    ):

    assert output.dim() == 2
    assert label.dim() == 1
    assert output.size(0) == label.size(0)

    maxk = max(ks)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    label = label.unsqueeze(1).expand_as(pred)
    correct = pred.eq(label).float()

    accu_list = []
    for k in ks:
        accu = correct[:, :k].sum(1).mean()
        accu_list.append(accu.item())
    return accu_list


def topk_accuracy(
        output: Tensor,
        label: Tensor,
        k: int = 1,
    ):

    return topk_accuracies(output, label, (k,))[0]
