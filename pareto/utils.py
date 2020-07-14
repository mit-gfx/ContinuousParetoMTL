from typing import Iterable
from itertools import product

from termcolor import colored

import numpy as np


class TopTrace(object):
    def __init__(
            self,
            num_objs: int,
            *,
            indent_size: 4,
        ):

        self.tops = [[] for _ in range(num_objs)]
        self.msgs = [[] for _ in range(num_objs)]
        self.indent_size = indent_size

    def print(
            self,
            new_tops: Iterable[float],
            *,
            show: bool = True,
        ):

        for new_top, top, msg in zip(new_tops, self.tops, self.msgs):
            new_top_msg = f'{new_top * 100.0:.2f}%'
            if top:
                new_top_msg = colored(new_top_msg, 'green' if new_top >= top[-1] else 'red')
                delta = '\u0394=' + colored(f'{(new_top - top[-1]) * 100.0:.2f}%', 'green' if new_top >= top[-1] else 'red')
                abs_delta = 'abs\u0394=' + colored(f'{(new_top - top[0]) * 100.0:.2f}%', 'green' if new_top >= top[0] else 'red')
            top.append(new_top)
            msg.append(new_top_msg)
            if show:
                print(' ' * self.indent_size + ' '.join(msg + [delta, abs_delta]))
                print(flush=True)


def evenly_dist_weights(num_weights, dim):
    return [ret for ret in product(
        np.linspace(0.0, 1.0, num_weights), repeat=dim) if round(sum(ret), 3) == 1.0 and all(r not in (0.0, 1.0) for r in ret)]


if __name__ == "__main__":
    print(evenly_dist_weights(7, 2))
