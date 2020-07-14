# Source code for ICML submission #640 "Efficient Continuous Pareto Exploration in Multi-Task Learning"
class PrettyTabular(object):
    def __init__(self, head):
        self.head = head

    def head_string(self):
        line = ''
        for key, value in self.head.items():
            try:
                dummy = value.format(0)     # Try digits.
            except:
                dummy = value.format('0')   # Try strings.
            span = max(len(dummy), len(key)) + 2
            key_format = '{:^' + str(span) + '}'
            line += key_format.format(key)
        return line

    def row_string(self, row_data):
        line = ''
        for key, value in self.head.items():
            data = value.format(row_data[key])
            span = max(len(key), len(data)) + 2
            line += ' ' * (span - len(data) - 1) + data + ' '
        return line

if __name__ == '__main__':
    # head[name] = (format).
    head = { 'iter': '{:4d}', 'objective': '{:3.6e}', 'violations': '{:3.6e}' }
    tabular = PrettyTabular(head)

    import numpy as np
    from common import *
    for i in range(20):
        if i % 10 == 0:
            print_info(tabular.head_string())
        row_data = { 'iter': i, 'objective': np.random.rand(), 'violations': np.random.rand() }
        print(tabular.row_string(row_data))
