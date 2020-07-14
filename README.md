# Efficient Continuous Pareto Exploration in Multi-Task Learning

![zdt2](assets/zdt2.png)

[Pingchuan Ma](https://pingchuan.ma/)\*,
[Tao Du](https://people.csail.mit.edu/taodu/)\*,
and
[Wojciech Matusik](http://people.csail.mit.edu/wojciech/)

**ICML 2020**
[[Paper]](http://people.csail.mit.edu/pcma/documents/cpmtl/paper.pdf)
[[Appendix]](http://people.csail.mit.edu/pcma/documents/cpmtl/supp.pdf)
[[arXiv]](https://arxiv.org/abs/2006.16434)
[[Video]](https://icml.cc/virtual/2020/poster/5856)
[[Slides]](http://people.csail.mit.edu/pcma/documents/cpmtl/slides.pdf)

```text
@inproceedings{ma2020continuous,
    title={Efficient Continuous Pareto Exploration in Multi-Task Learning},
    author={Ma, Pingchuan and Du, Tao and Matusik, Wojciech},
    booktitle={International Conference on Machine Learning},
    year={2020},
}
```

## Prerequisites

- Ubuntu 16.04 or higher;
- conda 4.8 or higher.

## Installation

We will use `$ROOT` to refer to the root folder where you want to put this project in. We compiled continuous pareto MTL into a package `pareto` for easier deployment and application.

```sh
cd $ROOT
git clone https://github.com/mit-gfx/ContinuousParetoMTL.git
cd ContinuousParetoMTL
conda env create -f environment.yml
conda activate cpmtl
python setup.py install
```

## Example for MultiMNIST

After `pareto` is installed, we are free to call any primitive functions and classes which are useful for Pareto-related tasks, including continuous Pareto exploration. We provide an example for MultiMNIST dataset, which can be found by:

```sh
cd multi_mnist
```

First, we run weighted sum method for initial Pareto solutions:

```sh
python weighted_sum.py
```

The output should be like:

```text
0: loss [2.313036/2.304537] top@1 [7.65%/10.65%]
0: 1/30: loss [1.463346/0.909529] top@1 [51.52%/69.72%]
0: 2/30: loss [0.889257/0.638646] top@1 [71.29%/78.55%]
0: 3/30: loss [0.703745/0.534612] top@1 [77.77%/81.86%]
0: 4/30: loss [0.622291/0.491764] top@1 [80.13%/83.02%]
```

Based on these starting solutions, we can run our continuous Pareto exploration by:

```sh
python cpmtl.py
```

The output should be like:

```text
0: 1/10: loss [0.397692/0.350267] top@1 [86.57%/88.11%]
    86.37% 86.57% Δ=0.20% absΔ=0.20%
    88.10% 88.11% Δ=0.01% absΔ=0.01%

0: 2/10: loss [0.392314/0.351280] top@1 [86.85%/88.07%]
    86.37% 86.57% 86.85% Δ=0.28% absΔ=0.48%
    88.10% 88.11% 88.07% Δ=-0.04% absΔ=-0.03%

0: 3/10: loss [0.387585/0.352643] top@1 [86.92%/88.03%]
    86.37% 86.57% 86.85% 86.92% Δ=0.07% absΔ=0.55%
    88.10% 88.11% 88.07% 88.03% Δ=-0.04% absΔ=-0.07%
```

Now you can play it on your own dataset and network architecture!

## Jupyter Notebooks for Submission

Open up a terminal to launch Jupyter:

```sh
cd submission
jupyter notebook
```

You can run the following Jupyter script to reproduce figures in the paper:

```text
fig2.ipynb
fig3.ipynb
fig4.ipynb
fig5_multimnist.ipynb
fig5_uci.ipynb
```

## Contact

If you have any questions about the paper or the codebase, please feel free to contact pcma@csail.mit.edu or taodu@csail.mit.edu.
