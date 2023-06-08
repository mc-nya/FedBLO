# Federated Multi-Sequence  Stochastic Approximation with Local Hypergradient Estimation

This directory contains source code for evaluating FedMSA on various models and tasks. The code is adopted from FedNest ([arXiv link](https://arxiv.org/abs/2205.02215)) ([ICML 2022 link](https://icml.cc/Conferences/2022/Schedule?showEvent=17792)). Check the reproduce folder to reproduce the result. FedMCO code is in the jupyter notebook file, [fedMCO_stochastic_final.ipynb](fedMCO_stochastic_final.ipynb).

To reproduce the result, run the following 4 python scripts in the order of:

```
python reproduce/run_blo_10clients.py
python reproduce/run_fednest.py
python figure_q.py
python figure_tau.py
```

[run_blo_10clients.py](reproduce/run_blo_10clients.py) train the model under FedMSA and [run_fednest](reproduce/run_fednest.py) use the FedNest original code and parameters to run the baseline results. Based on FedNest, we mainly change [core.Client](core/Client.py), add [core.ClientManage_blo](core/ClientManage_blo.py) and [main_imbalance_blo](./main_imbalance_blo.py).


# Requirements
python>=3.6  
pytorch>=0.4


# Customize Run

The imbalanced MNIST experiments are produced by:
> python [main_imbalance_blo.py](main_imbalance_blo.py)

The FedMCO synthetic experiments are produced in the Jupyter Notebook [fedMCO_stochastic_final.ipynb](fedMCO_stochastic_final.ipynb)

The arguments are avaliable in [options.py](utils/options.py). And the reproduce scripts also provides several scripts to run the experiments.

