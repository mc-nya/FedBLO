#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import torch


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument("--epochs", type=int, default=100, help="rounds of training")
    parser.add_argument("--round", type=int, default=0, help="rounds of communication")
    parser.add_argument("--num_users", type=int, default=100, help="number of users: K")
    parser.add_argument(
        "--frac", type=float, default=0.1, help="the fraction of clients: C"
    )
    parser.add_argument(
        "--local_ep", type=int, default=5, help="the number of local epochs: E"
    )
    parser.add_argument("--local_bs", type=int, default=10, help="local batch size: B")
    parser.add_argument("--bs", type=int, default=128, help="test batch size")
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")
    parser.add_argument(
        "--momentum", type=float, default=0, help="SGD momentum (default: 0.5)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="user",
        help="train-test split type, user or sample",
    )

    # bilevel arguments
    parser.add_argument(
        "--neumann", type=int, default=5, help="The iteration of nuemann series"
    )
    parser.add_argument(
        "--inner_ep", type=int, default=1, help="the number of hyper local epochs: H_E"
    )
    parser.add_argument(
        "--outer_tau", type=int, default=5, help="the number of hyper local epochs: H_E"
    )
    parser.add_argument("--hlr", type=float, default=0.01, help="learning rate")
    parser.add_argument(
        "--hvp_method", type=str, default="global_batch", help="hvp method"
    )
    parser.add_argument("--no_blo", action="store_true", help="whether blo or not")
    # FedBlo arguments
    parser.add_argument(
        "--momentum_rho",
        type=float,
        default=0.5,
        help="FedBLO SARAH momentum (default: 0.5)",
    )
    parser.add_argument(
        "--q_noniid", type=float, default=0.1, help="the fraction of non-iid clients"
    )
    parser.add_argument("--alpha", type=float, default=0.1, help="lr for x")
    parser.add_argument("--beta", type=float, default=0.1, help="lr for z")
    parser.add_argument("--sarah_momentum", type=float, default=0.0, help="momentum for SARAH/SPIDER local update")
    # Minmax arguments
    parser.add_argument(
        "--minmax_s", type=int, default=5, help="The heterogenity of synthetic dataset"
    )

    # model arguments
    parser.add_argument("--model", type=str, default="mlp", help="model name")
    parser.add_argument(
        "--kernel_num", type=int, default=9, help="number of each kind of kernel"
    )
    parser.add_argument(
        "--kernel_sizes",
        type=str,
        default="3,4,5",
        help="comma-separated kernel size to use for convolution",
    )
    parser.add_argument(
        "--norm", type=str, default="batch_norm", help="batch_norm, layer_norm, or None"
    )
    parser.add_argument(
        "--num_filters", type=int, default=32, help="number of filters for conv nets"
    )
    parser.add_argument(
        "--max_pool",
        type=str,
        default="True",
        help="Whether use max pooling rather than strided convolutions",
    )
    parser.add_argument("--optim", type=str, default="svrg", help="optimizer name")

    # other arguments
    parser.add_argument("--dataset", type=str, default="mnist", help="name of dataset")
    parser.add_argument("--iid", action="store_true", help="whether i.i.d or not")
    parser.add_argument("--num_classes", type=int, default=10, help="number of classes")
    parser.add_argument(
        "--num_channels", type=int, default=1, help="number of channels of imges"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument(
        "--stopping_rounds", type=int, default=10, help="rounds of early stopping"
    )
    parser.add_argument("--verbose", action="store_true", help="verbose print")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument(
        "--all_clients", action="store_true", help="aggregation over all clients"
    )
    parser.add_argument("--output", type=str, default=None, help="output path")
    args = parser.parse_args()
    args.device = torch.device(
        "cuda:{}".format(args.gpu)
        if torch.cuda.is_available() and args.gpu != -1
        else "cpu"
    )

    return args
