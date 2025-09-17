# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random
import numpy as np
import torch


def img_param_init(args):
    dataset = args.dataset
    if dataset == 'pacs':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
        args.num_classes = 7
    elif dataset == 'office_home':
        domains = ['Art', 'Clipart', 'Product', 'Real_World']
        args.num_classes = 65
    elif dataset == 'vlcs':
        domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
        args.num_classes = 5
    elif dataset == 'domain_net':
        domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        args.num_classes = 345
    else:
        print('No such dataset exists!')
    args.domains = domains
    return args


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
