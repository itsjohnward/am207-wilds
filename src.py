import os, csv
import time
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import sys
from collections import defaultdict

from tqdm import tqdm

import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper

from wilds.examples.utils import set_seed, Logger, BatchLogger, log_config, ParseKwargs, load, initialize_wandb, log_group_data, parse_bool, get_model_prefix
from wilds.examples.train import train, evaluate
from wilds.examples.algorithms.initializer import initialize_algorithm
from wilds.examples.transforms import initialize_transform
from wilds.examples.configs.utils import populate_defaults
import wilds.examples.configs.supported as supported

import torch.multiprocessing

def run_experiment(config={}):
    config = populate_defaults(config)

    # For the GlobalWheat detection dataset,
    # we need to change the multiprocessing strategy or there will be
    # too many open file descriptors.
    if config.get('dataset') == 'globalwheat':
        torch.multiprocessing.set_sharing_strategy('file_system')

    # Set device
    config['device'] = torch.device("cuda:" + str(config.get('device'))) if torch.cuda.is_available() else torch.device("cpu")

    # Initialize logs
    if os.path.exists(config.get('log_dir')) and config.get('resume'):
        resume=True
        mode='a'
    elif os.path.exists(config.get('log_dir')) and config.get('eval_only'):
        resume=False
        mode='a'
    else:
        resume=False
        mode='w'

    if not os.path.exists(config.get('log_dir')):
        os.makedirs(config.get('log_dir'))
    logger = Logger(os.path.join(config.get('log_dir'), 'log.txt'), mode)

    # Record config
    log_config(config, logger)

    # Set random seed
    set_seed(config.get('seed'))

    # Data
    full_dataset = config.get('full_dataset')
    if full_dataset is None:
        full_dataset = wilds.get_dataset(
            dataset=config.get('dataset'),
            version=config.get('version'),
            root_dir=config.get('root_dir'),
            download=config.get('download'),
            split_scheme=config.get('split_scheme'),
            **config.get('dataset_kwargs')
        )

    print('dataset')

    # To modify data augmentation, modify the following code block.
    # If you want to use transforms that modify both `x` and `y`,
    # set `do_transform_y` to True when initializing the `WILDSSubset` below.
    train_transform = initialize_transform(
        transform_name=config.get('transform'),
        config=config,
        dataset=full_dataset,
        is_training=True)

    print('train')

    eval_transform = initialize_transform(
        transform_name=config.get('transform'),
        config=config,
        dataset=full_dataset,
        is_training=False)

    print('eval')

    train_grouper = CombinatorialGrouper(
        dataset=full_dataset,
        groupby_fields=config.get('groupby_fields'))

    datasets = defaultdict(dict)
    for split in tqdm(full_dataset.split_dict.keys()):
        if split=='train':
            transform = train_transform
            verbose = True
        elif split == 'val':
            transform = eval_transform
            verbose = True
        else:
            transform = eval_transform
            verbose = False
        # Get subset
        datasets[split]['dataset'] = full_dataset.get_subset(
            split,
            frac=config.get('frac'),
            transform=transform)

        if split == 'train':
            datasets[split]['loader'] = get_train_loader(
                loader=config.get('train_loader'),
                dataset=datasets[split]['dataset'],
                batch_size=config.get('batch_size'),
                uniform_over_groups=config.get('uniform_over_groups'),
                grouper=train_grouper,
                distinct_groups=config.get('distinct_groups'),
                n_groups_per_batch=config.get('n_groups_per_batch'),
                **config.get('loader_kwargs'))
        else:
            datasets[split]['loader'] = get_eval_loader(
                loader=config.get('eval_loader'),
                dataset=datasets[split]['dataset'],
                grouper=train_grouper,
                batch_size=config.get('batch_size'),
                **config.get('loader_kwargs'))

        # Set fields
        datasets[split]['split'] = split
        datasets[split]['name'] = full_dataset.split_names[split]
        datasets[split]['verbose'] = verbose

        # Loggers
        datasets[split]['eval_logger'] = BatchLogger(
            os.path.join(config.get('log_dir'), f'{split}_eval.csv'), mode=mode, use_wandb=(config.get('use_wandb') and verbose))
        datasets[split]['algo_logger'] = BatchLogger(
            os.path.join(config.get('log_dir'), f'{split}_algo.csv'), mode=mode, use_wandb=(config.get('use_wandb') and verbose))

        if config.get('use_wandb'):
            initialize_wandb(config)

    # Logging dataset info
    # Show class breakdown if feasible
    if config.get('no_group_logging') and full_dataset.is_classification and full_dataset.y_size==1 and full_dataset.n_classes <= 10:
        log_grouper = CombinatorialGrouper(
            dataset=full_dataset,
            groupby_fields=['y'])
    elif config.get('no_group_logging'):
        log_grouper = None
    else:
        log_grouper = train_grouper
    log_group_data(datasets, log_grouper, logger)

    ## Initialize algorithm
    algorithm = initialize_algorithm(
        config=config,
        datasets=datasets,
        train_grouper=train_grouper)

    model_prefix = get_model_prefix(datasets['train'], config)
    if not config.get('eval_only'):
        ## Load saved results if resuming
        resume_success = False
        if resume:
            save_path = model_prefix + 'epoch:last_model.pth'
            if not os.path.exists(save_path):
                epochs = [
                    int(file.split('epoch:')[1].split('_')[0])
                    for file in os.listdir(config.get('log_dir')) if file.endswith('.pth')]
                if len(epochs) > 0:
                    latest_epoch = max(epochs)
                    save_path = model_prefix + f'epoch:{latest_epoch}_model.pth'
            try:
                prev_epoch, best_val_metric = load(algorithm, save_path)
                epoch_offset = prev_epoch + 1
                logger.write(f'Resuming from epoch {epoch_offset} with best val metric {best_val_metric}')
                resume_success = True
            except FileNotFoundError:
                pass

        if resume_success == False:
            epoch_offset=0
            best_val_metric=None

        train(
            algorithm=algorithm,
            datasets=datasets,
            general_logger=logger,
            config=config,
            epoch_offset=epoch_offset,
            best_val_metric=best_val_metric)
    else:
        if config.get('eval_epoch') is None:
            eval_model_path = model_prefix + 'epoch:best_model.pth'
        else:
            eval_model_path = model_prefix +  f'epoch:{config.get("eval_epoch")}_model.pth'
        best_epoch, best_val_metric = load(algorithm, eval_model_path)
        if config.get('eval_epoch') is None:
            epoch = best_epoch
        else:
            epoch = config.get('eval_epoch')
        if epoch == best_epoch:
            is_best = True
        evaluate(
            algorithm=algorithm,
            datasets=datasets,
            epoch=epoch,
            general_logger=logger,
            config=config,
            is_best=is_best)

    logger.close()
    for split in datasets:
        datasets[split]['eval_logger'].close()
        datasets[split]['algo_logger'].close()
