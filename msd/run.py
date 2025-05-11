import os
from argparse import ArgumentParser
from os import path as osp

from omegaconf import OmegaConf

from msd.configurations.config_initializer import ConfigInitializer
from msd.utils.environment import get_environment_details
from msd.utils.loading_utils import load_config, set_seed

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--run_config', type=str, help='Config file', required=True)
    parser.add_argument('--meta_config', type=str, help='meta config file')
    parser.add_argument('--train', action='store_true', help='Training mode')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    args = parser.parse_args()
    print(f'Current working directory: {os.getcwd()}')
    if not osp.exists(args.run_config):
        print(f'Config file {args.run_config} not found.')
        exit(1)
    if not args.train and not args.eval:
        print('No train/eval specified. Exiting.')
        exit(1)
    if args.train and args.eval:
        print('This benchmarks only supports one of training or evaluation at a time. Will execute training.')
        args.eval = False
    meta_config = 'configurations/meta.yaml' if args.meta_config is None else args.meta_config
    cfg = load_config(args.run_config, meta_config, args.train)
    cfg_init = ConfigInitializer(cfg)
    logger = cfg_init.logger
    logger.info(f'Config path: {args.run_config}')
    logger.log_file('config_file', args.run_config)
    logger.log_dict('config', OmegaConf.to_container(cfg, resolve=True))
    logger.log_dict('environment', get_environment_details())
    name, device = cfg.name, cfg.device

    if args.train and 'seed' in cfg:
        logger.info(f'Setting seed: {cfg.seed}')
        set_seed(cfg.seed)
    else:
        logger.info('Seed not set.')

    if args.train:
        trainer = cfg_init.get_trainer()
        trainer.train()
    if args.eval:
        evaluation_manager = cfg_init.get_evaluator()
        evaluation_manager.run_test()
