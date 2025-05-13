from typing import Any, Dict, Optional, Tuple, Union

from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader

from msd.configurations.class_registry import ClassRegistry
from msd.data.disentanglement_dataset import DisentanglementDataset
from msd.evaluations.evaluation_manager import EvaluationManager
from msd.methods.abstract_model import AbstractModel
from msd.methods.abstract_trainer import AbstractTrainer
from msd.utils.loggers.base_logger import CompositeLogger


class ConfigInitializer:
    """
    A utility class to initialize datasets, models, trainers, evaluators, and loggers from configuration.

    Components are lazily instantiated and may cached by identifier. This allows configuration-based
    modular experimentation without modifying code.
    """
    def __init__(self, config: DictConfig) -> None:
        """
        :param config: Hierarchical configuration object.
        """
        self.registry = ClassRegistry(package_root=".")
        self.registry.scan_and_register()
        self.config: DictConfig = config
        self.objects: Dict[str, Any] = {}
        self.logger: CompositeLogger = self.get_logger()
        self.logger.info('Initializer ready.')
        # CLASS_REGISTRY |= scan_and_generate_registry(package_root='.')

    def get_dataset(self, split: str, loaders: bool = False, labels: bool = False, return_names: bool = False) -> Union[DisentanglementDataset, Tuple[DisentanglementDataset, DataLoader]]:
        """
        Load a dataset and optionally return a corresponding DataLoader.

        :param split: Dataset split to load ('train', 'val', or 'test').
        :param loaders: Whether to return a DataLoader.
        :param labels: Whether to load labels (for evaluation).
        :param return_names: Whether to return the class names instead of labels.
        :return: DisentanglementDataset or (DisentanglementDataset, DataLoader) tuple.
        """
        data_cfg = self.config.dataset
        reader_cfg = data_cfg.reader
        split_cfg = data_cfg.splits[split]

        identifier = f'{reader_cfg.name}({split})'
        cached = f'Reader({identifier})' in self.objects
        reader = self.initialize(reader_cfg, identifier=f'Reader({identifier})', split=split)
        hooks = [self.initialize(h) for h in split_cfg.preprocess_hooks]
        dataset = self.initialize(data_cfg.dataset, identifier=f'Dataset({identifier})', use_cache=False,
                                  reader=reader, preprocess_hooks=hooks, supervised=labels, return_names=return_names)
        if not cached:
            self.logger.info(f'Loaded {dataset} from {reader}')
        if loaders:
            loader_cfg = split_cfg.loader
            data_loader = self.initialize(loader_cfg,
                                          identifier=f'DataLoader({identifier};{loader_cfg.parameters})',
                                          use_cache=False, dataset=dataset)
            return dataset, data_loader
        else:
            return dataset

    def get_model(self, **kwargs) -> AbstractModel:
        """
        Instantiate and return the model defined in the configuration.

        :param kwargs: Additional keyword arguments passed to the model constructor.
        :return: Instantiated model.
        """
        model = self.initialize(self.config.model, **kwargs)
        self.logger.info(f'Loaded model:\n{model}')
        return model

    def get_trainer(self) -> AbstractTrainer:
        """
        Instantiate and return the trainer as defined in the configuration.

        :return: Trainer instance.
        """
        return self.initialize(self.config.trainer, initializer=self)

    def get_evaluator(self, model: Optional[nn.Module] = None) -> EvaluationManager:
        """
        Instantiate the evaluation manager as defined in the configuration.

        :param model: Optional model instance to evaluate.
        :return: EvaluationManager instance.
        """
        evaluation_manager = self.initialize(self.config.evaluation.evaluation_manager, initializer=self, model=model)
        return evaluation_manager

    def get_logger(self) -> CompositeLogger:
        """
        Construct and return a CompositeLogger based on the logger configuration.

        :return: Composed logger instance.
        """
        if hasattr(self, 'logger'):
            return self.logger
        else:
            loggers = [self.initialize(l) for l in self.config.loggers]
            self.logger = CompositeLogger(loggers)
        return self.logger


    def initialize(
        self,
        init_cfg: DictConfig,
        identifier: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> Any:
        """
        Generic object instantiation with optional caching.

        :param init_cfg: Configuration block with `name` and `parameters`.
        :param identifier: Unique identifier for object caching.
        :param use_cache: Whether to return a cached instance if available.
        :param kwargs: Additional arguments for the class constructor.
        :return: Instantiated object.
        """
        name = init_cfg.name
        if identifier is None:
            identifier = name
        if identifier in self.objects and use_cache:
            self.logger.debug(f'Using cached object {identifier}')
            return self.objects[identifier]
        else:
            if hasattr(self, 'logger'):
                self.logger.debug(f'Initializing new {identifier}')
            obj = self.initialize_class(name, kwargs | dict(init_cfg.parameters))
            if use_cache:
                self.objects[identifier] = obj
            return obj

    def initialize_class(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Dynamically instantiate a class from the registry.

        :param name: Class name as registered in the class registry.
        :param args: Dictionary of keyword arguments to pass to the constructor.
        :return: Instantiated object.
        :raises Exception: If instantiation fails due to invalid arguments or missing class.
        """
        try:
            cls = self.registry.class_of(name)(**dict(args))
        except Exception as e:
            print(f'Failed to initialize class {name} with args {args}')
            raise e
        return cls