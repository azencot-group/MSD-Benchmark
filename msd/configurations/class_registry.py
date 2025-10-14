import ast
import importlib
import os
from os import path as osp
from typing import Dict, List, Type


class ClassRegistry:
    def __init__(self, package_root: str = "."):
        self.package_root = package_root
        self.registry: Dict[str, str] = {
            # External modules (manually registered)
            "MSELoss": "torch.nn",
            "CrossEntropyLoss": "torch.nn",
            "SGD": "torch.optim",
            "Adam": "torch.optim",
            "DataLoader": "torch.utils.data",
            "StepLR": "torch.optim.lr_scheduler",
            "DecisionTreeClassifier": "sklearn.tree",
            "RandomForestClassifier": "sklearn.ensemble",
            "GradientBoostingClassifier": "sklearn.ensemble",
            "FramewiseCrossEntropyLoss": "msd.evaluations.classifiers.framewise_cross_entropy",
        }
        self.cache: Dict[str, Type] = {}

    def scan_and_register(self):
        for dirpath, _, filenames in os.walk(self.package_root):
            for filename in filenames:
                if filename.endswith(".py") and not filename.startswith("_"):
                    filepath = osp.join(dirpath, filename)
                    rel_path = osp.relpath(filepath, self.package_root).replace(os.sep, ".")
                    module_name = osp.splitext(rel_path)[0]

                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            node = ast.parse(f.read(), filename=filepath)
                    except SyntaxError as e:
                        print(f"Skipping {filepath} due to SyntaxError: {e}")
                        continue

                    for item in node.body:
                        if isinstance(item, ast.ClassDef):
                            class_name = item.name
                            try:
                                module = importlib.import_module(module_name)
                                cls = getattr(module, class_name, None)

                                if cls and isinstance(cls, type):
                                    super_cls = [c.__name__ for c in cls.__mro__]
                                    if 'MSDComponent' in super_cls:
                                        if class_name not in self.registry:
                                            self.registry[class_name] = module_name
                                        else:
                                            print(f"Duplicate class '{class_name}' in: {module_name} & {self.registry[class_name]}")
                            except Exception as e:
                                print(f"Failed to import {class_name} from {module_name}: {e}")
                                continue

    def register_class(self, cls: Type) -> Type:
        class_name = cls.__name__
        self.registry[class_name] = cls.__module__
        self.cache[class_name] = cls
        return cls

    def class_of(self, class_name: str) -> Type:
        if class_name in self.cache:
            return self.cache[class_name]

        if class_name not in self.registry:
            raise KeyError(f"Class '{class_name}' not found in registry.")

        module_name = self.registry[class_name]
        try:
            module = importlib.import_module(module_name)
            class_ref = getattr(module, class_name)
            self.cache[class_name] = class_ref
            return class_ref
        except ImportError as e:
            raise ImportError(f"Could not import '{class_name}' from '{module_name}': {e}")

    def list_classes(self) -> List[str]:
        return sorted(self.registry.keys())