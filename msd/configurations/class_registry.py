# from typing import Any, Dict, Type, List
# import ast
# import importlib
# import os
# from os import path as osp
#
# CLASS_REGISTRY = {
#     # External modules (manually registered)
#     "MSELoss": "torch.nn",
#     "CrossEntropyLoss": "torch.nn",
#     "SGD": "torch.optim",
#     "Adam": "torch.optim",
#     "DataLoader": "torch.utils.data",
#     "StepLR": "torch.optim.lr_scheduler",
#     "DecisionTreeClassifier": "sklearn.tree",
#     "RandomForestClassifier": "sklearn.ensemble",
#     "GradientBoostingClassifier": "sklearn.ensemble",
#     "FramewiseCrossEntropyLoss": "msd.evaluations.classifiers.framewise_cross_entropy",
# }
# CLASS_CACHE = {}  # Stores { "ClassName": ClassReference }
#
# def scan_and_generate_registry(package_root: str) -> Dict[str, str]:
#     """
#     Statically scan Python files in the given directory to extract top-level class definitions
#     and generate a mapping from class names to their corresponding module import paths.
#
#     :param package_root: Root folder of the package (e.g., './msd').
#     :return: Dictionary mapping class names to their importable module paths.
#     """
#     class_registry: Dict[str, str] = {}
#
#     for dirpath, _, filenames in os.walk(package_root):
#         for filename in filenames:
#             if filename.endswith(".py") and not filename.startswith("_"):
#                 filepath = osp.join(dirpath, filename)
#                 rel_path = osp.relpath(filepath, package_root).replace(os.sep, ".")
#                 module_name = osp.splitext(rel_path)[0]
#
#                 try:
#                     with open(filepath, "r", encoding="utf-8") as f:
#                         node = ast.parse(f.read(), filename=filepath)
#                 except SyntaxError as e:
#                     print(f"Skipping {filepath} due to SyntaxError: {e}")
#                     continue
#
#                 for item in node.body:
#                     if isinstance(item, ast.ClassDef):
#                         class_name = item.name
#                         try:
#                             module = importlib.import_module(module_name)
#                             cls = getattr(module, class_name, None)
#                             super_cls = [c.__name__ for c in cls.__mro__]
#
#                             if cls and isinstance(cls, type) and 'MSDComponent' in super_cls:
#                                 if class_name not in class_registry:
#                                     class_registry[class_name] = module_name
#                                 else:
#                                     print(f"Duplicate class '{class_name}' in: {module_name} & {class_registry[class_name]}")
#
#                         except Exception as e:
#                             print(f"Failed to import {class_name} from {module_name}: {e}")
#                             continue
#                         #
#                         # if class_name not in class_registry:
#                         #     class_registry[class_name] = module_name
#                         # else:
#                         #     print(f"Duplicate class name '{class_name}' in: {module_name} & {class_registry[class_name]}")
#
#     return class_registry
#
#
# def register_class(cls: Type) -> Type:
#     """
#     Decorator to register a class into the global class registry and cache it.
#
#     :param cls: Class to register.
#     :return: The original class (for decorator chaining).
#     """
#     class_name = cls.__name__
#     CLASS_REGISTRY[class_name] = cls.__module__  # Store module path
#     CLASS_CACHE[class_name] = cls  # Cache the actual class
#     return cls
#
#
# def class_of(class_name: str) -> Type:
#     """
#     Retrieve a class by name using the registry and import it dynamically if needed.
#
#     :param class_name: Name of the class to load.
#     :return: The class type associated with the name.
#     :raises KeyError: If the class name is not found in the registry.
#     :raises ImportError: If the corresponding module cannot be imported.
#     """
#     if class_name in CLASS_CACHE:
#         return CLASS_CACHE[class_name]
#
#     if class_name not in CLASS_REGISTRY:
#         raise KeyError(f"Class '{class_name}' not found in CLS_REGISTRY.")
#
#     module_name = CLASS_REGISTRY[class_name]  # Get module path
#
#     try:
#         module = importlib.import_module(module_name)
#         class_ref = getattr(module, class_name)
#         CLASS_CACHE[class_name] = class_ref
#         return class_ref
#
#     except ImportError as e:
#         raise ImportError(f"Could not import '{class_name}' from '{module_name}': {e}")
#

from typing import Any, Dict, Type, List
import ast
import importlib
import os
from os import path as osp

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