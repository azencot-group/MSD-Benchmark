import time

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from msd.evaluations.abstract_evaluator import AbstractEvaluator

class ComputationalEvaluator(AbstractEvaluator):
    def __init__(self, initializer, dataset_type, evaluation_manager):
        super().__init__(initializer, dataset_type, evaluation_manager)
        self.batch_size = 256
        self.dataset, self.data_loader = self.initializer.get_dataset(self.dataset_type, loaders=True, labels=True)

    def get_params_num(self):
        # Number of parameters
        params_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return params_num

    def calculate_flops(self):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                     with_flops=True) as prof:
            with record_function("model_inference"):
                X = next(iter(self.data_loader))[0]
                self.model(torch.randn(self.batch_size, *X.shape[1:]).to(self.device))
        Gflops = sum([f.flops for f in prof.key_averages()]) / 1e6  # In GFLOPs
        return Gflops

    # def get_model_one_epoch_time_in_milisec(self):
    #     # _ = self.initializer.get_trainer()  # todo
    #     start_time = time.time()
    #     # trainer.train_step()
    #     epoch_time = time.time() - start_time
    #
    #     return epoch_time

    def get_model_swap_time_in_milisec(self):
        x = next(iter(self.data_loader))[0]
        z1 = self.model.encode(torch.randn(self.batch_size, *x.shape[1:]).to(self.device))
        z2 = self.model.encode(torch.randn(self.batch_size, *x.shape[1:]).to(self.device))

        start_time = time.time()
        self.model.swap_channels(z1, z2, [0])
        action_time = time.time() - start_time

        return action_time

    def get_model_generation_time_in_milisec(self):
        x = next(iter(self.data_loader))[0]
        z = self.model.encode(torch.randn(self.batch_size, *x.shape[1:]).to(self.device))

        start_time = time.time()
        self.model.sample(z)
        action_time = time.time() - start_time

        return action_time

    def eval(self, epoch) -> (dict[str, float], pd.DataFrame):
        params_num = self.get_params_num()
        # epoch_time = self.get_model_one_epoch_time_in_milisec()
        Gflops = self.calculate_flops()
        swap_time = self.get_model_swap_time_in_milisec()
        swap_gen_time = self.get_model_generation_time_in_milisec()

        res_dict = {'Number of Parameters': params_num,
                    # 'Epoch Time (Milliseconds)': epoch_time,
                    'GFLOPs': Gflops,
                    'Time for Swap Function (Milliseconds)': swap_time,
                    'Time for Generation Function (Milliseconds)': swap_gen_time}

        res_pd = pd.DataFrame(index=list(res_dict.keys()), data=list(res_dict.values()))

        return res_dict, res_pd

    @property
    def name(self):
        return 'computational_evaluator'
