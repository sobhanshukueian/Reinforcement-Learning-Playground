import os
import json
import numpy as np
from copy import deepcopy
import random
import glob


import torch

def count_parameters(model: torch.nn.Module) -> None:
    # table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        # table.add_row([name, params])
        total_params+=params
    # print(table)
    print(f"Total Trainable Params: {total_params}")


# sets seeds for random number generators to achieve reproducibility in experiments
def reproducibility(SEED: int, env=None) -> None:
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    random.seed(SEED)
    if env:
        env.action_space.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

def save_checkpoint(save_dir: str, checkpoint) -> None:
    torch.save(self.state_dict(), save_dir)

# saves the model's weights, optimizer state, and the current training epoch to a specified directory
def save(save_dir: str, model: torch.nn.Module, epoch: int, optimizer: torch.optim, model_name="last") -> None:
    print("Saving model in {}".format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # save model
    save_ckpt_dir = os.path.join(save_dir, 'weights')
    if not os.path.exists(save_ckpt_dir):
        os.makedirs(save_ckpt_dir)
    filename = os.path.join(save_ckpt_dir, f'{model_name}.pt')

    # save ckpt
    ckpt = {
            'model': deepcopy(model).half(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            }
    torch.save(ckpt, filename)

# creates an optimizer (SGD, Adam, or AdamW) for the model, considering options for resuming training and adjusting hyperparameters.
def get_optimizer(model_parameters: torch.nn.Module, resume: str, ckpt: dict, optimizer: str, lr0=0.001, momentum=0.937, weight_decay=0.00005):
    assert optimizer == 'SGD' or 'Adam' or 'LARS', 'ERROR: unknown optimizer, use SGD defaulted'
    if optimizer == 'SGD':
        optim = torch.optim.SGD(model_parameters, lr=lr0, momentum=momentum, weight_decay=weight_decay)
    elif optimizer == 'Adam':
        optim = torch.optim.Adam(model_parameters, lr=lr0, weight_decay=1e-6)
    elif optimizer == 'AdamW':
        optim = torch.optim.AdamW(model_parameters, lr=lr0, weight_decay=weight_decay)
    if resume:
        optim.load_state_dict(ckpt['optimizer'])

    print(f"{'optimizer:'} {type(optim).__name__}")
    return optim

# loads a previously saved model and optimizer state from a specified directory, enabling training continuation from a checkpoint.
def load(model: torch.nn.Module, optimizer: torch.optim, resume: str, model_name="last") -> list:
    # Find the most recent saved checkpoint in search_dir
    print("Loading Model from {}".format(resume))
    checkpoint_list = glob.glob(f'{resume}/**/{model_name}*.pt', recursive=True)
    checkpoint_path = max(checkpoint_list, key=os.path.getctime) if checkpoint_list else ''
    assert os.path.isfile(checkpoint_path), f'the checkpoint path is not exist: {checkpoint_path}'
    print(f'Resume training from the checkpoint file :{checkpoint_path}')
    #load checkpoint
    ckpt = torch.load(checkpoint_path)

    model.load_state_dict(ckpt['model'].state_dict())
    optimizer.load_state_dict(ckpt['optimizer'])
    start_epoch = ckpt['epoch'] + 1

    return [model, optimizer, start_epoch]


# prepares a directory for saving the results of the training run, including storing the experiment's arguments in a text file
def prepare_save_dir(args: dict) -> str:
    temm=0
    tmp_save_dir = os.path.join(args.save_dir, "run")
    while os.path.exists(tmp_save_dir):
        tmp_save_dir = os.path.join(args.save_dir, "run")
        temm+=1
        tmp_save_dir += (str(temm))
    results_dir = tmp_save_dir
    os.makedirs(results_dir)

    # Save Arguments in results dir
    with open(os.path.join(results_dir, "args.txt"), "w") as text_file:
        for key, value in vars(args).items():
            text_file.write(f"{key}: {value}\n")

    print("Save Project in {} directory.".format(results_dir))
    return results_dir

class EMA():
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.alpha + (1 - self.alpha) * new
