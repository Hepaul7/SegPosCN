import torch
from models import CWSPOSTransformer


def train(model: CWSPOSTransformer,
          lr: float, input_ids: torch.Tensor,
          attention_masks: torch.tensor,
          labels: torch.tensor,
          epochs: int):
    """
    Train model
    :param model:
    :param lr:
    :param input_ids:
    :param attention_masks:
    :param labels:
    :param epochs:
    :return:
    """




