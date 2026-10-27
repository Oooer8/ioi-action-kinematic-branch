import pytest
import torch

from ioi_action_branch.config import tiny_config
from ioi_action_branch.model import create_model
from ioi_action_branch.smoke import synthetic_batch


@pytest.fixture
def cfg(request):
    return tiny_config(getattr(request, "param", "action_branch"))


@pytest.fixture
def model(cfg):
    torch.set_num_threads(2)
    torch.manual_seed(22)
    return create_model(cfg, allow_random=True)


@pytest.fixture
def batch(cfg):
    return synthetic_batch(cfg)
