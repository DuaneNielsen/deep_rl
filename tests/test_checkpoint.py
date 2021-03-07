import torch
import torch.nn as nn
import checkpoint
from pathlib import Path


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))


def test_save():
    dum_net = DummyModule()
    checkpoint.save(directory='delete', dum_net=dum_net)
    assert Path('delete/dum_net.sd').exists()
    Path('delete/dum_net.sd').unlink()
    Path('delete').rmdir()


def test_save_prefix():
    dum_net = DummyModule()
    checkpoint.save(directory='delete', prefix='best', dum_net=dum_net)
    assert Path('delete/best_dum_net.sd').exists()
    Path('delete/best_dum_net.sd').unlink()
    Path('delete').rmdir()


def test_save_multi():
    dum_net = DummyModule()
    dum_net2 = DummyModule()
    checkpoint.save(directory='delete', prefix='best', dum_net=dum_net, dum_net2=dum_net2)
    assert Path('delete/best_dum_net.sd').exists()
    assert Path('delete/best_dum_net2.sd').exists()
    Path('delete/best_dum_net.sd').unlink()
    Path('delete/best_dum_net2.sd').unlink()
    Path('delete').rmdir()


def test_save_load():
    dum_net = DummyModule()
    dum_net.param[0] = 1.0
    checkpoint.save(directory='delete', dum_net=dum_net)
    assert Path('delete/dum_net.sd').exists()

    dum_net = DummyModule()
    assert dum_net.param[0] == 0.0
    checkpoint.load(directory='delete', dum_net=dum_net)
    assert dum_net.param[0] == 1.0

    Path('delete/dum_net.sd').unlink()
    Path('delete').rmdir()


def test_save_load_prefix():
    dum_net = DummyModule()
    dum_net.param[0] = 1.0
    checkpoint.save(directory='delete', prefix='best', dum_net=dum_net)
    assert Path('delete/best_dum_net.sd').exists()

    dum_net = DummyModule()
    assert dum_net.param[0] == 0.0
    checkpoint.load(directory='delete', prefix='best', dum_net=dum_net)
    assert dum_net.param[0] == 1.0

    Path('delete/best_dum_net.sd').unlink()
    Path('delete').rmdir()


def test_save_load_prefix_multi():
    dum_net = DummyModule()
    dum_net2 = DummyModule()
    dum_net.param[0] = 1.0
    dum_net2.param[0] = 2.0
    checkpoint.save(directory='delete', prefix='best', dum_net=dum_net, dum_net2=dum_net2)
    assert Path('delete/best_dum_net.sd').exists()
    assert Path('delete/best_dum_net2.sd').exists()

    dum_net = DummyModule()
    dum_net2 = DummyModule()
    assert dum_net.param[0] == 0.0
    assert dum_net2.param[0] == 0.0
    checkpoint.load(directory='delete', prefix='best', dum_net=dum_net, dum_net2=dum_net2)
    assert dum_net.param[0] == 1.0
    assert dum_net2.param[0] == 2.0

    Path('delete/best_dum_net.sd').unlink()
    Path('delete/best_dum_net2.sd').unlink()
    Path('delete').rmdir()
