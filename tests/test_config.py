import config as cfg


def test_argparse_defaults():
    parser = cfg.ArgumentParser(description='Test argparse')
    parser.add_argument('--env_name', type=str, default='CartPoleContinuous-v1')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--min_variance', type=float, default=0.05)
    config = parser.parse_args([])

    assert config.env_name == 'CartPoleContinuous-v1'
    assert config.demo == False
    assert config.epochs == 2000
    assert config.min_variance == 0.05


def test_config_file():
    parser = cfg.ArgumentParser(description='Test argparse')
    parser.add_argument('--config', type=str)
    parser.add_argument('--env_name', type=str, default='CartPoleContinuous-v1')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--min_variance', type=float, default=0.05)
    config = parser.parse_args(['--config', 'test_config.yaml'])

    assert config.env_name == 'CartPoleContinuous-v2'
    assert config.demo == True
    assert config.epochs == 1000
    assert config.min_variance == 0.5


def test_command_overrides():
    parser = cfg.ArgumentParser(description='Test argparse')
    parser.add_argument('--config', type=str)
    parser.add_argument('--env_name', type=str, default='CartPoleContinuous-v1')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--min_variance', type=float, default=0.05)
    config = parser.parse_args(['--config', 'test_config.yaml',
                                '--env_name', 'CartPoleContinuous-v3',
                                '--epochs', '4000',
                                '--min_variance', '0.2'])

    assert config.env_name == 'CartPoleContinuous-v3'
    assert config.demo == True
    assert config.epochs == 4000
    assert config.min_variance == 0.2