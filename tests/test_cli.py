import os

from typer.testing import CliRunner

from omnisafe import app


runner = CliRunner()
base_path = os.path.dirname(os.path.abspath(__file__))


def test_benchmark():
    result = runner.invoke(
        app,
        [
            'benchmark',
            'test_benchmark',
            '2',
            os.path.join(base_path, './saved_source/benchmark_config.yaml'),
        ],
    )
    assert result.exit_code == 0


def test_train():
    result = runner.invoke(
        app,
        [
            'train',
            '--algo',
            'PPO',
            '--total-steps',
            '1024',
            '--vector-env-nums',
            '1',
            '--custom-cfgs',
            'algo_cfgs:update_cycle',
            '--custom-cfgs',
            '512',
        ],
    )
    assert result.exit_code == 0


def test_train_config():
    result = runner.invoke(
        app, ['train-config', os.path.join(base_path, './saved_source/train_config.yaml')]
    )
    assert result.exit_code == 0


def test_eval():
    result = runner.invoke(
        app,
        [
            'eval',
            os.path.join(base_path, './saved_source/PPO-{SafetyPointGoal1-v0}'),
            '--num-episode',
            '1',
            '--width',
            '1',
            '--height',
            '1',
        ],
    )
    assert result.exit_code == 0
