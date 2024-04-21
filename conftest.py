import os


try:
    from metadrive import SafeMetaDriveEnv

    meta_drive_env_available = True
except ImportError:
    meta_drive_env_available = False


def pytest_ignore_collect(path, config):
    if os.path.basename(path) == 'meta_drive_env.py' and not meta_drive_env_available:
        return True
    return False
