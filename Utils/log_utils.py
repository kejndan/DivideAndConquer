import time
import os
import mlflow
from shutil import copyfile


def log_metrics(path, metrics, step, cfg, with_mlflow=True):
    time_step = str(time.time()).replace('.', '')[:13]
    if not os.path.exists(os.path.join(cfg.metrics_dir, '/'.join(path.split('/')[:-1]))):
        os.makedirs(os.path.join(cfg.metrics_dir, '/'.join(path.split('/')[:-1])))
    with open(os.path.join(cfg.metrics_dir, path), 'a') as f:
        f.write(f'{time_step} {metrics} {step}\n')
    if with_mlflow:
        mlflow.log_metric(path, metrics, step)


def log_artifacts(artifact_path, file_name, cfg, with_mlflow=True):
    if not os.path.exists(os.path.join(cfg.metrics_dir, 'artifacts')):
        os.makedirs(os.path.join(cfg.metrics_dir, 'artifacts'))
    copyfile(artifact_path, os.path.join(os.path.join(cfg.metrics_dir, 'artifacts'), file_name))
    if with_mlflow:
        mlflow.log_artifact(artifact_path)
