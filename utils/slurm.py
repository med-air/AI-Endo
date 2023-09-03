#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import logging
import pathlib
import shutil

import submitit

from train_resnet import train_resnet
from generate_resnet import generate_features
from train_fusion import train_fusion
from train_transformer import train_trans
from test_transnet import test_model


# Refer to https://github.com/facebookresearch/pytorchvideo/blob/main/tutorials/video_classification_example/slurm.py

def init_and_run(run_fn, run_config):
    os.environ["RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["NODE_RANK"] = os.environ["SLURM_LOCALID"]
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    run_fn(run_config)


def copy_and_run_with_config(run_fn, run_config, directory, **cluster_config):
    working_directory = pathlib.Path(directory) / "jobs" / cluster_config["job_name"]
    ignore_list = [
        "ckpts",
        "runs",
        ".git",
        "tem",
        "jobs",
        "logs",
        "results",
        "__pycache__"
    ]
    shutil.copytree(".", working_directory, ignore=lambda x, y: ignore_list)
    os.chdir(working_directory)
    print(f"Running at {working_directory}")

    executor = submitit.SlurmExecutor(folder=working_directory)
    executor.update_parameters(
        additional_parameters={
            'mail-type': 'ALL',
            'mail-user': 'jfcao@cse.cuhk.edu.hk',
            'qos': "pheng_gpu",
            'account': "pheng_gpu",
        }
    )
    executor.update_parameters(**cluster_config)
    job = executor.submit(init_and_run, run_fn, run_config)
    print(f"job_id: {job}")


def run_all(args):
    args = train_resnet(args)
    args = generate_features(args)

    args = train_fusion(args)
    args = train_trans(args)
    args = test_model(args)

    logging.info("\n\n" + "=="*10)
    logging.info(args)