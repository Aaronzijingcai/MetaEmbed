# hack: ignore space check
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

import datasets

datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory=".": True

import os
from pathlib import Path

import configue
import torch
import typer
from colpali_engine.trainer.colmodel_training import (
    ColModelTraining,
    ColModelTrainingConfig,
)
from colpali_engine.utils.gpu_stats import print_gpu_utilization

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(config_file: Path) -> None:
    """
    Training script for ColVision models.

    Args:
        config_file (Path): Path to the configuration file.
    """
    print_gpu_utilization()

    print("Loading config")
    config = configue.load(config_file, sub_path="config")

    print("Creating Setup")
    if isinstance(config, ColModelTrainingConfig):
        training_app = ColModelTraining(config, config_file)
    else:
        raise ValueError("Config must be of type ColModelTrainingConfig")

    training_app.init_trainer()

    if config.run_eval_before_train:
        print("Evaluating model before training")
        training_app.eval()

    if config.run_train:
        print("Training model")
        training_app.train()
        training_app.save()
        os.system(
            f"cp {config_file} {training_app.config.output_dir}/training_config.yml"
        )
        # add run_eval in the end, so even if it fails we still have the model saved
        if config.run_eval:
            print("Evaluating model")
            training_app.eval()

    print("Done!")


if __name__ == "__main__":
    app()
