import glob
import os
import pprint
import time

import hydra
import torch
import torch.profiler
from omegaconf import DictConfig, OmegaConf
from torch.profiler import tensorboard_trace_handler
from tqdm import tqdm

import wandb
from bend_hybrid.data_downstream import get_data

WANDB_KEY = os.getenv("WANDB_KEY", None)


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the wandb sweep."""

    wandb.login(anonymous="allow", key=WANDB_KEY)

    cfg.embeddings_output_dir = os.path.join(
        cfg.embeddings_output_dir, cfg.task.task, cfg.embedder
    )

    sweep_config = OmegaConf.to_container(cfg.sweep, resolve=True)
    parameters_dict = {
        key: ({"values": value} if isinstance(value, list) else {"value": value})
        for key, value in sweep_config["parameters"].items()
    }
    sweep_config["parameters"] = parameters_dict
    pprint.pprint(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
    wandb.agent(sweep_id, run_experiment)


def run_experiment(config=None):
    """Run the experiment with the given config."""

    try:
        with wandb.init(config=config) as run:
            config = wandb.config

            train_loader, _, _ = get_data(
                data_dir=config.data_dir,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                prefetch_factor=config.prefetch_factor,
                pin_memory=config.pin_memory,
                shuffle=config.shuffle,
                shardshuffle=config.shardshuffle,
            )

            wait, warmup, repeat = 1, 1, 0
            active = max(1, round(config.n_samples / config.batch_size))
            total_steps = (wait + warmup + active) * (1 + repeat)

            with torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=wait, warmup=warmup, active=active, repeat=repeat
                ),
                on_trace_ready=tensorboard_trace_handler("wandb/latest-run/tbprofile"),
                profile_memory=True,
                with_stack=False,
            ) as profiler:

                start_time = time.time()
                for step, _ in tqdm(enumerate(train_loader), total=total_steps):
                    if step == wait + warmup - 1:
                        start_time = time.time()
                    if step >= total_steps:
                        break
                    profiler.step()

                total_samples = config.batch_size * (active * (1 + repeat))

                wandb.log({"samples/sec": total_samples / (time.time() - start_time)})

            profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
            profile_art.add_file(
                glob.glob("wandb/latest-run/tbprofile/*.pt.trace.json")[0],
                "trace.pt.trace.json",
            )
            run.log_artifact(profile_art)
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    main()  # pylint: disable=E1120
