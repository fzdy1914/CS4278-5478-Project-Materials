import argparse

from ray.tune.logger import pretty_print

from gym_duckietown.wrappers import NormalizeWrapper, ResizeWrapper, StackWrapper
from gymnasium.wrappers import EnvCompatibility

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

from gym_duckietown.envs import GuidedBotEnv

import torch
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=10000, help="Number of iterations to train."
)
parser.add_argument(
    "--no-tune",
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.",
    default=True
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
    default=True
)


class TorchCustomModel(TorchModelV2, nn.Module):
    """Example of a PyTorch custom model that just delegates to a fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        self._convs = nn.Sequential(
            nn.ZeroPad2d((2, 2, 2, 2)),
            nn.Conv2d(12, 32, kernel_size=8, stride=4),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(32, 32, kernel_size=4, stride=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 256, kernel_size=11, stride=1),
            nn.LeakyReLU(),
        )

        self._mlp = nn.Sequential(
            nn.Linear(in_features=256 + 3, out_features=32),
            nn.LeakyReLU(),
        )

        self._policy = nn.Sequential(
            nn.Linear(in_features=32, out_features=2),
            nn.Tanh(),
        )

        self._value = nn.Sequential(
            nn.Linear(in_features=32, out_features=1),
        )

    def forward(self, input_dict, state, seq_lens):
        _cnn_feature = self._convs(input_dict["obs"][0].permute(0, 3, 1, 2))
        _all_feature = torch.cat((_cnn_feature.flatten(1), input_dict["obs"][1]), dim=1)
        self._feature = self._mlp(_all_feature)
        return self._policy(self._feature), []

    def value_function(self):
        return self._value(self._feature).reshape([-1])


def launch_and_wrap_env(ctx):
    env = GuidedBotEnv(
        domain_rand=False,
        max_steps=100,
        map_name="map1_0",
        randomize_maps_on_reset=True
    )

    env = EnvCompatibility(env)
    env = ResizeWrapper(env)
    env = StackWrapper(env)
    env = NormalizeWrapper(env)

    return env


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    ray.init()

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel
    )

    register_env('MyDuckietown', launch_and_wrap_env)

    config = (
        PPOConfig()
        # or "corridor" if registered above
        .environment("MyDuckietown")
        .framework("torch")
        .rollouts(num_rollout_workers=6, create_env_on_local_worker=True)
        .training(
            model={
                "custom_model": "my_model",
            }
        )
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=1)
    )

    stop = {
        "training_iteration": args.stop_iters,
    }

    if args.no_tune:
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        config.lr = 5e-5
        config.train_batch_size = 4096
        config.gamma = 0.99
        config.evaluation_interval = 25
        config.evaluation_num_episodes = 5

        algo = config.build()
        # run manual training loop and print results after each iteration
        for _ in range(args.stop_iters):
            result = algo.train()
            print(pretty_print(result))
            checkpoint_dir = algo.save()
            print(checkpoint_dir)
        algo.stop()
    else:
        # automated run with Tune and grid search and TensorBoard
        print("Training automatically with Ray Tune")
        tuner = tune.Tuner(
            args.run,
            param_space=config.to_dict(),
            run_config=air.RunConfig(stop=stop),
        )
        results = tuner.fit()

        if args.as_test:
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)

    ray.shutdown()