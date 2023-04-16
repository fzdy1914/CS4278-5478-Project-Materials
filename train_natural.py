import argparse

from gym_duckietown.new_wrappers import NormalizeWrapper, ResizeWrapper, StackWrapper
from gymnasium.wrappers import EnvCompatibility

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.registry import register_env

from gym_duckietown.envs import *

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


def launch_and_wrap_env(ctx):
    env = DirectedBotEnv(
        direction=-1,
        domain_rand=False,
        max_steps=100,
        map_name="map2_0",
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

    register_env('MyDuckietown', launch_and_wrap_env)

    config = (
        PPOConfig()
        # or "corridor" if registered above
        .environment("MyDuckietown")
        .framework("torch")
        .rollouts(num_rollout_workers=6, create_env_on_local_worker=True)
        .training()
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
        config.train_batch_size = 2048
        config.gamma = 0.99
        config.evaluation_interval = 25
        config.evaluation_num_episodes = 5

        algo = config.build()
        algo.restore("D:\\natural_result\\checkpoint_000419")
        # run manual training loop and print results after each iteration
        for _ in range(args.stop_iters):
            result = algo.train()
            checkpoint_dir = algo.save("D:\\natural_result")
            print("episode_reward_mean", result["episode_reward_mean"])
            print("episode_len_mean", result["episode_len_mean"])
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
