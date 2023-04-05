"""Example."""
import logging

import numpy as np
import typer
from gym_duckietown.envs import DuckietownEnv

from .lane_follower import LaneFollower

logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def example():
    """Example usage of the lane follower."""
    intentions = {
        (1, 1): "forward",
        (1, 2): "forward",
        (1, 3): "forward",
        (1, 4): "forward",
        (1, 5): "forward",
        (1, 6): "forward",
        (1, 7): "left",
        (2, 7): "forward",
        (3, 7): "forward",
        (4, 7): "forward",
        (5, 7): "forward",
        (6, 7): "forward",
        (7, 7): "forward",
    }
    env = DuckietownEnv(
        domain_rand=False,
        max_steps=1800,
        map_name="map2_1",
        seed=12,
        user_tile_start=(1, 1),
        goal_tile=(7, 7),
        randomize_maps_on_reset=False,
    )
    try:
        env.render()
        rewards = []
        actions = []
        map_img, goal, start_pos = env.get_task_info()
        # NOTE: Here is where the lane follower is constructed
        robot = LaneFollower(intentions, map_img, goal, visualize=True)
        action = [0, 0]
        obs, reward, done, info = env.step(action)
        for _ in range(1800):
            env.render()
            if done:
                break
            action = robot(obs, info, action)
            obs, reward, done, info = env.step(action)
            done = done or (info["curr_pos"][0] == 10 and info["curr_pos"][1] == 1)
            rewards.append(reward)
            actions.append(action)
            logger.debug(
                "step_count = %s, reward=%.3f", env.unwrapped.step_count, reward
            )
        avg_reward = np.mean(rewards)
        logger.info("Average reward: %.3f", avg_reward)
    except Exception as ex:
        logger.exception("Exception raised during evaluation.")
        return dict(
            status="exception",
            exception="%s" % ex,
        )
    finally:
        env.render(close=True)


def main():
    app()


if __name__ == "__main__":
    main()
