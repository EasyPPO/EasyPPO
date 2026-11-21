"""Launch the multi-node asynchronous Terminus-2 PPO recipe with the full_detach critic."""

from examples.easyppo.terminus_ppo import main


if __name__ == "__main__":
    main("full_detach")
