import buffer as bf
import wandb
import checkpoint
import driver
from statistics import mean, stdev


class Evaluator:
    """
    Evaluates policies and saves them if they improve

    Args:
        make_env: make_env() -> env a function that returns the environment
        seed: random seed
        plot: plot the episode returns if True
    """
    def __init__(self, env):
        self.best_mean_return = -999999999
        env, buffer = bf.wrap(env)
        self.env = env
        self.buffer = buffer

    def sample_policy_returns(self, policy, samples, render=False):
        """
        samples n trajectories from environment using policy

        Args:
            policy: policy(state) -> action, the policy to evaluate
            samples: number of episodes to sample
            render: render while running
        Returns:
             mean_return
             stdev_return
        """
        start = len(self.buffer.trajectories)
        # run test trajectories and compute the returns
        for _ in range(samples):
            driver.episode(self.env, policy, render=render)
        returns = [traj_info['return'] for traj_info in self.env.trajectory_info[start:]]
        mean_return = mean(returns)
        stdev_return = stdev(returns)
        self.buffer.clear()
        return mean_return, stdev_return

    def evaluate(self, policy, run_dir, params, sample_n=10, render=False):
        """
        Evaluate the policy and save if improved

        Args:
            policy: policy(state) -> action to evaluate
            run_dir: directory to save parameters to
            params: dict of parameters to write..

            .. code-block:: python
              policy_net = nn.Linear(4, 1)
              optim = nn.optim.SGD(policy_net.parameters(), lr=1e-4)
              params={'policy_net': policy_net, 'optim': optim}

            sample_n: number of episodes to sample
            render: render the environment while running if True

        Returns:
            mean_return
            stdev_return

        """
        mean_return, stdev_return = self.sample_policy_returns(policy, sample_n, render)

        wandb.run.summary["last_mean_return"] = mean_return
        wandb.run.summary["last_stdev_return"] = stdev_return

        # checkpoint policy if mean return is better
        if mean_return > self.best_mean_return:
            best_mean_return = mean_return
            wandb.run.summary["best_mean_return"] = best_mean_return
            wandb.run.summary["best_stdev_return"] = stdev_return
            checkpoint.save(run_dir, 'best', **params)

        return mean_return, stdev_return

    def demo(self, demo, policy):
        """
        Infinite loop that runs policy on environment, and renders
        Args:
            demo: if true enters infinite loop
            policy: policy(state) -> action to run

        """
        if demo:
            while True:
                driver.episode(self, policy, render=True)
                self.buffer.clear()