import wandb
import rl
import torch_utils
from rich.progress import Progress
import wandb_utils


def log(config, env, policy, project, env_name, log_episodes, test_steps, video_episodes=3, test_episodes=16):
    """

    Args:
        config: config to send to wandb
        env: gym environment
        policy: logging policy
        project: wandb project name to write to
        env_name: environment name
        log_episodes: number of episodes to log
        video_episodes: number of episodes to capture mp4 video for
        test_steps: number of steps between evaluations
        test_episodes: number of episodes to evaluate per test

    """
    """ logging loop """
    wandb.init(project=project, config=config)
    buffer = rl.ReplayBuffer()

    episodes_captured = 0
    vidstream = []
    test_number = 1

    with Progress() as progress:
        run = progress.add_task('Generating', total=log_episodes)
        for step, s, a, s_p, r, d, i, m in rl.step(env, policy, buffer, render=True):

            if episodes_captured < log_episodes:
                buffer.append(s, a, s_p, r, d)
                if episodes_captured < video_episodes:
                    vidstream.append(m['frame'])
                else:
                    rl.global_render = False
                episodes_captured += 1 if d else 0
                progress.update(run, advance=1 if d else 0)
            else:
                break

            """ test """
            if step > test_steps * test_number:
                stats = rl.evaluate(env, policy, sample_n=test_episodes)
                stats = wandb_utils.log_test_stats(stats, test_number)
                test_number += 1
                wandb.log(stats)

    """ log transitions """
    filename = f'./{env_name}_{len(buffer)}.pkl'
    rl.save(buffer, filename)
    wandb.run.tags = [*wandb.run.tags, filename]

    """ log video """
    video_filename = f'./{env_name}_{len(buffer)}.mp4'
    torch_utils.write_mp4(video_filename, vidstream)
    wandb.log({'video': wandb.Video(video_filename, fps=4, format="mp4")})
