import rl
from argparse import ArgumentParser
from rich import print
from rich.table import Table


if __name__ == '__main__':
    parser = ArgumentParser(description='configuration switches')
    parser.add_argument('--filename', type=str, default='buffer.h5')
    config = parser.parse_args()
    buffer = rl.OnDiskReplayBuffer.load(config.filename)

    reward_pos = 0
    reward_neg = 0
    reward_zero = 0
    for row in buffer.transitions:
        r = row['reward']
        if r > 0:
            reward_pos += 1
        if r < 0:
            reward_neg += 1
        if r == 0:
            reward_zero += 1

    table = Table(title="Buffer stats")

    table.add_column("Stat", justify="right", style="cyan", no_wrap=True)
    table.add_column("Title", style="magenta")

    table.add_row("Episodes", f"{len(buffer.episodes)}")
    table.add_row("Transitions", f"{len(buffer)}")
    table.add_row("Transitions with + reward", f"{reward_pos}")
    table.add_row("Transitions with - reward", f"{reward_neg}")
    table.add_row("Transitions with 0 reward", f"{reward_zero}")

    print(table)
    buffer.close()