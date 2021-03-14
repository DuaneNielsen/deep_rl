from argparse import ArgumentParser
import os
from subprocess import call, Popen

if __name__ == '__main__':
    parser = ArgumentParser(description='kicks off multiple wandb scripts')
    parser.add_argument('command', nargs=3)
    args = parser.parse_args()

    Popen(args.command,
          stdout=open('null1', 'w'),
          stderr=open('wandb.log', 'a'),
          start_new_session=True)
