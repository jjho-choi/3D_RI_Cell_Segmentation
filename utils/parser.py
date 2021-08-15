import argparse


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--config', type=str)

    def get_args(self):
        args = self.parser.parse_args()
        return args
