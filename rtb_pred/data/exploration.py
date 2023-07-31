import argparse

# import sys
import pandas as pd
from ydata_profiling import ProfileReport


def prepare_report(data_path):
    df = pd.read_csv(data_path)
    return ProfileReport(df)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--report", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report = prepare_report(args.data)
    report.to_file(args.report)
