import argparse

import pandas as pd


def split_version(df):
    def split_version_value(version):
        try:
            major, minor, fix = version.split(".")
        except ValueError:
            major, minor = version.split(".")
            fix = ""
        return major, minor, fix, "beta" in fix

    df[
        [
            "app_version_major",
            "app_version_minor",
            "app_version_fix",
            "app_version_beta",
        ]
    ] = (
        df["appVersion"].map(split_version_value).tolist()
    )

    return df.drop("appVersion", axis=1)


def split_os_version(df):
    with pd.option_context("mode.chained_assignment", None):
        df[["os", "version"]] = df["osAndVersion"].str.split("-", expand=True)
    return df.drop("osAndVersion", axis=1)


def sort_sizes(df):
    with pd.option_context("mode.chained_assignment", None):
        df["size_sorted"] = df["size"].map(lambda x: "x".join(sorted(x.split("x"))))
    return df.drop("size", axis=1)


def last_winning_bids(df):
    # Last winning bid by device_id
    df["last_win_bid"] = df.groupby(["deviceId"], as_index=False).winBid.shift(1)
    # Last winning bid by c1
    df["last_win_bid_c1"] = df.groupby(["deviceId", "c1"], as_index=False).winBid.shift(
        1
    )
    # Last winning bid by unitDisplayType
    df["last_win_bid_udt"] = df.groupby(
        ["deviceId", "unitDisplayType"], as_index=False
    ).winBid.shift(1)
    # Last winning bid by size
    df["last_win_bid_size"] = df.groupby(
        ["deviceId", "size_sorted"], as_index=False
    ).winBid.shift(1)

    return df


def datetime_to_dow_hour(df):
    with pd.option_context("mode.chained_assignment", None):
        df["day_of_week"] = pd.to_datetime(df["eventTimestamp"], unit="ms").dt.dayofweek
        df["hour"] = pd.to_datetime(df["eventTimestamp"], unit="ms").dt.dayofweek
    #        df["minute"] =  pd.to_datetime(betas["eventTimestamp"], unit="ms").dt.dayofweek
    return df.drop("eventTimestamp", axis=1)


def preprocess_data(data_path):
    mappings = [
        split_os_version,
        split_version,
        sort_sizes,
        datetime_to_dow_hour,
        last_winning_bids,
    ]
    df = (
        pd.read_csv(data_path)
        .sort_values(by=["deviceId", "eventTimestamp"])
        .drop_duplicates(subset=["deviceId", "eventTimestamp"], keep="first")
    )
    # Fill missing values
    df.connectionType.fillna("UNKNOWN", inplace=True)
    for mapping in mappings:
        print(mapping.__name__)
        print(df)
        df = mapping(df)
    return df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--output", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preprocessed_df = preprocess_data(args.data)
    preprocessed_df.to_csv(args.output)
