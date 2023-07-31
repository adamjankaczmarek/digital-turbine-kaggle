# pylint: disable=C0103, C0116, C0114
import argparse

import numpy as np
import pandas as pd
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--model_out", type=str)
    return parser.parse_args()


def split_data(df):
    train, val, *_ = np.split(
        df.sample(frac=1, random_state=42),
        [
            int(0.8 * len(df)),
        ],
    )
    return train, val


def main(data, model_out):
    features = [
        "os",
        "version",
        "app_version_major",
        "app_version_minor",
        "app_version_fix",
        "app_version_beta",
        "mediationProviderVersion",
        "day_of_week",
        "hour",
        "c1",
        "c2",
        "c3",
        "c4",
        "size_sorted",
        "last_win_bid",
        "last_win_bid_c1",
        "last_win_bid_udt",
        "last_win_bid_size",
        "connectionType",
        "unitDisplayType",
        "correctModelName",
        "bidFloorPrice",
        "countryCode",
        "bundleId",
        "brandName",
        "sentPrice",  # test if necessary
        #    'mediationProviderVersion'
    ]
    label = "winBid"
    df = pd.read_csv(data)

    cat_attribs = [
        "os",
        "version",
        "app_version_major",
        "app_version_minor",
        "app_version_fix",
        "day_of_week",
        "hour",
        "c1",
        "c2",
        "c3",
        "c4",
        "size_sorted",
        "connectionType",
        "unitDisplayType",
        "correctModelName",
        "countryCode",
        "bundleId",
        "brandName",
        #   'mediationProviderVersion'
    ]
    full_pipeline = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs)],
        remainder="passthrough",
    )

    train, val = split_data(df)
    train_X, train_y = train[features], train[label]
    val_X, val_y = val[features], val[label]
    train_X = train_X.drop("mediationProviderVersion", axis=1)
    encoder = full_pipeline.fit(train_X)
    train_X = encoder.transform(train_X)
    val_X = encoder.transform(val_X)

    model = xgboost.XGBRegressor()
    model.fit(
        train_X, train_y, eval_metric="rmse", eval_set=[(val_X, val_y)], verbose=1
    )
    model.predict(val_X)
    model.save_model(model_out)


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
