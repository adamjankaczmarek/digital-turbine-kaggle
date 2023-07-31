# pylint: disable=C0103, C0116, C0114
import argparse

import numpy as np
import pandas as pd
import xgboost
from matplotlib import pyplot
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--model_out", type=str)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def split_data(df):
    train, val, *_ = np.split(
        df.sample(frac=1, random_state=42),
        [
            int(0.8 * len(df)),
        ],
    )
    return train, val


def main(data, model_out, limit):
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
    if limit:
        df = df[:limit]

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

    feature_names = full_pipeline.get_feature_names_out()

    model = xgboost.XGBRegressor(n_estimators=300)
    model.fit(
        train_X,
        train_y,
        eval_metric="rmse",
        eval_set=[(train_X, train_y), (val_X, val_y)],
        verbose=1,
    )
    val_pred_y = model.predict(val_X)
    model.save_model(model_out)

    results = model.evals_result()
    epochs = len(results["validation_0"]["rmse"])
    x_axis = range(0, epochs)

    fig, ax = pyplot.subplots(figsize=(12, 12))
    ax.plot(x_axis, results["validation_0"]["rmse"], label="Train")
    ax.plot(x_axis, results["validation_1"]["rmse"], label="Test")
    ax.legend()
    pyplot.ylabel("RMSE Loss")
    pyplot.title("XGBoost RMSE Loss")
    pyplot.savefig("xgb.png")

    importances = model.get_booster().get_score(importance_type="weight")
    name_map = {f"f{i}": feature_name for i, feature_name in enumerate(feature_names)}
    most_important_features = [
        name_map[f] for f, _ in sorted(importances.items(), key=lambda x: -x[1])
    ]
    print(most_important_features)

    xgboost.plot_importance(model, max_num_features=10).set_yticklabels(
        most_important_features[:10]
    )
    pyplot.gcf().subplots_adjust(left=0.5)
    pyplot.savefig("xgb_importances.png")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
