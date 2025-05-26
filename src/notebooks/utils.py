import pandas as pd
import numpy as np
from constants import SANREMO, HOUR_RANGES, NEXT_DAY_RANGES, SERATE
import os
from datetime import datetime, timedelta
from tabulate import tabulate


def load_data_night(
    night_info, serata, datetime_str, TIME_TO_PREDICT, base_path="analyzed_data"
):

    # Creating a dictionary with keys for each night and, for each one, for each hour range
    data_night = {s: {hr: [] for hr in HOUR_RANGES} for s in SERATE}

    # Prepare iteration over all nights selected
    serate_to_search = list(data_night.keys())
    serata_index = serate_to_search.index(TIME_TO_PREDICT["serata"])
    serate_to_search = serate_to_search[: serata_index + 1]

    for serata in serate_to_search:
        for artist in SANREMO["NIGHTS"][serata]["scaletta"]:
            for hour_range in HOUR_RANGES:
                if hour_range in NEXT_DAY_RANGES:
                    date_obj = datetime.strptime(
                        SANREMO["NIGHTS"][serata]["data"], "%Y-%m-%d"
                    )
                    data_next_hour = date_obj + timedelta(days=1)
                    data_next_hour_str = data_next_hour.strftime("%Y-%m-%d")
                    filename = f"{data_next_hour_str}_{hour_range}.csv"
                else:
                    filename = f"{SANREMO['NIGHTS'][serata]['data']}_{hour_range}.csv"

                # Use the name of the night to form the path
                filepath = os.path.join(
                    base_path,
                    serata.replace(" ", "_"),
                    artist.replace(" ", "_"),
                    filename,
                )

                if os.path.exists(filepath):
                    df_hour = pd.read_csv(filepath)
                    df_hour["datetime"] = pd.to_datetime(df_hour["datetime"])
                    cutoff_datetime = pd.to_datetime(datetime_str)
                    df_filtered = df_hour[df_hour["datetime"] < cutoff_datetime]
                    # Save the dataframe within the corresponding night and hour range
                    data_night[serata][hour_range].append(df_filtered)
                else:
                    print(f"File {filepath} not found")

    # Remove empty dataframes for each night and each hour range
    for s in SERATE:
        for hr in HOUR_RANGES:
            data_night[s][hr] = [df for df in data_night[s][hr] if not df.empty]

    return data_night


def aggregate_data(dfs):
    # Creating a dictionary to aggregate data by night and hour range
    dfs_aggregated = {}
    for serata, hours in dfs.items():
        dfs_aggregated[serata] = {}
        for hr, df_list in hours.items():
            dfs_aggregated[serata][hr] = {
                artist: {"positive_count": 0, "negative_count": 0}
                for artist in SANREMO["ARTISTS"]
            }
            for df in df_list:
                for index, row in df.iterrows():
                    artist = row["artist"]
                    sentiment = row["sentiment"]

                    if artist not in dfs_aggregated[serata][hr]:
                        dfs_aggregated[serata][hr][artist] = {
                            "positive_count": 0,
                            "negative_count": 0,
                        }

                    if sentiment == "positive":
                        dfs_aggregated[serata][hr][artist]["positive_count"] += 1
                    elif sentiment == "negative":
                        dfs_aggregated[serata][hr][artist]["negative_count"] += 1
    return dfs_aggregated


def print_aggregated_data(dfs_aggregated, sort_by_positive=True):
    # Function to print aggregated data with customizable sorting
    for serata, hours in dfs_aggregated.items():
        print(f"--- Aggregazione per serata: {serata} ---\n")
        for hr, artists in hours.items():
            print(f"Intervallo orario: {serata} {hr}")
            table = []
            for artist, counts in artists.items():
                table.append(
                    [artist, counts["positive_count"], counts["negative_count"]]
                )
            # Ordina in base alla scelta: positivi o negativi
            if sort_by_positive:
                table_sorted = sorted(table, key=lambda x: x[1], reverse=True)
            else:
                table_sorted = sorted(table, key=lambda x: x[2], reverse=True)

            print(
                tabulate(
                    table_sorted,
                    headers=["Artist", "Positive", "Negative"],
                    tablefmt="psql",
                )
            )
            print("\n")


def clean_data(dfs, TIME_TO_PREDICT):
    # Remove just the nights after the selected one
    serate = list(dfs.keys())
    serata_index = serate.index(TIME_TO_PREDICT["serata"])
    serate_to_remove = serate[serata_index + 1 :]
    for serata in serate_to_remove:
        dfs.pop(serata)

    target_dt = TIME_TO_PREDICT["datetime"]

    hour_ranges = list(dfs[TIME_TO_PREDICT["serata"]].keys())
    hour_ranges_to_keep = []

    for hr in hour_ranges:
        start_hr, end_hr = map(int, hr.split("-"))
        # Build the datetime for the start and end of the interval
        interval_start = datetime.combine(
            target_dt.date(), datetime.min.time()
        ) + timedelta(hours=start_hr)
        interval_end = datetime.combine(
            target_dt.date(), datetime.min.time()
        ) + timedelta(hours=end_hr)

        # If hr in NEXT_DAY_RANGES shift the interval to the next day
        if hr in NEXT_DAY_RANGES:
            interval_start += timedelta(days=1)
            interval_end += timedelta(days=1)

        # Keep only if the interval ends before the target time
        if interval_end <= target_dt:
            hour_ranges_to_keep.append(hr)

    # Filter the hour ranges to keep
    for hr in hour_ranges:
        if hr not in hour_ranges_to_keep:
            dfs[TIME_TO_PREDICT["serata"]].pop(hr, None)

    return dfs


def get_time_to_predict(serata, time_str):
    data_night_str = SANREMO["NIGHTS"][serata]["data"]
    data_night = datetime.strptime(data_night_str, "%Y-%m-%d")

    # If the time is before 20:00, we consider it the next day
    if time_str < "20:00:00":
        data_night += timedelta(days=1)

    datetime_to_predict = datetime.strptime(
        f"{data_night.strftime('%Y-%m-%d')} {time_str}", "%Y-%m-%d %H:%M:%S"
    )

    return {
        "serata": serata,
        "datetime": datetime_to_predict,
    }


def prepare_lstm_data_with_labels(dfs_aggregated, lookback=5, validation_split=0.2):
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from sklearn.model_selection import train_test_split

    # Flatten time series data into artist: [time series of [positive, negative]]
    artist_data = {artist: [] for artist in SANREMO["ARTISTS"]}

    # Collect both positive and negative counts
    for serata, hours in dfs_aggregated.items():
        for hr in HOUR_RANGES:
            if hr in hours:
                for artist, counts in hours[hr].items():
                    artist_data[artist].append(
                        [counts["positive_count"], counts["negative_count"]]
                    )  # Both counts

    # Build sequences
    X_all, y_all, artists_all = [], [], []
    scalers = {}
    label_encoder = LabelEncoder()
    artist_names = list(artist_data.keys())
    label_encoder.fit(artist_names)

    for artist, series in artist_data.items():
        if len(series) <= lookback:
            continue
        # Convert to numpy array (n_samples, 2)
        series_array = np.array(series)
        scaler = MinMaxScaler()
        series_scaled = scaler.fit_transform(series_array)
        scalers[artist] = scaler

        # Build sequences with both features
        for i in range(lookback, len(series_scaled)):
            X_all.append(series_scaled[i - lookback : i, :])  # (lookback, 2)
            y_all.append(series_scaled[i, :])  # Both features as target
            artists_all.append(label_encoder.transform([artist])[0])

    X_all = np.array(X_all)
    y_all = np.array(y_all)
    # Remove expand_dims as X_all is already (samples, lookback, 2)

    # Split data (ensure no shuffling to preserve time order)
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=validation_split, shuffle=False
    )

    return X_train, X_val, y_train, y_val, artists_all, scalers, label_encoder


def predict_next_for_artist(
    model, artist_name, dfs_aggregated, scalers, label_encoder, lookback=5, verbose=True
):
    # Collect both positive and negative counts
    series = []
    for serata, hours in dfs_aggregated.items():
        for hr in HOUR_RANGES:
            if hr in hours and artist_name in hours[hr]:
                counts = hours[hr][artist_name]
                series.append([counts["positive_count"], counts["negative_count"]])

    if artist_name not in scalers or len(series) < lookback:
        if verbose:
            print(f"Not enough data for {artist_name}")
        return None

    # Scale using the artist's scaler
    scaler = scalers[artist_name]
    series_scaled = scaler.transform(np.array(series))
    last_sequence = series_scaled[-lookback:].reshape(1, lookback, 2)

    # Verbose controls whether to show progress bar
    prediction_scaled = model.predict(last_sequence, verbose=1 if verbose else 0)
    prediction = scaler.inverse_transform(prediction_scaled)
    return {"positive": prediction[0][0], "negative": prediction[0][1]}


def get_predictions(model, dfs_aggregated, scalers, label_encoder, verbose=True):
    predictions = {}
    for artist in SANREMO["ARTISTS"]:
        pred = predict_next_for_artist(
            model,
            artist,
            dfs_aggregated,
            scalers,
            label_encoder,
            verbose=verbose,
        )
        if pred is not None:
            predictions[artist] = pred
    return predictions
