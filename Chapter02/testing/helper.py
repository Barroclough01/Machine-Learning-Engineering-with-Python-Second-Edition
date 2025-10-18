import os
import pandas as pd
import numpy as np
import datetime
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence

rs = RandomState(MT19937(SeedSequence(123456789)))


# Define simulate ride data function
def simulate_ride_data():
    """
    Simulates ride data.

    Simulates 370 ride distances with normal distribution around 10,
    10 ride distances with normal distribution around 30 (long distances),
    10 ride distances with normal distribution around 10 (same distance),
    and 10 ride distances with normal distribution around 10 (same distance).
    
    Simulates 370 ride speeds with normal distribution around 30,
    10 ride speeds with normal distribution around 30 (same speed),
    10 ride speeds with normal distribution around 50 (high speed),
    and 10 ride speeds with normal distribution around 15 (low speed).
    
    Assembles them into a Data Frame with ride_id as the index.

    Returns:
        df_sim (pandas.DataFrame): A DataFrame containing simulated ride data.
    """
    # Simulate some ride data ...
    ride_dists = np.concatenate(
        (
            10 * np.random.random(size=370),
            30 * np.random.random(size=10),  # long distances
            10 * np.random.random(size=10),  # same distance
            10 * np.random.random(size=10)  # same distance
        )
    )
    ride_speeds = np.concatenate(
        (
            np.random.normal(loc=30, scale=5, size=370),
            np.random.normal(loc=30, scale=5, size=10),  # same speed
            np.random.normal(loc=50, scale=10, size=10),  # high speed
            np.random.normal(loc=15, scale=4, size=10)  # low speed
        )
    )
    ride_times = ride_dists / ride_speeds

    # Assemble into Data Frame
    df_sim = pd.DataFrame(
        {
            'ride_dist': ride_dists,
            'ride_time': ride_times,
            'ride_speed': ride_speeds
        }
    )
    ride_ids = datetime.datetime.now().strftime("%Y%m%d") + df_sim.index.astype(str)
    df_sim['ride_id'] = ride_ids

    return df_sim


def get_taxi_data():
    
    """
    Reads in taxi ride data from a csv file or simulates it if not present.

    Args:
        None

    Returns:
        df (pandas.DataFrame): A DataFrame containing taxi ride data.
    """
    # If data present, read it in
    file_path = '../../Chapter01/clustering/taxi-rides.csv' #relative
    #file_path = f'''chapter1/batch-anomaly/data/taxi-rides.csv''' #from top dir
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        df = simulate_ride_data()
        df.to_csv(file_path, index=False)
    assert isinstance(df, object)
    return df
