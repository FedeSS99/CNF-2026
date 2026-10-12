import polars as pl
import numpy as np

from dtaidistance.dtw import distance_matrix_fast
from sklearn.manifold import MDS

from tqdm.notebook import tqdm


class DTECAnalysis:
    def __init__(self, k_max : int) -> None:
        self.__k_max = k_max

    def __extract_normalized_dtec(self, list_of_dfs : list[pl.DataFrame]) -> list[pl.DataFrame]:
        list_of_norm_dtec_dfs = []

        for df in list_of_dfs:
            dtec_values = df["dtec"].cast(pl.Float64).to_numpy()
            finite_mask = np.isfinite(dtec_values)
            if not finite_mask.any():
                continue

            df = df.filter(pl.Series("finite_dtec", finite_mask))
            dtec_values = dtec_values[finite_mask]
            median = np.median(dtec_values)
            scale = np.quantile(dtec_values, 0.75) - np.quantile(dtec_values, 0.25)
            if scale == 0:
                scale = np.std(dtec_values)

            if np.isfinite(scale) and scale > 0:
                norm_dtec = (dtec_values - median) / scale
            else:
                norm_dtec = np.zeros_like(dtec_values)

            df = df.with_columns(pl.Series("norm_dtec", norm_dtec))

            df = df.select(["station", "sv", "epoch", "norm_dtec"])
            list_of_norm_dtec_dfs.append(df)

        return list_of_norm_dtec_dfs


    def __extract_dtec_time_series(self, list_of_dfs : list[pl.DataFrame]) -> list[tuple[object, list, np.ndarray]]:
        dtec_time_series = []

        for df in list_of_dfs:
            dtec_time_series.append(
                (df["sv"][0], df["epoch"].to_list(), df["norm_dtec"].to_numpy())
            )

        return dtec_time_series


    def __compute_distance_matrix(self, dtec_values : list[np.ndarray]) -> np.ndarray:
        dtec_values = [np.ascontiguousarray(values, dtype=np.float64) for values in dtec_values]
        distance_matrix = distance_matrix_fast(dtec_values)
        return distance_matrix


    def __compute_mds(self, num_components: int, distance_matrix : np.ndarray) -> tuple[np.ndarray, float]:
        mds = MDS(n_components = num_components,
                  metric_mds = True,
                  init = "classical_mds",
                  max_iter = 1000,
                  eps = 1e-6,
                  metric = "precomputed",
                  normalized_stress = True,
                  random_state = 42)
        mds_coordinates = mds.fit_transform(distance_matrix)

        return mds_coordinates, mds.stress_


    def compute_mds_embeddings(self, dtec_data : dict[str, list[pl.DataFrame]]) -> tuple[pl.DataFrame, dict[int, tuple[np.ndarray, float]]]:
        dtec_time_series = dict(
            station = [],
            sv = [],
            epoch = [],
            norm_dtec = []
        )
        mds_results = dict()

        for station, station_data in tqdm(dtec_data.items(), desc = "Processing stations", unit = "station"):
            norm_dtec_df = self.__extract_normalized_dtec(station_data)
            dtec_time_series_list = self.__extract_dtec_time_series(norm_dtec_df)

            for sv, epochs, norm_dtec in dtec_time_series_list:
                dtec_time_series["station"].append(station)
                dtec_time_series["sv"].append(sv)
                dtec_time_series["epoch"].append(epochs)
                dtec_time_series["norm_dtec"].append(norm_dtec)


        dtec_time_series = pl.DataFrame(dtec_time_series)
        dtec_time_series = dtec_time_series.with_columns(
            pl.col("epoch").list.min().alias("start_epoch"),
            pl.col("epoch").list.max().alias("end_epoch")
        )

        print(f"Computing distance matrix for {len(dtec_time_series['norm_dtec'])} time series...")
        distance_matrix = self.__compute_distance_matrix(dtec_time_series["norm_dtec"].to_list())    
        
        for k in tqdm(range(1, self.__k_max + 1), desc = "Computing MDS embeddings", unit = "k"):
            mds_components, stress = self.__compute_mds(
                num_components = k,
                distance_matrix = distance_matrix
            )

            mds_results[k] = (mds_components, stress)

        return dtec_time_series, mds_results