import os
import polars as pl
import numpy as np

from dtaidistance.dtw import distance_matrix_fast
from sklearn.manifold import MDS


class DTECAnalysis:
    def __init__(self, data_path : str) -> None:
        self.__data_path = data_path


    def __read_station_PRN_segments(self, data_file_path : str) -> pl.DataFrame:
        station_name = data_file_path.split("/")[-1].split("_")[0]
        PRN = int(data_file_path.split("/")[-1].split("_")[1][3:5])

        file_data = pl.read_csv(data_file_path, schema_overrides = {"Datetime": pl.Datetime("ms")})
        grouped_data = file_data.group_by(
            "Segment", maintain_order = True
        ).agg(
            pl.col("Datetime").sort_by(pl.col("Datetime")).alias("Datetime"),
            pl.col("Dtec").sort_by(pl.col("Datetime")).alias("Dtec"),
        )

        grouped_data = grouped_data.with_columns(
            pl.Series("Station", grouped_data.height * [station_name]),
            pl.Series("PRN", grouped_data.height * [PRN])
        )
        grouped_data = grouped_data.select(
            ["Station", "PRN", "Segment", "Datetime", "Dtec"]
        )

        return grouped_data

    def __get_all_data_file_paths(self) -> list[str]:
        all_data_file_paths = []

        for root, dirs, files in os.walk(self.__data_path):
            for file in files:
                if file.endswith(".csv"):
                    all_data_file_paths.append(os.path.join(root, file))

        return all_data_file_paths

    def __compute_normalized_dtec(self, all_data : pl.DataFrame) -> pl.DataFrame:
        all_data = all_data.with_columns(
            Normalized_Dtec = pl.col("Dtec").list.eval(
                (pl.element() - pl.element().mean()) / pl.element().std()
            )
        )

        return all_data

    def __get_all_data(self) -> pl.DataFrame:
        all_data_file_paths = self.__get_all_data_file_paths()
        all_data = []

        for data_file_path in all_data_file_paths:
            station_PRN_segments = self.__read_station_PRN_segments(data_file_path)
            all_data.append(station_PRN_segments)

        all_data = pl.concat(all_data)
        all_data = all_data.sort(
            ["Station", "PRN", "Segment", "Datetime"],
            descending = [False, False, False, False]
        )
        all_data = self.__compute_normalized_dtec(all_data)

        return all_data

    def __compute_dtw_matrix(self, all_data_df : pl.DataFrame) -> np.ndarray:
        dtw_matrix = distance_matrix_fast(
            [np.array(norm_dtec_list) for norm_dtec_list in all_data_df["Normalized_Dtec"]]
        )

        return dtw_matrix

    def commpute_segments_and_dist_matrix(self) -> tuple[pl.DataFrame, np.ndarray]:
        all_data_df = self.__get_all_data()
        dtw_matrix = self.__compute_dtw_matrix(all_data_df)

        return all_data_df, dtw_matrix
