import polars as pl
import numpy as np

from scipy.signal import savgol_filter


class TECProcesser:
    def __init__(self, poly_order : int, window_size : int, min_elev : float = 30.0) -> None:
        self.__poly_order = poly_order
        self.__window_size = window_size
        self.__min_elev = min_elev


    @staticmethod
    def __get_cut_indexes_by_sampling_time(df : pl.DataFrame, sv : str) -> list[int]:
        df_sv = df.select(["sv", "epoch"]).filter(pl.col("sv") == sv)
        datetime_values = df_sv["epoch"].to_numpy()

        diff_times = np.diff(datetime_values).astype("timedelta64[s]")
        sample_time_median = np.median(diff_times)

        cut_indexes = np.argwhere(diff_times > 2 * sample_time_median) + 1
        cut_indexes = cut_indexes.flatten().tolist()

        return cut_indexes

    @staticmethod
    def __apply_savgol_filter(values : np.ndarray, poly_order : int, window_size : int) -> np.ndarray:
        return savgol_filter(values, window_length = window_size, polyorder = poly_order, mode = "nearest")

    def __split_dataframe_by_indexes(self, df: pl.DataFrame, indexes: list[int], window_size: int) -> list[pl.DataFrame]:
        boundaries = [0] + indexes + [df.height]
        slices = [(boundaries[i], boundaries[i + 1] - boundaries[i]) for i in range(len(boundaries) - 1)]
        chunks = [df.slice(pivot_index, length) for pivot_index, length in slices]
        chunks = [chunk for chunk in chunks if chunk.height >= window_size]
    
        PRN_data_time_cut = []
        for segment in chunks:
            vtec_values_segment = segment.select("vtec").to_numpy().flatten()
            vtec_sg_values = self.__apply_savgol_filter(vtec_values_segment, self.__poly_order, self.__window_size)
    
            dtec_values_segment = vtec_values_segment - vtec_sg_values.flatten()
    
            PRN_data_time_cut.append(segment.with_columns(pl.Series("dtec", dtec_values_segment)))
    
        return PRN_data_time_cut

    def __preprocess_data(self, df : pl.DataFrame, min_elev : float) -> list[pl.DataFrame]:
        df_filtered = df.filter(pl.col("ele") >= min_elev)
        unique_svs = df_filtered["sv"].unique().to_list()

        sv_cut_data = list()
        for sv in unique_svs:
            time_index_cuts = self.__get_cut_indexes_by_sampling_time(df_filtered, sv)

            PRN_data_time_cut = self.__split_dataframe_by_indexes(
                df_filtered.filter(pl.col("sv") == sv),
                time_index_cuts,
                self.__window_size
                )

            sv_cut_data += PRN_data_time_cut
            
        return sv_cut_data
    
    
    def process_data(self, df : pl.DataFrame) -> list[pl.DataFrame]:
        return self.__preprocess_data(df, self.__min_elev)