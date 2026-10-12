import polars as pl

from os import walk
from pathlib import Path

class PyTECGgReader:
    def __init__(self, file_path : str) -> None:
        self.__file_path = file_path

        self.__csv_schema = {
            "epoch" : pl.Datetime("ms"),
            "sv" : pl.Categorical,
            "is_cycle_slip" : pl.Boolean,
            "ele" : pl.Float64,
            "stec": pl.Float64,
            "vtec": pl.Float64
        }

    def __get_all_path_files(self) -> list[str]:
        all_files = []
        for (dir_path, dir_names, file_names) in walk(self.__file_path):
            for file_name in file_names:
                if file_name.endswith(".csv"):
                    all_files.append(
                        f"{dir_path}/{file_name}"
                    )
        return all_files

    def __read_csv_file(self, csv_file_path : str) -> tuple[str, pl.DataFrame]:
        csv_path = Path(csv_file_path)
        station_name = csv_path.stem

        df = pl.read_csv(
            csv_file_path,
            schema_overrides = self.__csv_schema
        )

        df = df.with_columns(
            pl.lit(station_name).alias("station")
        ).select(
            ["station"] + list(self.__csv_schema.keys())
        )

        df = df.unique(subset = ["station", "sv", "epoch"], keep = "first")
        df = df.filter(pl.col("is_cycle_slip") == False)

        return station_name, df

    @staticmethod
    def __join_station_dataframes(dataframes : list[pl.DataFrame]) -> pl.DataFrame:
        return pl.concat(dataframes).sort(
            ["station", "sv", "epoch"],
            descending = [False, False, False]
        )

    def read_all(self) -> dict[str, pl.DataFrame]:
        all_csv_file_paths = self.__get_all_path_files()

        data_stations = dict()

        for file in all_csv_file_paths:
            station_name, file_data = self.__read_csv_file(file)
        
            if station_name in data_stations.keys():
                data_stations[station_name].append(file_data)
            else:
                data_stations[station_name] = [file_data]
        
        for station_name in data_stations.keys():
            data_stations[station_name] = self.__join_station_dataframes(data_stations[station_name])
        
        return data_stations