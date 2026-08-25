import pandas as pd
import polars as pl
from re import sub

from os import walk
from pathlib import Path
from io import StringIO

class CMNReader:
    def __init__(self, file_path: str) -> None:
        self.__file_path = file_path

    def __get_all_path_files(self) -> list[str]:
        all_files = []
        for (dir_path, dir_names, file_names) in walk(self.__file_path):
            for file_name in file_names:
                if file_name.endswith(".Cmn"):
                    all_files.append(f"{dir_path}/{file_name}")

        return all_files

    @staticmethod
    def __get_station_name(station_name : str) -> str:
        clean_station_name = sub(r"\d+", "", station_name)
        return clean_station_name
    
    @staticmethod
    def __read_cmn_file(cmn_file_path : str) -> tuple[str, pl.DataFrame]:
        cmn_path = Path(cmn_file_path)
        station_name = sub(r"\d+", "", cmn_path.name.split("-")[0])
        date_text = "-".join(cmn_path.name.split(".")[0].split("-")[1:])
        
        cmn_text = cmn_path.read_text(encoding = "utf-8").splitlines()
        table_text = "\n".join(cmn_text[5:])

        df = pd.read_table(
            StringIO(table_text),
            sep = r"\s+",
            header = None,
        )
        df.columns = [
            "MJdatet", "Time", "PRN", "Az", "Ele", "Lat", "Lon", "Stec", "Vtec", "S4"
        ]
        df = pl.from_pandas(df, rechunk = True)
        df = df.with_columns(
            pl.col("Time").replace(-24.0, 0.0)
        ).with_columns(
            pl.lit(date_text).alias("Date")
        ).with_columns(
            pl.col("Date").str.strptime(pl.Date, format = "%Y-%m-%d").alias("Date"),
            (pl.col("Time") * 3600 * 1000).cast(pl.Duration("ms")).alias("Time")
        ).with_columns(
            Datetime = pl.col("Date").dt.combine(pl.col("Time"))
        ).with_columns(
            pl.col("Datetime").dt.round("1s").cast(pl.Datetime(time_unit = "ms"))
        )

        df = df.select(["PRN", "Datetime", "Az", "Ele", "Lat", "Lon", "Stec", "Vtec"])

        return station_name, df

    @staticmethod
    def __join_station_dataframes(dataframes : list[pl.DataFrame]) -> pl.DataFrame:
        return pl.concat(dataframes, rechunk = True).sort(by = ["PRN", "Datetime"], descending = [False, False])

    def read_all(self) -> dict[str, pl.DataFrame]:
        all_files = self.__get_all_path_files()

        data_stations = dict()

        for file in all_files:
            station_name, file_data = self.__read_cmn_file(file)

            if station_name in data_stations.keys():
                data_stations[station_name].append(file_data)
            else:
                data_stations[station_name] = [file_data]

        for station_name in data_stations.keys():
            data_stations[station_name] = self.__join_station_dataframes(data_stations[station_name])

        return data_stations