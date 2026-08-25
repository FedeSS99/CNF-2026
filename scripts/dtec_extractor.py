from .reading import CMNReader
from .preprocessing import TECProcesser

from polars import DataFrame
from numpy import ndarray

def get_dtec_data_from_stations(cmn_global_path : str, min_elev : float = 30, window_size : int =  240, poly_order : int = 5) -> dict[str, dict[int, list[ndarray]]]:
    # extract all data from different stations (still not grouped by PRN satellite number)
    stations_data = CMNReader(cmn_global_path).read_all()

    # Preprocess stations data
    # 1 -. Group by PRN satellite number
    # 2 -. Filter by minimum elevation and series size
    # 3 -. Apply Savitzky-Golay filter and compute DTEC
    tec_processer = TECProcesser(
        min_elev = min_elev,
        window_size = window_size,
        poly_order = poly_order
    )
    tec_data = {
        station_name : tec_processer.process_data(station_data)
        for station_name, station_data in stations_data.items()
    }

    return tec_data