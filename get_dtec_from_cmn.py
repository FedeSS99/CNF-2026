from scripts.dtec_extractor import get_dtec_data_from_stations
import polars as pl

# CMN global data path and parameters
cmn_path = "./data/gopi-output"

min_elev = 30.0
window_size = 240
poly_order = 5

# Get all CMN data and DTEC computation
all_tec_data = get_dtec_data_from_stations(
    cmn_global_path = cmn_path,
    min_elev = min_elev,
    window_size = window_size,
    poly_order = poly_order
)

# Save all data in files
for station_name, station_dict_data in all_tec_data.items():
    for PRN, PRN_data in station_dict_data.items():

        if PRN_data:
            station_PRN_segments = []
            for k, segment in enumerate(PRN_data, start = 1):
                mod_segment = segment.with_columns(
                    pl.Series("Segment", segment.height * [k])
                )
                mod_segment = mod_segment.select(
                    ["Segment"] + segment.columns
                )

                station_PRN_segments.append(mod_segment)
            joined_station_PRN_segments : pl.DataFrame = pl.concat(station_PRN_segments)
            joined_station_PRN_segments = joined_station_PRN_segments.sort(
                ["Segment", "Datetime"], descending = [False, False]
            )

            joined_station_PRN_segments.write_csv(f"./data/output/{station_name}_PRN{PRN:02d}_segments.csv")