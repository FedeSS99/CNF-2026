import polars as pl

import plotly.graph_objects as go
import colorcet as cc


def plot_dtec_data(PRN_data : dict[int, list[pl.DataFrame]], title : str) -> None:
    unique_PRNs : list[int] = list(PRN_data.keys())

    line_objects : list[go.Scatter] = []
    for prn in unique_PRNs:
        for k, segment in enumerate(PRN_data[prn], start = 1):
            line_plot_object = go.Scatter(
                x = segment["Datetime"].to_list(),
                y = segment["Dtec"].to_list(),
                mode = "lines",
                name = f"Segment {k}",
                legendgroup = f"PRN {prn}", 
                legendgrouptitle_text = f"PRN {prn}",
                line=dict(color = cc.b_glasbey_bw[prn])
            )
            line_objects.append(line_plot_object)

    Figure = go.Figure(data = line_objects)
    Figure.update_layout(
        xaxis_title="UTC",
        yaxis_title="DTEC (TEC units)",
        title = title,
        width = 1080,
        height = 720
        )
    Figure.show()