"""Plotly figure builders. Each function returns a figure (never None) so pages never crash."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config as cfg

LAYOUT = dict(template="plotly_white", margin=dict(l=10, r=10, t=50, b=10),
              font=dict(family="Segoe UI, Arial, sans-serif", size=12), legend_title_text="")


def _empty(msg="No data for the current filters"):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color="#6B7785"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(**LAYOUT, height=300)
    return fig


def _style(fig, title, height=420, **kw):
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=height, **LAYOUT, **kw)
    return fig


def route_bar(routes: pd.DataFrame, title: str, ascending=True, n=10, unit="days"):
    if routes is None or routes.empty:
        return _empty()
    d = routes.head(n).iloc[::-1]
    fig = px.bar(d, x="Avg Lead Time", y="Route", color="Factory", orientation="h",
                 color_discrete_map=cfg.FACTORY_COLORS,
                 custom_data=["Shipments", "Delay Rate", "Efficiency Score", "Median Lead Time"])
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Avg lead time: %{x:.1f}<br>Median: %{customdata[3]:.1f}"
                                    "<br>Shipments: %{customdata[0]:,}<br>Delay rate: %{customdata[1]:.1%}"
                                    "<br>Efficiency score: %{customdata[2]:.0f}<extra></extra>")
    return _style(fig, title, height=max(320, 34 * len(d) + 120), xaxis_title=f"Average lead time ({unit})", yaxis_title="")


def volume_vs_leadtime(routes: pd.DataFrame, unit="days"):
    if routes is None or routes.empty:
        return _empty()
    fig = px.scatter(routes, x="Shipments", y="Avg Lead Time", color="Factory", size="Total Sales",
                     color_discrete_map=cfg.FACTORY_COLORS, hover_name="Route", size_max=28,
                     custom_data=["Delay Rate", "Efficiency Score"])
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Shipments: %{x:,}<br>Avg lead time: %{y:.1f}"
                                    "<br>Delay rate: %{customdata[0]:.1%}<extra></extra>")
    fig.add_vline(x=routes["Shipments"].median(), line_dash="dot", line_color="#9AA5B1",
                  annotation_text="median volume", annotation_position="top")
    fig.add_hline(y=routes["Avg Lead Time"].median(), line_dash="dot", line_color="#9AA5B1",
                  annotation_text="median lead time", annotation_position="right")
    return _style(fig, "Route volume vs. average lead time (bubble = total sales)",
                  xaxis_title="Shipments per route", yaxis_title=f"Average lead time ({unit})")


def efficiency_hist(routes: pd.DataFrame):
    el = routes[routes["Eligible"]] if routes is not None and not routes.empty else None
    if el is None or el.empty:
        return _empty("No routes meet the minimum shipment count")
    fig = px.histogram(el, x="Efficiency Score", color="Performance", nbins=20,
                       color_discrete_map=cfg.PERF_COLORS)
    return _style(fig, "Distribution of route efficiency scores", height=320,
                  xaxis_title="Efficiency score (100 = fastest ranked route)", yaxis_title="Routes")


def group_bar(df: pd.DataFrame, x: str, title: str, color=None, cmap=None, unit="days", sort=True, orient="v"):
    if df is None or df.empty:
        return _empty()
    d = df.sort_values("Avg Lead Time") if sort else df
    fig = px.bar(d, x=x if orient == "v" else "Avg Lead Time", y="Avg Lead Time" if orient == "v" else x,
                 color=color, color_discrete_map=cmap, orientation=orient,
                 custom_data=["Shipments", "Median Lead Time", "Delay Rate"])
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Avg lead time: %{y:.1f}<br>Median: %{customdata[1]:.1f}"
                                    "<br>Shipments: %{customdata[0]:,}<br>Delay rate: %{customdata[2]:.1%}<extra></extra>"
                      if orient == "v" else
                      "<b>%{y}</b><br>Avg lead time: %{x:.1f}<br>Median: %{customdata[1]:.1f}"
                      "<br>Shipments: %{customdata[0]:,}<br>Delay rate: %{customdata[2]:.1%}<extra></extra>")
    if orient == "v":
        return _style(fig, title, height=380, xaxis_title="", yaxis_title=f"Average lead time ({unit})")
    return _style(fig, title, height=max(360, 22 * len(d) + 100), xaxis_title=f"Average lead time ({unit})", yaxis_title="")


def delay_rate_bar(df: pd.DataFrame, x: str, title: str, color=None, cmap=None):
    if df is None or df.empty:
        return _empty()
    d = df.sort_values("Delay Rate", ascending=False)
    fig = px.bar(d, x=x, y="Delay Rate", color=color, color_discrete_map=cmap, custom_data=["Shipments"])
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Delay rate: %{y:.1%}<br>Shipments: %{customdata[0]:,}<extra></extra>")
    fig.update_yaxes(tickformat=".0%")
    return _style(fig, title, height=380, xaxis_title="", yaxis_title="Share of shipments above threshold")


def volume_bar(df: pd.DataFrame, x: str, title: str, color=None, cmap=None):
    if df is None or df.empty:
        return _empty()
    d = df.sort_values("Shipments", ascending=False)
    fig = px.bar(d, x=x, y="Shipments", color=color, color_discrete_map=cmap)
    return _style(fig, title, height=380, xaxis_title="", yaxis_title="Shipments")


def leadtime_box(ship: pd.DataFrame, by: str, title: str, cmap=None, unit="days"):
    if ship is None or ship.empty:
        return _empty()
    fig = px.box(ship, x=by, y="Lead Time", color=by, color_discrete_map=cmap, points=False)
    fig.update_layout(showlegend=False)
    return _style(fig, title, height=380, xaxis_title="", yaxis_title=f"Lead time ({unit})")


def tradeoff_scatter(mode_df: pd.DataFrame, y: str, title: str, unit="days"):
    if mode_df is None or mode_df.empty:
        return _empty()
    fig = px.scatter(mode_df, x="Avg Lead Time", y=y, size="Shipments", color="Ship Mode", text="Ship Mode",
                     color_discrete_map=cfg.SHIPMODE_COLORS, size_max=45)
    fig.update_traces(textposition="top center")
    return _style(fig, title, height=380, xaxis_title=f"Average lead time ({unit})", yaxis_title=y)


def shipmode_grouped(mode_df: pd.DataFrame, unit="days"):
    if mode_df is None or mode_df.empty:
        return _empty()
    m = mode_df.melt(id_vars="Ship Mode", value_vars=["Avg Lead Time", "Median Lead Time"],
                     var_name="Metric", value_name="Days")
    fig = px.bar(m, x="Ship Mode", y="Days", color="Metric", barmode="group",
                 color_discrete_sequence=["#1F4E79", "#7FA6CC"])
    return _style(fig, "Average vs. median lead time by ship mode", height=380, yaxis_title=f"Lead time ({unit})", xaxis_title="")


def leadtime_hist(series: pd.Series, title: str, nbins=60, unit="days", cut=None):
    s = series.dropna()
    if s.empty:
        return _empty()
    fig = px.histogram(s.to_frame("Lead Time"), x="Lead Time", nbins=nbins, color_discrete_sequence=["#1F4E79"])
    if cut is not None:
        fig.add_vline(x=cut, line_dash="dash", line_color="#C0392B", annotation_text=f"threshold {cut:,.0f}")
    return _style(fig, title, height=340, xaxis_title=f"Lead time ({unit})", yaxis_title="Rows / shipments")


def monthly_trend_chart(tr: pd.DataFrame, unit="days"):
    if tr is None or tr.empty:
        return _empty()
    fig = go.Figure()
    fig.add_bar(x=tr["Order Month"], y=tr["Shipments"], name="Shipments", marker_color="#C9D6E3", yaxis="y2")
    fig.add_scatter(x=tr["Order Month"], y=tr["Avg Lead Time"], name="Avg lead time", mode="lines+markers",
                    line=dict(color="#1F4E79"))
    fig.update_layout(yaxis=dict(title=f"Avg lead time ({unit})"),
                      yaxis2=dict(title="Shipments", overlaying="y", side="right", showgrid=False))
    return _style(fig, "Monthly shipments and average lead time (by order month)", height=360)


def timeline_chart(ship: pd.DataFrame, max_rows=60):
    """Order-level shipment timeline: bar from Order Date to Ship Date per shipment."""
    if ship is None or ship.empty:
        return _empty()
    d = ship.sort_values("Order Date").head(max_rows).copy()
    d["Label"] = d["Order ID"].str.slice(0, 22)
    fig = px.timeline(d, x_start="Order Date", x_end="Ship Date", y="Label", color="Factory",
                      color_discrete_map=cfg.FACTORY_COLORS,
                      hover_data={"Lead Time": ":.0f", "Ship Mode": True, "Customer State": True, "Label": False})
    fig.update_yaxes(autorange="reversed", title="")
    return _style(fig, f"Shipment timeline: Order Date → Ship Date (first {len(d)} shipments by order date)",
                  height=max(360, 16 * len(d) + 120))


def leadtime_over_time(ship: pd.DataFrame, unit="days"):
    if ship is None or ship.empty:
        return _empty()
    fig = px.scatter(ship, x="Order Date", y="Lead Time", color="Ship Mode", color_discrete_map=cfg.SHIPMODE_COLORS,
                     opacity=0.6, hover_data=["Order ID", "Customer State", "Factory"])
    return _style(fig, "Lead time of each shipment by order date", height=340, yaxis_title=f"Lead time ({unit})")


def route_map(state_stats: pd.DataFrame, lines: pd.DataFrame, color_metric: str, unit="days", show_lines=True):
    """Interactive North-America map.

    state_stats : output of analysis.state_map_stats (one row per customer state)
    lines       : route table rows to draw as Factory -> State lines (may be empty)
    """
    if state_stats is None or state_stats.empty:
        return _empty()
    s = state_stats.dropna(subset=["Latitude", "Longitude"]).copy()
    metric_cfg = {
        "Average lead time": ("Avg Lead Time", "RdYlGn_r", f"Avg lead time ({unit})"),
        "Efficiency score": ("Efficiency Score", "RdYlGn", "Efficiency score"),
        "Delay rate": ("Delay Rate", "RdYlGn_r", "Delay rate"),
    }
    col, scale, cbar_title = metric_cfg[color_metric]
    fig = go.Figure()

    if show_lines and lines is not None and not lines.empty:
        coords = s.set_index("Customer State")[["Latitude", "Longitude"]]
        vmax = max(lines["Shipments"].max(), 1)
        for _, r in lines.iterrows():
            if r["Destination"] not in coords.index or r["Factory"] not in cfg.FACTORY_COORDS:
                continue
            flat, flon = cfg.FACTORY_COORDS[r["Factory"]]
            slat, slon = coords.loc[r["Destination"]]
            fig.add_trace(go.Scattergeo(
                lon=[flon, slon], lat=[flat, slat], mode="lines",
                line=dict(width=0.6 + 3.0 * r["Shipments"] / vmax, color=cfg.FACTORY_COLORS[r["Factory"]]),
                opacity=0.45, showlegend=False, hoverinfo="text",
                text=(f"<b>{r['Route']}</b><br>Shipments: {r['Shipments']:,}"
                      f"<br>Avg lead time: {r['Avg Lead Time']:.1f}<br>Delay rate: {r['Delay Rate']:.1%}")))

    vmax_n = max(s["Shipments"].max(), 1)
    sizes = 7 + 26 * np.sqrt(s["Shipments"] / vmax_n)
    hover = [
        (f"<b>{r['Customer State']}</b> ({r['Region']}, {r['Country']})"
         f"<br>Main factory: {r['Main_Factory']} ({int(r['Factories'])} factories serve)"
         f"<br>Shipments: {r['Shipments']:,}<br>Avg lead time: {r['Avg Lead Time']:.1f} | Median: {r['Median Lead Time']:.1f}"
         f"<br>Delay rate: {r['Delay Rate']:.1%}"
         f"<br>Efficiency score: {'n/a (low volume)' if pd.isna(r['Efficiency Score']) else format(r['Efficiency Score'], '.0f')}")
        for _, r in s.iterrows()]
    fig.add_trace(go.Scattergeo(
        lon=s["Longitude"], lat=s["Latitude"], mode="markers", name="Customer states", text=hover, hoverinfo="text",
        marker=dict(size=sizes, color=s[col], colorscale=scale, showscale=True, opacity=0.9,
                    line=dict(width=0.8, color="white"),
                    colorbar=dict(title=cbar_title, thickness=12, len=0.7, x=1.0))))

    for f, (lat, lon) in cfg.FACTORY_COORDS.items():
        fig.add_trace(go.Scattergeo(
            lon=[lon], lat=[lat], mode="markers", name=f, hoverinfo="text", text=f"<b>Factory: {f}</b>",
            marker=dict(size=17, symbol="diamond", color=cfg.FACTORY_COLORS[f], line=dict(width=2, color="black"))))

    fig.update_geos(scope="north america", projection_type="mercator", lonaxis_range=[-135, -52],
                    lataxis_range=[24, 60], showland=True, landcolor="#F3F6FA", showcountries=True,
                    countrycolor="#B8C2CC", showsubunits=True, subunitcolor="#D5DCE3", showlakes=True,
                    lakecolor="#E6F0FA", bgcolor="rgba(0,0,0,0)")
    fig.update_layout(**{**LAYOUT, "margin": dict(l=0, r=0, t=50, b=0)}, height=560,
                      title=dict(text=f"Factories → customer states (bubble size = shipments, colour = {color_metric.lower()})",
                                 font=dict(size=15)),
                      legend=dict(orientation="h", y=-0.02))
    return fig


def us_heatmap(state_stats: pd.DataFrame, color_metric: str, unit="days"):
    """US choropleth (state-level heatmap). Canadian provinces are not drawn here."""
    if state_stats is None or state_stats.empty:
        return _empty()
    s = state_stats.copy()
    s["Abbr"] = s["Customer State"].map(cfg.STATE_ABBR)
    s = s.dropna(subset=["Abbr"])
    if s.empty:
        return _empty("No US states in the current selection")
    col, scale, label = {
        "Average lead time": ("Avg Lead Time", "RdYlGn_r", f"Avg lead time ({unit})"),
        "Efficiency score": ("Efficiency Score", "RdYlGn", "Efficiency score"),
        "Delay rate": ("Delay Rate", "RdYlGn_r", "Delay rate"),
    }[color_metric]
    fig = px.choropleth(s, locations="Abbr", locationmode="USA-states", scope="usa", color=col,
                        color_continuous_scale=scale, hover_name="Customer State",
                        hover_data={"Abbr": False, "Shipments": ":,", "Avg Lead Time": ":.1f", "Median Lead Time": ":.1f",
                                    "Delay Rate": ":.1%", "Main_Factory": True, col: False},
                        labels={"Main_Factory": "Main factory"})
    fig.update_coloraxes(colorbar_title=label, colorbar_thickness=12)
    fig.update_geos(bgcolor="rgba(0,0,0,0)", lakecolor="#E6F0FA")
    if color_metric == "Efficiency score":
        fig.add_annotation(text="Grey = fewer shipments than the ranking minimum", x=0, y=0, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=11, color="#6B7785"))
    return _style(fig, f"US heatmap: {color_metric.lower()} by customer state", height=520)
