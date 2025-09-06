"""Get yield curves from a variety of sources."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.interpolate as interpolate
import sqlalchemy as db

# Engine
engine = db.create_engine(
    "mssql+pyodbc:///?odbc_connect=Driver={SQL Server};Server=192.168.11.102;Database=IDPM;Trusted_Connection=True;"
)


def tenor_in_days(tenor: list[str] | np.ndarray) -> list | np.ndarray:
    """Convert tenor to days.\n"""
    out = []
    for string in tenor:
        match string:
            case "1M":
                out.append(1 / 365)
            case "3M":
                out.append(3 / 12)
            case "6M":
                out.append(6 / 12)
            case "9M":
                out.append(9 / 12)
            case "1Y":
                out.append(1)
            case "2Y":
                out.append(2)
            case "3Y":
                out.append(3)
            case "4Y":
                out.append(4)
            case "5Y":
                out.append(5)
            case "6Y":
                out.append(6)
            case "7Y":
                out.append(7)
            case "8Y":
                out.append(8)
            case "9Y":
                out.append(9)
            case "10Y":
                out.append(10)
            case "11Y":
                out.append(11)
            case "12Y":
                out.append(12)
            case "13Y":
                out.append(13)
            case "14Y":
                out.append(14)
            case "15Y":
                out.append(15)
            case "20Y":
                out.append(20)
            case "25Y":
                out.append(25)
            case "30Y":
                out.append(30)
            case "40Y":
                out.append(40)
            case "50Y":
                out.append(50)
    return out


def clean_data(x: np.ndarray) -> np.ndarray:
    """Clean data by removing any missing values.\n
    Args:
        x: Data to clean.\n
    Returns:
        x: Cleaned data.
    """
    result = []
    for i in range(len(x)):
        result.append(x[i][~np.isnan(x[i])])
    return np.array(result)


def get_interpolation_coef(x: np.ndarray, y: np.ndarray | list) -> list:
    """Get interpolation coefficients.\n"""
    result = []
    for i in range(len(x)):
        result.append(interpolate.CubicSpline(y, x[i]))
    return result


def get_interpolation_result(x: list, y: np.ndarray) -> list:
    """Get interpolation result.\n"""
    result = []
    for i in range(len(x)):
        result.append(x[i](y))
    return result


class YieldCurve:
    """
    YieldCurve class to manipulate yield curves.\n
    Attributes:
        yieldcurve_id: ID of the yield curve to retrieve.
        start_date: Start date of the yield curve.
        end_date: End date of the yield curve.
        interpolation_method: Interpolation method to use. \n
        Methods:
            _get_yield_curve: Get yield curve from the database.\n
            _interpolate_yield: Interpolate yield curve. \n
    """

    def __init__(
        self,
        yieldcurve_id: int,
        start_date: str,
        end_date: str,
        interpolation_method=None,
    ):
        """Initialise the YieldCurve class. \n"""
        self.yieldcurve_id = yieldcurve_id
        self.start_date = start_date
        self.end_date = end_date
        self.interpolation_method = interpolation_method

        for i in ["get_yield", "interpolate_yield"]:
            self.__dict__[i] = None

        self.get_yield = self._get_yield_curve()
        self.interpolate_yield = self._interpolate_yield()
        self.plot_surface = self._plot_surface()

    def _get_yield_curve(self):
        """Get yield curve from the database. \n
        Args:
            yieldcurve_id: ID of the yield curve to retrieve.
            start_date: Start date of the yield curve.
            end_date: End date of the yield curve.\n
        Returns:
            [0] -> data_piv: Pivot table of the yield curve.
            [1] -> data_piv_with_missing: Pivot table of the yield curve with missing values.
        """
        query: str = f"""DECLARE
        @StartDate datetime = '{self.start_date}',
        @EndDate datetime = '{self.end_date}',
        @YieldCurveID varchar(5) = '{self.yieldcurve_id}'
        SELECT
            ycd.MarketDate
            ,ycd.Yield as [Yield]
            ,yt.Code
        FROM YieldCurve yc
        JOIN YieldCurvePoint ycp
            ON ycp.YieldCurveID = yc.YieldCurveID
        JOIN YieldCurveData ycd
            ON ycd.YieldCurvePointID = ycp.YieldCurvePointID
        LEFT JOIN YieldCurveDefnTenor yt
            ON ycp.YieldCurveTenorID = yt.YieldCurveTenorID
        WHERE yc.YieldCurveID = @YieldCurveID and ycd.MarketDate BETWEEN @StartDate AND @EndDate
        ORDER BY yt.Years*365, yt.Months*30, yt.[Days]
        """
        data = pd.read_sql(db.text(query), engine.connect(), index_col="MarketDate")
        column_order = [
            "1M",
            "3M",
            "6M",
            "9M",
            "1Y",
            "2Y",
            "3Y",
            "4Y",
            "5Y",
            "6Y",
            "7Y",
            "8Y",
            "9Y",
            "10Y",
            "11Y",
            "12Y",
            "13Y",
            "14Y",
            "15Y",
            "20Y",
            "25Y",
            "30Y",
            "40Y",
            "50Y",
        ]
        data["Code"] = pd.Categorical(
            data["Code"], categories=column_order, ordered=True
        )
        data_piv = data.pivot(columns="Code", values="Yield")
        data_piv_with_missing = data_piv.reindex(
            columns=column_order, fill_value=np.nan
        )
        return data_piv, data_piv_with_missing

    def _interpolate_yield(self):
        match self.interpolation_method:
            case "Cubic":
                data = self.get_yield[0]
                tenor = np.array(data.columns)
                tenor_num = tenor_in_days(tenor)
                new_tenor = np.linspace(0, max(tenor_num), 10000)  # type: ignore
                data_np = data.to_numpy()
                data_clean = clean_data(data_np)
                cub_spline_coef = get_interpolation_coef(data_clean, tenor_num)
                cub_spline_result = get_interpolation_result(cub_spline_coef, new_tenor)
                return cub_spline_result

    def _plot_surface(self):
        """Plot the surface of the yield curve.\n"""
        fig = go.Figure(
            data=go.Surface(
                z=self.interpolate_yield,
                x=np.linspace(0, 50, 10000),
                y=self.get_yield[0].index,
            )
        )
        fig.update_traces(
            contours_z=dict(
                show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
            )
        )
        fig.update_layout(
            title="yield curve",
            autosize=False,
            scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
            width=1200,
            height=1000,
            margin=dict(l=65, r=50, b=65, t=90),
        )
        fig.show()
