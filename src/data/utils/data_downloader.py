# TODO: check BancoCentral api connections
import bcchapi
import pandas as pd
import numpy as np
from utils.config import get_config

def download_data_from_api(config_mode: str) -> pd.DataFrame:
    """
    Downloads data from BancoCentral API and returns it as a DataFrame

    Args:
        config (str): Configuration file name
    Returns:
        DataFrames with the downloaded data
    """
    config_dict = get_config(config_mode)

    if "pib" in config_dict["series"].keys():
        pib_series = config_dict["series"].pop("pib")
        monthly_series = config_dict["series"]
    else:
        monthly_series = config_dict["series"]

    try:
        siete = bcchapi.Siete(file=config_dict["creds"])
    except:
        print("Error: Could not connect to the BancoCentral API")
        return pd.DataFrame()

    start = config_dict["dates"]["start"]
    end = config_dict["dates"]["end"]

    # TODO: change hardcoded values to config values
    exog_monthly = siete.cuadro(
        series=list(monthly_series.values()),
        nombres=list(monthly_series.keys()),
        desde=start,
        hasta=end
    )

    exog_quarterly = siete.cuadro(
        series=[pib_series],
        nombres=["pib"],
        desde=start,
        hasta=end,
        frecuencia="Q",
        observado={"pib": "last"}
    )
    # The data comes with "fecha" as index, so we reset it
    exog_monthly = exog_monthly.reset_index()
    exog_quarterly = exog_quarterly.reset_index()
    exog_monthly = exog_monthly.rename(columns={"index": "fecha"})
    exog_quarterly = exog_quarterly.rename(columns={"index": "fecha"})
    exog_monthly = exog_monthly.reset_index(drop=True)
    exog_quarterly = exog_quarterly.reset_index(drop=True)

    exog_monthly["fecha"] = pd.to_datetime(exog_monthly["fecha"])
    exog_quarterly["fecha"] = pd.to_datetime(exog_quarterly["fecha"])

    exog_monthly['trimestre'] = exog_monthly['fecha'].dt.to_period('Q')
    exog_quarterly['trimestre'] = exog_quarterly['fecha'].dt.to_period('Q')

    # merge
    exog_macro = pd.merge(exog_monthly, exog_quarterly, on='trimestre', how='left')
    print(exog_macro.columns)
    # rename fecha_x to fecha
    exog_macro = exog_macro.rename(columns={"fecha_x": "fecha"})
    exog_macro = exog_macro.drop(columns=["fecha_y"])

    df_expanded = exog_macro.set_index('fecha').resample('D').ffill().reset_index()
    return df_expanded