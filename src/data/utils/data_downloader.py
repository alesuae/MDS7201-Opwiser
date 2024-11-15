# TODO: check BancoCentral api connections
import bcchapi
import pandas as pd
import numpy as np
from utils.config import get_config

def download_data_from_api(config: str) -> pd.DataFrame:
    """
    Downloads data from BancoCentral API and returns it as a DataFrame

    Args:
        config (str): Configuration file name
    Returns:
        DataFrames with the downloaded data
    """
    config_dict = get_config(config)

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
        hasta=end,
        variacion=12,
        frecuencia="M",
        observado={key: np.mean for key in monthly_series.keys()},
    )

    exog_quarterly = siete.cuadro(
        series=[pib_series],
        nombres=["pib"],
        desde=start,
        hasta=end,
        frecuencia="Q",
        observado={"pib": "last"},
    )

    assert type(exog_monthly) == pd.DataFrame and type(exog_quarterly) == pd.DataFrame
    return exog_monthly.dropna(), exog_quarterly.dropna()
