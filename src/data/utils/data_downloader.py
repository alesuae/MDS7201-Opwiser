# TODO: check BancoCentral api connections

import requests
import pandas as pd

def download_data_from_api(api_url: str, params=None, api_key:str = None) -> pd.DataFrame:
    """
    Downloads data from BancoCentral API and returns it as a DataFrame

    :param api_url: API URL.
    :param params: GET request parameters.
    :param api_key: API key, if required.
    :return: DataFrame with the downloaded data.
    """
    headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
    response = requests.get(api_url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        return pd.DataFrame(data)  # Transforms the data into a DataFrame
    else:
        print(f"Error retrieving data: {response.status_code}")
        return pd.DataFrame()  # Returns an empty DataFrame in case of error
