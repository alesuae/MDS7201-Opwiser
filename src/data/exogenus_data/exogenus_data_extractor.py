import pandas as pd
from utils.data_downloader import download_data_from_api
from base.progress_bar import ProgressBar
from meteostat import Point, Daily
import holidays
from holidays import country_holidays
from datetime import datetime
from utils.config import get_config

class ExogenousDataExtractor:
    def __init__(self, config_mode):
        self.config_mode = config_mode
        self.progress_bar = ProgressBar()

        self.config_dict = get_config(self.config_mode)
        self.dates = self.config_dict["dates"]
        self.weather_data = self.config_dict["weather"]
        self.discounts = self.config_dict["discounts"]

    def fetch_macro_data(self):
        """
        Uses the download_data_from_api function from utils to fetch macroeconomic data.
        """
        self.progress_bar.update_total_steps(1)
        self.progress_bar.log("Requesting data from API...")

        df_macro = download_data_from_api(self.config_mode)
        
        self.progress_bar.check()
        self.progress_bar.close()
        return df_macro
    
    def fetch_weather_data(self):
        """
        Fetches weather data from the Meteostat API
        """
        location = Point(self.weather_data['location'][0], self.weather_data['location'][1])
        start_date = datetime.strptime(self.dates["start"], "%Y-%m-%d")
        end_date = datetime.strptime(self.dates["end"], "%Y-%m-%d")
        weather = Daily(location, start=start_date, end=end_date)
        weather = weather.fetch()
        weather.reset_index(inplace=True)
        weather = weather.rename(columns={"time": "fecha"})
        weather['fecha'] = pd.to_datetime(weather['fecha'])

        weather = weather[self.weather_data["variables"]]
        weather.fillna(0, inplace=True)
        return weather
    
    def fetch_holidays_data(self):
        """
        Fetches holidays data
        """
        years = range(int(self.dates["start"][:4]), int(self.dates["end"][:4]))
        
        cl_holidays = country_holidays(self.weather_data["country"], years=years)
        cl_holidays = pd.DataFrame(list(cl_holidays.keys()), columns=["fecha"])
        cl_holidays['es_festivo'] = 1
        return cl_holidays
    
    def fetch_discounts_data(self):
        """
        Fetches discounts data
        TODO: this should be eventually extracted from a .csv file cointaing the discounts data
        """
        cyber_monday = pd.DataFrame({"fecha": self.discounts["cybermonday"]})
        black_friday = pd.DataFrame({"fecha": self.discounts["blackfriday"]})

        cyber_monday['cyber_monday'] = 1
        black_friday['black_friday'] = 1
        discounts = cyber_monday.merge(black_friday, on='fecha', how='outer')
        return discounts
    
    def join_data(self):
        self.fetch_macro_data = self.preprocess_data(self.fetch_macro_data(), 'fecha')
        self.fetch_holidays_data = self.preprocess_data(self.fetch_holidays_data(), 'fecha')
        self.fetch_weather_data = self.preprocess_data(self.fetch_weather_data(), 'fecha')
        self.fetch_discounts_data = self.preprocess_data(self.fetch_discounts_data(), 'fecha')

        df_combined = pd.merge(self.fetch_macro_data, self.fetch_holidays_data, on='fecha', how='left')
        df_combined = pd.merge(df_combined, self.fetch_weather_data, on='fecha', how='left')
        df_combined = pd.merge(df_combined, self.fetch_discounts_data, on='fecha', how='left')

        df_combined['cyber_monday'] = df_combined['cyber_monday'].fillna(0).astype(int)
        df_combined['black_friday'] = df_combined['black_friday'].fillna(0).astype(int)
        df_combined['es_festivo'] = df_combined['es_festivo'].fillna(0).astype(int)

        return df_combined

    def preprocess_data(self, df, date_column):
        """
        Performs initial data preprocessing
        """
        self.progress_bar.update_total_steps(1)
        self.progress_bar.log(f"Converting {date_column} to date format")
        df[date_column] = pd.to_datetime(df[date_column])
        self.progress_bar.check()
        self.progress_bar.close()
        return df

