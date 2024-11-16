from dataset_makers.sales_dataset_maker import SalesDatasetMaker
from dataset_makers.stock_dataset_maker import StockDatasetMaker
from data_integrator.data_integrator import DataIntegrator
from exogenus_data.exogenus_data_extractor import ExogenousDataExtractor

# TODO: remove hardcoded values
# TODO: add logging
# TODO: update pipeline data

# Create and process the sales and stock datasets
sales_maker = SalesDatasetMaker('data')
sales_maker.load_data()
sales_maker.basic_cleaning()
sales_maker.convert_dates('fecha')
sales_data = sales_maker.get_data()

stock_maker = StockDatasetMaker('data')
stock_maker.load_data()
stock_maker.basic_cleaning()
stock_maker.convert_dates('fecha')
stock_data = stock_maker.get_data()

exogenous_extractor = ExogenousDataExtractor(config_mode="exog")
exogenous_data = exogenous_extractor.fetch_data()


# Integrate the datasets
#integrator = DataIntegrator({'sales': sales_data, 'stock': stock_data}, exogenous_data=exogenous_data)
#integrator.merge_datasets()
#integrator.add_exogenous_variables()

# Get the final dataset
#final_data = integrator.get_integrated_data()
#print(final_data.head())
