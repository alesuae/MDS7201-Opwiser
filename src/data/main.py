from src.data.dataset_makers.sales_dataset_maker import SalesDatasetMaker
from src.data.dataset_makers.stock_dataset_maker import StockDatasetMaker
from src.data.data_integrator.data_integrator import DataIntegrator
from src.data.exogenus_data.exogenus_data_extractor import ExogenousDataExtractor
from src.data.exogenus_data.exogenus_data_selector import ExogenousDataSelector
from src.data.data_integrator.data_aggregator import DataAggregator
from src.data.data_preprocessing.prepare_data import DataPreparer
from src.data.data_preprocessing.data_splitter import DataSplitter

from ydata_profiling import ProfileReport

def data_pipeline():
    # Create and process the sales and stock datasets
    sales_maker = SalesDatasetMaker(config_mode='data')
    sales_maker.load_data()
    sales_maker.basic_cleaning()
    sales_maker.convert_dates('fecha')
    sales_maker.clean_data()
    sales_data = sales_maker.get_data()

    stock_maker = StockDatasetMaker(config_mode='data')
    stock_maker.load_data()
    stock_maker.basic_cleaning()
    stock_maker.convert_dates('fecha')
    stock_maker.clean_data()
    stock_data = stock_maker.get_data()

    # Create and process the exogenous dataset
    exogenous_extractor = ExogenousDataExtractor(config_mode="exog")
    exogenous_data = exogenous_extractor.join_data()
    selector = ExogenousDataSelector(exogenous_data=exogenous_data)
    exog_selected_data = selector.select(exogenous_data)
    print('termine exog')

    # Integrate the datasets
    print('emepece integrator')
    integrator = DataIntegrator(config_mode='data')
    dataset = integrator.integrate(sales_data, stock_data, exog_selected_data)
    print('y aqui?')
    profile_pp = ProfileReport(dataset, title="Profiling Report")
    profile_pp.to_file("your_report_int.html")

    # Aggregate data (config in data.config.yaml)
    aggregator = DataAggregator(config_mode='data')
    aggregator.interactive_aggregation_setup(dataset)
    aggregated_data = aggregator.aggregate(dataset)

    print(aggregated_data.head())

    # Preprocess and split data for future training
    preparer = DataPreparer(config_mode='data')
    processed_data = preparer.prepare(aggregated_data)

    splitter = DataSplitter(config_mode='data')
    X_train, X_val, X_test, y_train, y_val, y_test = splitter.split(processed_data, target='venta_total_neto')
    #print(X_train)
    return X_train, X_val, X_test, y_train, y_val, y_test
