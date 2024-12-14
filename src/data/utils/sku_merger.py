import pandas as pd

def merge_by_sku(sku: str, data1: pd.DataFrame, data2: pd.DataFrame) -> pd.DataFrame:

    # Check columns intersection
    columns1 = list(data1.columns)
    columns2 = list(data2.columns)
    common_names = list(set(columns1).intersection(columns2))

    if len(common_names) > 0:
        print(f"DataFrames have common columns: {common_names}")
        # Merge datasets
        common_names.remove(sku)
        # Supposing that data1 is sales data
        data1 = data1.drop(columns=common_names)
        merged_data = pd.merge(data1, data2, on=sku, how='left')
    
    else:
        print("DataFrames have no common columns.")
        merged_data = pd.merge(data1, data2, on=sku, how='left')
    
    return merged_data
    