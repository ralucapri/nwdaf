import pandas as pd

def compute_mean(dataframe, column_name):
    if column_name in dataframe.columns:
        mean_value = dataframe[column_name].mean()
        return round(mean_value, 2)
    else:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame")
    
def compute_peak(dataframe, column_name):
    if column_name in dataframe.columns:
        peak_value = dataframe[column_name].max()
        return peak_value
    else:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame")
    
def get_unique_values(dataframe, column_name):
    if column_name in dataframe.columns:
        unique_values = dataframe[column_name].unique().tolist()
        unique_values_str = ', '.join(map(str, unique_values))
        return unique_values_str
    else:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame")