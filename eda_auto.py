from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv('processed/dataset.csv')

profile_pp = ProfileReport(df, title="Profiling Report")
profile_pp.to_file("your_report_int.html")