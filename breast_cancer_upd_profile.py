from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv("breast_cancer_upd.csv")

profile = ProfileReport(df, title="Breast Cancer Updated Dataset")
profile.to_notebook_iframe()
profile.to_file("breast_cancer_updated.html")
