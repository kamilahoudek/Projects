from ydata_profiling import ProfileReport
import pandas as pd

df = pd.read_csv("Project_Regression_Wine/winequality-red.csv")

profile = ProfileReport(df, title="Wine Quality - original dataset")
profile.to_notebook_iframe()
profile.to_file("wine_quality_orig.html")


df = pd.read_csv("Project_Regression_Wine/cleaned_dataset_no_outliers.csv")

profile = ProfileReport(df, title="Wine Quality - cleaned dataset")
profile.to_notebook_iframe()
profile.to_file("wine_quality_cleaned.html")
