import xarray as xr
import pandas as pd

# Load GRIB file using xarray
grib_file = "data.grib"
ds = xr.open_dataset(grib_file, engine="cfgrib")

# Convert dataset to a Pandas DataFrame
df = ds.to_dataframe().reset_index()

df.to_csv("data.csv", index=False)
# # Pivot so that each variable-pressure level becomes a separate column
# df_pivoted = df.pivot(index="time", columns=["isobaricInhPa", "variable"], values="value")
#
# # Flatten multi-index columns
# df_pivoted.columns = [f"{var}_{level}hPa" for level, var in df_pivoted.columns]
#
# # Reset index for a clean DataFrame
# df_pivoted.reset_index(inplace=True)
#
# # Display the DataFrame
# print(df_pivoted.head())
