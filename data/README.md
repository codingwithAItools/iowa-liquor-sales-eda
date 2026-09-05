# Data

The raw dataset is not committed because the CSV files are large and can be downloaded from the official source.

Download all five CSV files and either:

1. Upload them directly to a Google Colab session; or
2. Save them in this `data` folder before running the notebook locally.

The notebook finds every `*.csv` file and combines them automatically. It does not request the CSV endpoint directly because the Iowa server may return HTTP 403 to automated pandas requests.
