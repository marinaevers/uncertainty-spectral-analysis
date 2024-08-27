# Uncertainty-aware Spectral Visualization
TODO: Screenshot
Spectral analysis is commonly used to investigate time series data and identify features of interest such as dominant frequencies. This approach takes the uncertainty of the input data into account. While the transformations are included in the library UADAPy (https://github.com/UniStuttgart-VISUS/uadapy), this repository contains the visual analysis tool to support the whole analysis process.

More information can be found in the paper "Uncertainty-aware Spectral Visualization" by Marina Evers and Daniel Weiskopf.

If you use our approach, please cite our paper.
> tba
```
BibTex tba
```

## Installations
The dependencies can be installed by
```
pip install requirements.txt
```
The code was tested on Windows.

## How to Run?
The visual analysis tool requires the mean and the covariance matrix of the time series. These files store numpy arrays. For metadata, it also needs a config file in json format. We included the artificial dataset from the paper for the approach to be ready to use. You can use the config file as a template.
Start the backend using
```
python app.py
```
When shown on the console, open http://127.0.0.1:8050

## How to Use?
# Load the data
Load the data by selecting the files in the corresponding fields in the top of the application. You can also use drag and drop to load the files.

TODO
