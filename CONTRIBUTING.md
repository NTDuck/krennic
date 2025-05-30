Directory structure: https://docs.python-guide.org/writing/structure/

Datasets:
- GlobalTemperatures.csv: https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data?resource=download&select=GlobalTemperatures.csv

How to run stuff:
```cmd
cd polynomial-regression

=======

python -m scripts.tune_polynomial_regression
python -m scripts.tune_ridge_regression
python -m scripts.tune_arima

python -m scripts.plot_polynomial_regression
python -m scripts.plot_ridge_regression
python -m scripts.plot_arima

python -m scripts.run_polynomial_regression
python -m scripts.run_ridge_regression
python -m scripts.run_arima
python -m scripts.run_prophet
```
