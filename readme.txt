Nursery Classification Project for CMSC 5723 (Machine Learning)
Spring 2022, Term 1, Dr. Robert Nix
Written by David Burris

This project was written using Python 3.10.2.

To run this project:
    First I recommend making a virtual environment and installing the libraries found in "requirements.txt".
    This can be done with the following commands in cmd:

    python -m venv my_env
    my_env\Scripts\activate.bat
    pip install -r requirements.txt

    The library installations may take a few minutes to complete.
    Once completed, the project can be run with the following command in cmd:

    python main.py

    After which you will be prompted to input the training file and test file.

References:
    Environment Setup:
        https://www.youtube.com/watch?v=Wuuiga0wKdQ
        https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment
    Data Structures and API:
        https://scikit-learn.org/stable/modules/tree.html#
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.itertuples.html#pandas.DataFrame.itertuples
        https://towardsdatascience.com/renaming-columns-in-a-pandas-dataframe-1d909360ddc6
