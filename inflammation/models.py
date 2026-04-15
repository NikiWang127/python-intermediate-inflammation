"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains 
inflammation data for a single patient taken over a number of days 
and each column represents a single day across all patients.
"""

import numpy as np


def load_csv(filename):
    """Load a NumPy array from a CSV file.

    :param filename: path to the CSV file to load
    :returns: NumPy array containing the inflammation data
    """
    return np.loadtxt(fname=filename, delimiter=',')


def daily_mean(data):
    """Calculate the mean inflammation value for each day.

    :param data: 2D NumPy array of inflammation data
    :returns: 1D NumPy array containing the daily mean values
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the maximum inflammation value for each day.

    :param data: 2D NumPy array of inflammation data
    :returns: 1D NumPy array containing the daily maximum values
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the minimum inflammation value for each day.

    :param data: 2D NumPy array of inflammation data
    :returns: 1D NumPy array containing the daily minimum values
    """
    return np.min(data, axis=0)


def patient_normalise(data):
    """
    Normalise patient data from a 2D inflammation data array.

    NaN values are ignored, and normalised to 0.

    Negative values are rounded to 0.
    """
    max = np.nanmax(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / max[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    normalised[normalised < 0] = 0
    return normalised
