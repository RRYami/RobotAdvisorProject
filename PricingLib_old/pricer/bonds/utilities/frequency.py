from enum import Enum


class FrequencyTypes(Enum):
    ANNUAL = 1
    SEMI_ANNUAL = 2
    QUARTERLY = 4
    MONTHLY = 12
    CONTINUOUS = 99
    ZERO = -1


def annual_frequency(freq_type: FrequencyTypes) -> float:
    """This is a function that takes in a Frequency Type and returns a
    float value for the number of times a year a payment occurs."""
    if isinstance(freq_type, FrequencyTypes) is False:
        print("FinFrequency:", freq_type)
        raise ValueError("Unknown frequency type")

    if freq_type == FrequencyTypes.CONTINUOUS:
        return -1
    elif (
        freq_type == FrequencyTypes.ZERO
    ):  # This means that there is no regular coupon I use 1 to avoid division by zero
        return 1.0
    elif freq_type == FrequencyTypes.ANNUAL:
        return 1.0
    elif freq_type == FrequencyTypes.SEMI_ANNUAL:
        return 2.0
    elif freq_type == FrequencyTypes.QUARTERLY:
        return 4.0
    elif freq_type == FrequencyTypes.MONTHLY:
        return 12.0
