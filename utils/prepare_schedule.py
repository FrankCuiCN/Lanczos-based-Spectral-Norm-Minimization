import numpy as np


def prepare_schedule(schedule):
    # example: [[0, 1], [19, 1], [20, 10], [29, 10], [30, 100], [40, 100]]
    # assumptions:
    #     1. np.array(schedule)[:, 0] starts from 0
    #     2. np.array(schedule)[:, 0] has no duplicates and is in ascending order
    #     3. np.array(schedule)[:, 0].max() >= total epoch number - 1
    schedule = np.array(schedule, dtype="float32")
    return np.interp(np.arange(0, schedule[:, 0].max() + 1), schedule[:, 0], schedule[:, 1])
