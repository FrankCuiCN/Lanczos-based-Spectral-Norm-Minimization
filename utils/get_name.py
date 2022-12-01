import numpy as np

from datetime import datetime


def get_name():
    now = datetime.now()
    out = f"{now.year:04d}{now.month:02d}{now.day:02d}"
    out += "_"
    out += f"{now.hour:02d}{now.minute:02d}{now.second:02d}"
    out += "_"
    out += f"{now.microsecond:06d}"
    out += "_"
    out += "("
    out += "".join([str(np.random.randint(10)) for _ in range(5)])
    out += ")"
    return out
