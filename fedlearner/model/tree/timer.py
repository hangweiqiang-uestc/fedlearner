import functools
import logging
import datetime
import time
from collections import defaultdict

total_time = defaultdict(lambda: datetime.timedelta(0.))


def timer(func_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.datetime.now()
            logging.info('%s start at %s', func_name, start_time)
            res = func(*args, **kwargs)
            end_time = datetime.datetime.now()
            logging.info('%s end at %s, use %s', func_name, end_time,
                         end_time - start_time)
            total_time[func_name] += end_time - start_time
            return res

        return wrapper

    return decorator

