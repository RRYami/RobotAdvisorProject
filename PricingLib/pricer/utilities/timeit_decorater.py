import time


def timeit():
    """
    Decorator to time a function.
    :return:
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            func(*args, **kwargs)
            time.sleep(1)
            end = time.time()
            print(f'Time elapsed for the function {func.__name__} is: {end - start} seconds')
            return end - start
        return wrapper
    return decorator
