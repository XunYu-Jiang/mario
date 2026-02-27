import logging, time
from tqdm import trange, tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import contextlib
import inspect
from log_setting import MyLogging

logger = MyLogging.get_root_logger()
logger.setLevel(logging.DEBUG)

def redirect_log():
    with logging_redirect_tqdm():
        my_bar_fmt = "{desc}:{bar}|{n_fmt}/{total_fmt} {remaining},{rate_fmt}{postfix}"
        for i in trange(3, colour="blue", desc="Iteration", bar_format=my_bar_fmt):
            for k in trange(5, leave=False, colour="cyan",desc="Self play", bar_format=my_bar_fmt):
                logger.debug("console logging redirected to `tqdm.write()`")
                time.sleep(0.5)

@contextlib.contextmanager
def redirect_print():
    old_print = print

    def new_print(*args, **kwargs):
        # If tqdm.write raises error, use built-in print
        try:
            tqdm.write(*args, **kwargs)
        except:
            old_print(*args, **kwargs)

    try:
        # Globaly replace print with new print
        inspect.builtins.print = new_print
        yield
    finally:
        inspect.builtins.print = old_print

# only root logger can redirect to tqdm.write
if __name__ == '__main__':

    # redirect print to tqdm.write
    for i in trange(3):
        with redirect_print():
            print("print redirected to `tqdm.write()`")
            time.sleep(0.5)