"""Logger that writes to the console and to a file in the run folder."""
import logging
import sys


def get_logger(name, log_file=None):
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    for h in handlers:
        h.setFormatter(fmt)
        log.addHandler(h)
    return log
