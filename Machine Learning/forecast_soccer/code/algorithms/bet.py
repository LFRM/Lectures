import pandas as pd
import time

import code.algorithms.config as config
import code.algorithms.models as models


def main():
    t0 = time.time()
    logger = config.config_logger(__name__, 10)



    config.time_taken_display(t0)


if __name__ == '__main__':
    main()