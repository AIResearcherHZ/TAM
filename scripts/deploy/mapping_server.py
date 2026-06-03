#!/usr/bin/env python3

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.40")

from simadaptor.deploy.mapping_server import main


if __name__ == "__main__":
    main()
