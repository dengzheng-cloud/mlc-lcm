import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../mlc_llm/relax_model')))

from mlc_lcm.relax_model.modules import *
import numpy as np


def main():
    print(build_relax_Linear(Linear))


if __name__ == "__main__":
    main()