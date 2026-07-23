# tools/inspect_wisig_manytx.py
# -*- coding: utf-8 -*-

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

from datasets.wisig_manytx_loader import inspect_manytx_pkl

PKL_PATH = r"C:\Users\123\Downloads\ManyTx.pkl\ManyTx.pkl"

if __name__ == "__main__":
    inspect_manytx_pkl(PKL_PATH)