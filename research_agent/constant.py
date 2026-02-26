"""
Compatibility shim for research_agent.constant
Re-exports from the main constant.py in parent directory
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constant import *
