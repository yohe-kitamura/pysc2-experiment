# coding=utf-8
import os.path

import pandas as pd

DATA_FILE = 'basic_protoss_agent_data'
pd.set_option('display.expand_frame_repr', False)
pd.set_option('line_width', 1000)
pd.set_option('display.max_rows', 100000)

q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

print(str(q_table.idxmax(axis=1)))
