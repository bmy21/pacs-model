import pacs_model
import pandas as pd
import numpy as np
from pathlib import Path

"""Run pacs_model on a batch of systems."""


#input file can be generated with get_obs_paths.ipynb
csv_filename = 'input/obs_path_list.csv'
df_in = pd.read_csv(csv_filename)


for row in df_in.itertuples():
    print(f'Performing fit number {row.Index} ({row.obsid} / {row.xid})...')

    try:
        pacs_model.run(row.path, savepath = f'../batch_results_1606/{row.obsid}/{row.xid}',
                       name = row.xid, dist = row.dist_pc, stellarflux = row.star_mjy,
                       hires_scale = 5, include_unres = True,
                       #initial_steps = 10, nwalkers = 20, nsteps = 30, burn = 20,
                       initial_steps = 150, nwalkers = 200, nsteps = 700, burn = 500,
                       ra = row.raj2000, dec = row.dej2000,
                       test = True)

    except Exception as e:
        #if any error is encountered, note this and skip to the next system
        print(f"Error encountered: {str(e)}. Proceeding to next system.")

        continue
