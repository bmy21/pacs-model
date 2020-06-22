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
        if row.xid == 'V* AU Mic': pacs_model.run(row.path, savepath = f'../testing/{row.obsid}/{row.xid}',
                       name = row.xid, dist = row.dist_pc, stellarflux = row.star_mjy,
                       boxsize = 13, hires_scale = 7, include_unres = True,
                       initial_steps = 100, nwalkers = 20, nsteps = 30, burn = 20,
                       #initial_steps = 150, nwalkers = 200, nsteps = 700, burn = 500,
                       ra = row.ra_obs, dec = row.de_obs,
                       test = False)

    except Exception as e:
        #if any error is encountered, note this and skip to the next system
        print(f"Error encountered: {str(e)}\nProceeding to next system.")

        continue
