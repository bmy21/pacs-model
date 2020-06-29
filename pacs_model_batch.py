import pacs_model
import pandas as pd
import numpy as np
from pathlib import Path

"""Run pacs_model on a batch of systems."""


#input file can be generated with get_obs_paths.ipynb
csv_filename = 'input/obs_path_list.csv'
df_in = pd.read_csv(csv_filename)

#the two lines below can be used to select certain systems to fit
#df_in = df_in[df_in.xid == '* 61 Vir']
#df_in.reset_index(drop = True, inplace = True)

num = len(df_in)

for row in df_in.itertuples():
    print(f'Performing fit number {row.Index + 1} of {num} ({row.obsid} / {row.xid})...')

    try:
        if row.chi_star >= 3:
            pacs_model.run(row.path, savepath = f'../batch_results/{row.obsid}/{row.xid}',
                           name = row.xid, dist = row.dist_pc, stellarflux = row.star_mjy,
                           boxsize = 15, hires_scale = 3, include_unres = False,
                           initial_steps = 100, nwalkers = 200, nsteps = 700, burn = 500,
                           ra = row.ra_obs, dec = row.de_obs, test = True,
                           model_type = pacs_model.ModelType.Particle, npart = 100000)
        else:
            print(f"Proceeding to next system (no significant excess: chi = {row.chi_star:.2f})")

    except Exception as e:
        #if any error is encountered, note this and skip to the next system
        print(f"Proceeding to next system (error encountered: {e})")

        continue
