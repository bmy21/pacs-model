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

    #note: why doesn't AU Mic work?
    if row.xid == "V* AU Mic":#row.xid == "CoRoT-10":# row.xid == "V* AU Mic": #row.xid == "GJ 3634": #48370

        print(row.path)

        try:
            pacs_model.run(row.path, savepath = f'../batch_results_1306/{row.obsid}',
                           name = row.xid, dist = row.dist_pc, stellarflux = 0, #row.star_mjy,
                           hires_scale = 5, include_unres = True,
                           initial_steps = 100,
                           nwalkers = 20, nsteps = 50, burn = 30,
                           test = True)

        except Exception as e:
            #if any error is encountered, simply skip to the next system
            print(f"Error encountered: {str(e)}. Proceeding to next system.")

            continue
