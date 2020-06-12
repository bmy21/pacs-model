import pacs_model
import pandas as pd
import numpy as np
from pathlib import Path

"""Run pacs_model on a batch of systems."""

#To do: create a progress log, noting when each system is done, so that
#execution can be resumed easily in case of unexpected shutdown


#input file can be generated with get_obs_paths.ipynb
csv_filename = 'input/obs_path_list.csv'
df_in = pd.read_csv(csv_filename)

for row in df_in.itertuples():
    #print(row.Index)

    #note: why doesn't AU Mic work?
    if row.xid == "* alf CrB": #row.xid == "GJ 3634": #48370

        print(row.path)

        try:
            pacs_model.run(20, 50, 30, row.path, '',
            row.dist_pc, row.star_mjy, 1.5, True,
            5, f'../batch_results/{row.obsid}', row.xid, True)

        except Exception as e:
            #if any error is encountered, skip to the next system
            print(f"Error encountered: {str(e)}. Proceeding to next system.")

            continue

    #def run(nwalkers, nsteps, burn, name_image, name_psf, dist, stellarflux,
    #        alpha, include_unres, hires_scale, savepath, name, test)
