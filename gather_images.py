from pathlib import Path
from shutil import copy
import pickle
import os

"""Sort the output of pacs_model_batch.py into succeeded/failed/unresolved folders."""

#succeeded: PSF subtraction not consistent with noise but model residuals are
#failed: PSF subtraction and model residuals both inconsistent with noise
#unresolved: PSF subtraction consistent with noise (no disc fit performed)

rootdir = '../batch_results_1606'


#first make sure the required folders exist
for dir in ['succeeded', 'failed']:
    for subdir in ['images', 'chains', 'corner']:
        os.makedirs(f'{rootdir}/sorted/{dir}/{subdir}/', exist_ok = True)

os.makedirs(f'{rootdir}/sorted/unresolved/images/', exist_ok = True)


for path in Path(rootdir).rglob('*params.pickle'):

    full_path = path.resolve()
    print(full_path)

    with open(str(full_path), 'rb') as input:
        dict = pickle.load(input)


    print(dict['resolved'])
    name = full_path.parents[0].name

    imgname = full_path.parents[0] /  'image_model.png'
    chainsname = full_path.parents[0] / 'chains.pdf'
    cornername = full_path.parents[0] / 'corner.pdf'

    print(name)

    if dict['resolved']:
        subdir = 'succeeded' if dict['model_consistent'] else 'failed'
        copy(imgname, f'{rootdir}/sorted/{subdir}/images/{name}.png')
        copy(chainsname, f'{rootdir}/sorted/{subdir}/chains/{name}.pdf')
        copy(cornername, f'{rootdir}/sorted/{subdir}/corner/{name}.pdf')

    else:
        copy(imgname, f'{rootdir}/sorted/unresolved/images/{name}.png')
