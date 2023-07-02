from glob import glob
import rasterio
import numpy as np
import os

def find_indices(item, val):
    indices = []
    for idx, value in enumerate(item):
        if value == val:
            indices.append(idx)
    return indices

def sortNames(files, start=11, end=19,time_text=False):
    '''Sorts alpha numeric file names of time series satellite images by time signature
    files: list of files to be sorted
    start: start position of date signature
    end: end position of time signature 
    returns: sorted file names
    '''
    dicts = {nn[start:end]:f'{os.path.split(files[0])[0]}/{nn}' for nn in [os.path.split(kes)[1] for kes in files]}
    outs = [dicts[key] for key in sorted(dicts.keys())]
    if time_text:
        return outs, sorted(list(dicts.keys()))
    else:
        return outs
    
def tensifyScene(files, fold, chunck_size=50000, index_fold=None, save_index=False):
    sntl = sortNames(files[0], start=11, end=19)
    print(f'Sentinel fines: {len(sntl)}')
    
    plnt = files[1] # sortNames(files[1], start=21, end=28)
    print(f'Planet finesl: {len(plnt)}')
    
    tsx = sortNames(files[2],start=28, end=36)
    print(f'TerraSAR-X fines: {len(tsx)}')
    
    if not os.path.exists(index_fold):
        os.makedirs(index_fold, exist_ok=True)
    
    mask = '/mnt/g/EO_Africa_DAT/Data_Masks/plntMask.tif'
    mask = rasterio.open(mask).read(1)
    mask = mask.reshape(-1,1)
    
    folds = ['Sentinel', 'Planet', 'TSX']
    datasets = [sntl, plnt, tsx]
    
    for o, dataset in enumerate(list(zip(datasets,folds))):
#         if dataset[1] not in ['Sentinel', 'Planet']:
        print(f'Processing for dataset {dataset[1]}...')
        out_fold = f'{fold}/{dataset[1]}'
        if not os.path.exists(out_fold):
            os.makedirs(out_fold, exist_ok=True)
        data_files = dataset[0]

        alls = []
        for file in data_files:
            a = rasterio.open(file).read() # [0].reshape(-1,1)
#             alls.append(a)
            print(file)
            if dataset[1] in ['Sentinel','Planet']:
                aa = np.squeeze(np.hstack((a[0].reshape(-1,1), a[1].reshape(-1,1), a[2].reshape(-1,1), a[3].reshape(-1,1))))
            else:
                aa = np.squeeze(np.hstack((a[0].reshape(-1,1), a[1].reshape(-1,1))))
            alls.append(aa)
        alls = np.dstack(alls)
        yes = list(~np.all(mask == 0, axis=(1,)))  # based on landcover mask
        alls = alls[yes]
        inds = list(range(0, alls.shape[0], chunck_size))

        print(f'Saving arrays for dataset {dataset[1]}')

        for j, ind in enumerate(inds):
            with open(f'{out_fold}/{str(j)}.npy', 'wb') as f:
                np.save(f, np.swapaxes(alls[ind:ind+chunck_size],2,1))   # to convert to [B, T, C]

        print(f'Finding and saving indexes for dataset {dataset[1]}')

        if o == 0 and save_index:
            indexes = find_indices(yes, True)
            indexes = np.array(indexes)

            indexes_all = np.array(yes)
            with open(f'{index_fold}/index_true.npy', 'wb') as ff:
                np.save(ff, indexes)
            with open(f'{index_fold}/index_all.npy', 'wb') as ff:
                np.save(ff, indexes_all)
        del alls
        print(f'Done for dataset {dataset[1]}')

if __name__=="__main__":
    a =  sorted(glob('/mnt/g/EO_Africa_DAT/Sentinel-2/Resampled/*.tif'))
    b =  sorted(glob('/mnt/g/EO_Africa_DAT/Planet_Scope_Analytic/Crops/*.tif'))
    c =  sorted(glob('/mnt/g/EO_Africa_DAT/TeraSAR-X/Coregistered/*.tif'))
    fold = '/mnt/g/EO_Africa_DAT/SITS_ALL_FullSceneSpectralMean_final'
    ind_fold = '/mnt/g/EO_Africa_DAT/AA_IndexOptim/SITS_ALL_index_spectralMean_final'
    tensifyScene(files = (a, b, c), fold=fold, chunck_size=50000, index_fold=ind_fold, save_index=True)
