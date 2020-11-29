import nilearn

import warnings
warnings.filterwarnings("ignore")

def tr_extractor(f):
    """
        Extracts repetition time (TR) for fMRI series from `.nii` data header.
    """
    import nibabel as nib
    
    try:
        img = nib.load(f)
        header = img.header
        dim = header["pixdim"].tolist()
        h = dim[4]
        return h
    except FileNotFoundError:
        print("NIFTI file not found")

def make_correlation_matrix(path_to_fmriprep_data, 
                            path_to_save_connectivity_matrices,
                            path_to_save_ts = False):
    """
        Process the fmriprep preprocessed functional MRI time-series into 2D correlation matrix as DataFrame using Nilearn lib. 
        Takes `fmriprep/preproc` file as input, frequently with suffix "MNI152NLin2009cAsym_preproc.nii.gz".
        Saves in dedicated folder `path_to_save_connectivity_matrices`.
    """
    import os 
    
    import pandas as pd
    import numpy as np

    import nilearn
    from nilearn import datasets
    from nilearn.image import concat_imgs
    from nilearn.input_data import NiftiLabelsMasker
    from nilearn.image import high_variance_confounds
    from nilearn.connectome import ConnectivityMeasure
    
    tr = tr_extractor(path_to_fmriprep_data)
    subject_name = path_to_fmriprep_data.split('/')[-1][4:11]

    dataset = datasets.fetch_atlas_aal(version='SPM12', data_dir='./datadir/', url=None, resume=True, verbose=0)
    atlas_filename = dataset.maps
    labels = dataset.labels
    correlation_measure = ConnectivityMeasure(kind='correlation')

    img = concat_imgs(path_to_fmriprep_data, auto_resample=True, verbose=0)
    atlas = nilearn.image.resample_to_img(atlas_filename, img, interpolation='nearest', copy=True, order='F', clip=False)
    masker = NiftiLabelsMasker(labels_img=atlas, standardize=True,
                                                detrend = True, low_pass=0.08,
                                                high_pass=0.009, t_r=tr,
                                                memory='nilearn_cache', memory_level=1,
                                                verbose=0)

    confounds = high_variance_confounds(img, 1)
    time_series = masker.fit_transform(img, confounds)
    
    if path_to_save_ts:
        np.save(path_to_save_ts + '/'+ subject_name, time_series)

    correlation_matrix = correlation_measure.fit_transform([time_series])[0]
    np.fill_diagonal(correlation_matrix, 1)
    df = pd.DataFrame(correlation_matrix)
    output_path = os.path.join(path_to_save_connectivity_matrices, subject_name)
    df.to_csv(output_path + '.csv', sep=',')
    
#   print ('TR: ', tr, ' subject:', subject_name)