from typing import Optional
import os
from multiprocessing import Pool, cpu_count
import glob
import re
import logging
from itertools import repeat, chain

import numpy as np
import pandas as pd
from tqdm import tqdm
from sktime.utils import load_data

from datasets import utils

from functools import partial

logger = logging.getLogger('__main__')


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type, mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y


class BaseData(object):

    def set_num_processes(self, n_proc):

        if (n_proc is None) or (n_proc <= 0):
            self.n_proc = cpu_count()  # max(1, cpu_count() - 1)
        else:
            self.n_proc = min(n_proc, cpu_count())


class WeldData(BaseData):
    """
    Dataset class for welding dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        self.set_num_processes(n_proc=n_proc)

        self.all_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        self.all_df = self.all_df.sort_values(by=['weld_record_index'])  # datasets is presorted
        # TODO: There is a single ID that causes the model output to become nan - not clear why
        self.all_df = self.all_df[self.all_df['weld_record_index'] != 920397]  # exclude particular ID
        self.all_df = self.all_df.set_index('weld_record_index')
        self.all_IDs = self.all_df.index.unique()  # all sample (session) IDs
        self.max_seq_len = 66
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = ['wire_feed_speed', 'current', 'voltage', 'motor_current', 'power']
        self.feature_df = self.all_df[self.feature_names]

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        """
        # each file name corresponds to another date. Also tools (A, B) and others.

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        if self.n_proc > 1:
            # Load in parallel
            _n_proc = min(self.n_proc, len(input_paths))  # no more than file_names needed here
            logger.info("Loading {} datasets files using {} parallel processes ...".format(len(input_paths), _n_proc))
            with Pool(processes=_n_proc) as pool:
                all_df = pd.concat(pool.map(WeldData.load_single, input_paths))
        else:  # read 1 file at a time
            all_df = pd.concat(WeldData.load_single(path) for path in input_paths)

        return all_df

    @staticmethod
    def load_single(filepath):
        df = WeldData.read_data(filepath)
        df = WeldData.select_columns(df)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            logger.warning("{} nan values in {} will be replaced by 0".format(num_nan, filepath))
            df = df.fillna(0)

        return df

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv, which typically contains a day of datasets of various weld sessions.
        """
        df = pd.read_csv(filepath)
        return df

    @staticmethod
    def select_columns(df):
        """"""
        df = df.rename(columns={"per_energy": "power"})
        # Sometimes 'diff_time' is not measured correctly (is 0), and power ('per_energy') becomes infinite
        is_error = df['power'] > 1e16
        df.loc[is_error, 'power'] = df.loc[is_error, 'true_energy'] / df['diff_time'].median()

        df['weld_record_index'] = df['weld_record_index'].astype(int)
        keep_cols = ['weld_record_index', 'wire_feed_speed', 'current', 'voltage', 'motor_current', 'power']
        df = df[keep_cols]

        return df


class TSRegressionArchive(BaseData):
    """
    Dataset class for datasets included in:
        1) the Time Series Regression Archive (www.timeseriesregression.org), or
        2) the Time Series Classification Archive (www.timeseriesclassification.com)
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        #self.set_num_processes(n_proc=n_proc)

        self.config = config

        self.all_df, self.labels_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):

        # Every row of the returned df corresponds to a sample;
        # every column is a pd.Series indexed by timestamp and corresponds to a different dimension (feature)
        if self.config['task'] == 'regression':
            df, labels = utils.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True, replace_missing_vals_with='NaN')
            labels_df = pd.DataFrame(labels, dtype=np.float32)
        elif self.config['task'] == 'classification':
            df, labels = load_data.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True, replace_missing_vals_with='NaN')
            labels = pd.Series(labels, dtype="category")
            self.class_names = labels.cat.categories
            labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss
        else:  # e.g. imputation
            try:
                data = load_data.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                                     replace_missing_vals_with='NaN')
                if isinstance(data, tuple):
                    df, labels = data
                else:
                    df = data
            except:
                df, _ = utils.load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                                 replace_missing_vals_with='NaN')
            labels_df = None

        lengths = df.applymap(lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series
        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        # most general check: len(np.unique(lengths.values)) > 1:  # returns array of unique lengths of sequences
        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            logger.warning("Not all time series dimensions have same length - will attempt to fix by subsampling first dimension...")
            df = df.applymap(subsample)  # TODO: this addresses a very specific case (PPGDalia)

        if self.config['subsample_factor']:
            df = df.applymap(lambda x: subsample(x, limit=0, factor=self.config['subsample_factor']))

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
            logger.warning("Not all samples have same length: maximum length set to {}".format(self.max_seq_len))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)
        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0]*[row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df


class PMUData(BaseData):
    """
    Dataset class for Phasor Measurement Unit dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length (optional). Used only if script argument `max_seq_len` is not
            defined.
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        self.set_num_processes(n_proc=n_proc)

        self.all_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)

        if config['data_window_len'] is not None:
            self.max_seq_len = config['data_window_len']
            # construct sample IDs: 0, 0, ..., 0, 1, 1, ..., 1, 2, ..., (num_whole_samples - 1)
            # num_whole_samples = len(self.all_df) // self.max_seq_len  # commented code is for more general IDs
            # IDs = list(chain.from_iterable(map(lambda x: repeat(x, self.max_seq_len), range(num_whole_samples + 1))))
            # IDs = IDs[:len(self.all_df)]  # either last sample is completely superfluous, or it has to be shortened
            IDs = [i // self.max_seq_len for i in range(self.all_df.shape[0])]
            self.all_df.insert(loc=0, column='ExID', value=IDs)
        else:
            # self.all_df = self.all_df.sort_values(by=['ExID'])  # dataset is presorted
            self.max_seq_len = 30

        self.all_df = self.all_df.set_index('ExID')
        # rename columns
        self.all_df.columns = [re.sub(r'\d+', str(i//3), col_name) for i, col_name in enumerate(self.all_df.columns[:])]
        #self.all_df.columns = ["_".join(col_name.split(" ")[:-1]) for col_name in self.all_df.columns[:]]
        self.all_IDs = self.all_df.index.unique()  # all sample (session) IDs

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        self.feature_names = self.all_df.columns  # all columns are used as features
        self.feature_df = self.all_df[self.feature_names]

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        """

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        if self.n_proc > 1:
            # Load in parallel
            _n_proc = min(self.n_proc, len(input_paths))  # no more than file_names needed here
            logger.info("Loading {} datasets files using {} parallel processes ...".format(len(input_paths), _n_proc))
            with Pool(processes=_n_proc) as pool:
                all_df = pd.concat(pool.map(PMUData.load_single, input_paths))
        else:  # read 1 file at a time
            all_df = pd.concat(PMUData.load_single(path) for path in input_paths)

        return all_df

    @staticmethod
    def load_single(filepath):
        df = PMUData.read_data(filepath)
        #df = PMUData.select_columns(df)
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            logger.warning("{} nan values in {} will be replaced by 0".format(num_nan, filepath))
            df = df.fillna(0)

        return df

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv, which typically contains a day of datasets of various weld sessions.
        """
        df = pd.read_csv(filepath)
        return df
    

class PoseErrorData(BaseData):
    """
    Dataset class for Machine dataset.
    Attributes:
        all_df: dataframe indexed by ID, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_dir, file_list=None, pattern=None, n_proc=1, limit_size=None, config=None):

        self.set_num_processes(n_proc=n_proc)
        self.window_length = 48

        # Modify the function load_all
        #self.all_df = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        result = self.load_all(root_dir, file_list=file_list, pattern=pattern)
        feature_df_list, label_df_list, sequence_list = zip(*result)
        #print(feature_df_list)
        #print(label_df_list)
        #print(sequence_list)
        self.all_df = pd.concat(feature_df_list, axis=0, ignore_index=True)
        self.all_label_df = pd.concat(label_df_list, axis=0, ignore_index=True)

        self.all_df = self.all_df.sort_values(by=['start_image_index'], kind='mergesort')  # datasets is presorted
        self.all_df = self.all_df.set_index('start_image_index')

        self.all_label_df = self.all_label_df.sort_values(by=['start_image_index'], kind='mergesort')
        self.all_label_df = self.all_label_df.set_index('start_image_index')

        self.all_IDs = self.all_df.index.unique()  # all sample (session) IDs
        
        
        self.max_seq_len = self.window_length
        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]
        
        
        self.feature_names = ['brightness','entropy', 'num_images', 'min_corners', 'min_keypoints',\
                              'max_keypoints_diff', 'min_keypoint_dist','min_inliers',\
                              'localba_error', 'localba_visual_error', 'localba_inertial_error',\
                              'acc_magnitude']
        self.label_name = 'error'
        self.feature_df = self.all_df[self.feature_names]
        self.labels_df = self.all_label_df

        self.feature_df.index.rename('index', inplace=True)
        self.labels_df.index.rename('index', inplace=True)

        self.feature_df = self.feature_df.astype('float32')
        self.labels_df = self.labels_df.astype('float32')

    

    def load_all(self, root_dir, file_list=None, pattern=None):
        """
        Loads datasets from csv files contained in `root_dir` into a dataframe, optionally choosing from `pattern`
        Args:
            root_dir: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_dir` to consider.
                Otherwise, entire `root_dir` contents will be used.
            pattern: optionally, apply regex string to select subset of files
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
        """
        # each file name corresponds to another date. Also tools (A, B) and others.

        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_dir, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_dir, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_dir, '*')))

        if pattern is None:
            # by default evaluate on
            selected_paths = data_paths
        else:
            selected_paths = list(filter(lambda x: re.search(pattern, x), data_paths))

        input_paths = [p for p in selected_paths if os.path.isfile(p) and p.endswith('.csv')]
        if len(input_paths) == 0:
            raise Exception("No .csv files found using pattern: '{}'".format(pattern))

        if self.n_proc > 1:
            # Load in parallel
            _n_proc = min(self.n_proc, len(input_paths))  # no more than file_names needed here
            logger.info("Loading {} datasets files using {} parallel processes ...".format(len(input_paths), _n_proc))
            with Pool(processes=_n_proc) as pool:
                result = pool.map(partial(PoseErrorData.load_single, 
                                                    window_length= self.window_length
                                                    ),
                                            input_paths)
        else:  # read 1 file at a time
            result = [PoseErrorData.load_single(path) for path in input_paths]

        return result

    @staticmethod
    def load_single(filepath, window_length):
        df = PoseErrorData.read_data(filepath)
        sequence_name = filepath.split('/')[-1].split('.')[-2]
        
        df = PoseErrorData.select_columns(df)

        rolling_window = 96
        label_df = df.loc[:, 'error'].copy().reset_index(drop=True).rolling(rolling_window).mean()
        label_df.loc[:rolling_window] = df.loc[:rolling_window, 'error']
        label_df = label_df.loc[window_length:].copy().reset_index(drop=True)
        
        num_nan = df.isna().sum().sum()
        if num_nan > 0:
            logger.warning("{} nan values in {} will be replaced by 0".format(num_nan, filepath))
            df = df.fillna(0)

        # implement the sliding window here
        window_number = 0
        slide_df = df.iloc[0:window_length].copy().drop('error', axis=1)
        slide_df = slide_df.assign(start_image_index = window_number)
        for idx in range(window_length+1, len(df)):
            window_number +=1
            window_df = df.iloc[idx-window_length:idx].copy().drop('error', axis=1)
            window_df = window_df.assign(start_image_index = window_number)
            slide_df = pd.concat([slide_df, window_df], axis=0, ignore_index=True)
        slide_df = PoseErrorData.modify_sequence_index(slide_df, sequence_name)

        index = slide_df['start_image_index'].unique()
        label_df = label_df.to_frame(name='error')
        label_df['start_image_index'] = index
        return [slide_df, label_df, sequence_name]

    @staticmethod
    def read_data(filepath):
        """Reads a single .csv, which typically contains a day of datasets of various machine sessions.
        """
        df = pd.read_csv(filepath)
        return df
    
    @staticmethod
    def modify_sequence_index(df, squence_name):
        sequence_num = int(squence_name.split('_')[-2][-1])
        run_num = int(squence_name.split('_')[-1])
        base_num = 1*10**11 + sequence_num*10**8 + run_num*10**5
        df.loc[:, 'start_image_index'] += base_num
        return df

    @staticmethod
    def select_columns(df):
        df['start_image_index'] = df['start_image_index'].astype(int)
        keep_cols = ['start_image_index', 'error',\
                     'brightness','entropy', 'num_images', 'min_corners', 'min_keypoints',\
                     'max_keypoints_diff', 'min_keypoint_dist','min_inliers',\
                     'localba_error', 'localba_visual_error', 'localba_inertial_error',\
                     'acc_magnitude']
        df = df[keep_cols]

        return df


data_factory = {'weld': WeldData,
                'tsra': TSRegressionArchive,
                'pmu': PMUData,
                'pose': PoseErrorData}
