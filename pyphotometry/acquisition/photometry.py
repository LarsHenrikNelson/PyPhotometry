from collections import namedtuple
import inspect
from itertools import combinations
from pathlib import PurePath, Path

import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd
from patsy import cr
from sklearn.linear_model import LinearRegression
from scipy import signal
import scipy as scp
import statsmodels.api as sm
import yaml

AnalyzedComps = namedtuple("Components", ["df_f", "baseline", "z_array", "peaks_x"])

PlotData = namedtuple(
    "PlotData", ["raw_array", "timestamp", "baseline", "df_f", "z_array", "peaks_x"]
)


class ExpManager:
    """
    ExpManager manages a experiment. It is up to the user on what
    define as an experiment. An experiment could be recordings from
    one animal over many days or many animals in one day. The user
    can add a file or multiple files. The ExpManager treats each
    file or each sheet in a dataframe as a separate sub-experiment.
    To compare signals within a mouse make sure to have each of the
    recordings in a separate column within the same sheet.
    """

    def __init__(self):
        self.exp_dict = {}
        self.filenames = []
        self.new_data = []

    def load_data(self, filenames):
        if isinstance(filenames, str):
            files = [filenames]
            self.parse_files(files)
        elif isinstance(filenames, list):
            files = filenames
            self.parse_files(files)
        else:
            self.new_data = []

    def parse_files(self, files):
        """
        This function creates the acquisition objects. There should be
        a separate aquisition object for each recording session for each
        mouse. The data can either contain an excel file with a sheet per
        recording session or it can be a file per recording session with
        a single sheet.
        """
        for file_path in files:
            filename = PurePath(file_path)
            if filename not in self.filenames:
                self.filenames.append(filename)
                if filename.suffix == ".xls" or filename.suffix == ".xlsx":
                    df = pd.read_excel(filename, sheet_name=None, nrows=0)
                elif filename.suffix == ".csv":
                    df = list(
                        pd.read_csv(filename, on_bad_lines="skip", nrows=0).columns
                    )
                else:
                    pass

            if isinstance(df, dict):
                for exp, j in df.items():
                    columns = list(j.columns.str.replace(" ", ""))
                    column_dict = {key: value for key, value in zip(columns, j)}
                    acq = AcqManager(column_dict, filename, exp)
                    self.exp_dict[exp] = acq
                    self.new_data.append(exp)
            else:
                columns = [i.replace(" ", "") for i in df]
                column_dict = {key: value for key, value in zip(columns, df)}
                acq = AcqManager(column_dict, filename, filename.stem)
                self.exp_dict[filename.stem] = acq
                self.new_data.append(filename.stem)

    def reset_new_data(self):
        self.new_data = []

    def get_raw_data(self, acq_manager, data_id):
        return self.exp_dict.get(acq_manager).retrieve_raw_data(data_id)

    def get_acq_list(self, acq_manager):
        return list(self.exp_dict[acq_manager].column_dict.keys())

    def get_exps(self):
        return list(self.exp_dict.keys())

    def analyze_acq(
        self,
        exp,
        array,
        analysis,
        sec_array,
        time_array,
        smooth,
        find_peaks,
        find_tau,
        peak_threshold,
        peak_direction,
        z_score,
        z_threshold,
        filter_array,
        filter_type,
        input_1,
        input_2,
    ):
        self.exp_dict.get(exp).analyze(
            analysis,
            array,
            sec_array,
            time_array,
            smooth,
            find_peaks,
            find_tau,
            peak_threshold,
            peak_direction,
            z_score,
            z_threshold,
            filter_array,
            filter_type,
            input_1,
            input_2,
        )

    def get_analyzed_data(self, acq_manager, array):
        data = self.exp_dict.get(acq_manager).analyzed_components(array)
        return data

    def save_as(self, dir):
        pass

    def load(self, dir):
        pass


class AcqManager:
    """
    The acquisition manager is for managing recordings captured at the same
    time. The AcqManager has some basic functions to correlation and cross analyze
    multiple signals. The AcqManager is useful in that it prevents duplication of
    shared arrays (such as timestamps) to help with memory usage since large arrays
    can take up a lot of space.
    """

    def __init__(self, column_dict, filename, exp):
        self.exp = exp
        self.filename = filename
        self.column_dict = column_dict
        self.raw_data = {}
        self.analyzed_data = {}
        self.baselined_data = {}
        self.peaks_data = {}
        self.settings = {}

    def retrieve_raw_data(self, data_id):
        print(data_id)
        column = self.column_dict[data_id]
        if data_id not in self.raw_data:
            if self.filename.suffix == ".xls" or self.filename.suffix == ".xlsx":
                data = (
                    pd.read_excel(self.filename, sheet_name=self.exp, usecols=[column])
                    .to_numpy()
                    .flatten()
                )
            elif self.filename.suffix == ".csv":
                data = (
                    pd.read_csv(self.filename, on_bad_lines="skip", usecols=[column])
                    .to_numpy()
                    .flatten()
                )
            else:
                pass
        else:
            data = self.raw_data[data_id]
        return data

    def running_corr(self):
        cross_corr_dict = {}
        cross_corr_combo = combinations(self.analysis_dict.keys(), 2)
        for i in cross_corr_combo:
            s1 = pd.Series(self.analysis_dict[i[0]].df_f)
            s2 = pd.Series(self.analysis_dict[i[1]].df_f)
            corr = s1.rolling(10).corr(s2).to_numpy()
            cross_corr_dict[f"{i[0]}_{i[1]}"] = corr
        self.analysis_dict["correlations"] = cross_corr_dict

    def correlation(self):
        corr_dict = {}
        corr_combo = combinations(self.analysis_dict.keys(), 2)

    def analyze(
        self,
        analysis,
        array,
        sec_array,
        time_array,
        smooth,
        find_peaks,
        find_tau,
        peak_threshold,
        peak_direction,
        z_score,
        z_threshold,
        filter_array,
        filter_type,
        input_1,
        input_2,
    ):
        acq_settings = locals()
        del acq_settings["self"]
        self.settings[array] = acq_settings
        if sec_array != "None":
            sec_array = self.retrieve_raw_data(sec_array)
        if time_array not in self.raw_data.keys():
            self.raw_data["time_array"] = self.retrieve_raw_data(time_array)
        if array not in self.raw_data.keys():
            self.raw_data[array] = self.retrieve_raw_data(array)
        acq_components = Acquisition(
            analysis,
            self.raw_data[array],
            sec_array,
            self.raw_data["time_array"],
            smooth,
            find_peaks,
            find_tau,
            peak_direction,
            peak_threshold,
            z_score,
            z_threshold,
            filter_array,
            filter_type,
            input_1,
            input_2,
        ).analyze()
        data = {
            "df_f": acq_components.df_f,
            "baseline": acq_components.baseline,
            "z_array": acq_components.z_array,
            "peaks_x": acq_components.peaks_x,
        }
        self.analyzed_data[array] = data

    def baselined_acqs(self):
        return list(self.baselined_data.keys())

    def analyzed_components(self, array):
        plot_data = PlotData(
            self.raw_data[array],
            self.raw_data["time_array"],
            self.analyzed_data[array]["baseline"],
            self.analyzed_data[array]["df_f"],
            self.analyzed_data[array]["z_array"],
            self.analyzed_data[array]["peaks_x"],
        )
        return plot_data

    def save_as(self, save_filename):
        self.save_settings(self.settings, save_filename)

    def load_settings(self, path=None):
        if path is None:
            file_name = Path().glob("*.yaml")[0]
        elif PurePath(path).suffix == ".yaml":
            file_name = PurePath(path)
        else:
            directory = Path(path)
            file_name = list(directory.glob("*.yaml"))[0]
        with open(file_name, "r") as file:
            yaml_file = yaml.safe_load(file)
        return yaml_file

    def save_settings(dictionary, save_filename):
        with open(f"{save_filename}.yaml", "w") as file:
            yaml.dump(dictionary, file)

    def save_data(self, save_filename):
        pass


class Acquisition:
    """
    The Acquisition class manages the acquisition for each
    dataframe/subject/recording. Acquisition acts as a class
    factory and choose the specific analysis method to return
    the analyzed data.
    """

    _registry = {}

    def __init_subclass__(cls, analysis_method, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[analysis_method] = cls

    def __new__(
        cls,
        analysis_method,
        array,
        sec_array,
        time_array,
        smooth,
        analyze_peaks,
        find_tau,
        peak_direction,
        threshold,
        z_score,
        z_threshold,
        filter_array,
        filter_type,
        input_1,
        input_2,
    ):
        subclass = cls._registry[analysis_method]
        obj = object.__new__(subclass)
        obj.analyis_method = analysis_method
        obj.array = array
        obj.sec_array = sec_array
        obj.time_array = time_array
        obj.smooth = smooth
        obj.analyze_peaks = analyze_peaks
        obj.find_tau = find_tau
        obj.peak_threshold = threshold
        obj.peak_direction = peak_direction
        obj.z_score = z_score
        obj.z_threshold = z_threshold
        obj.filter_array = filter_array
        obj.filter_type = filter_type
        obj.input_1 = input_1
        obj.input_2 = input_2
        return obj

    def analyze(self):
        raise NotImplementedError

    def deltaf_f(self):
        self.df_f = ((self.array - self.baseline) / self.baseline) * 100

    def z_score_df_f(self):
        if self.z_score:
            std_array = (
                pd.Series(self.df_f)
                .rolling(self.z_threshold, min_periods=10)
                .std()
                .fillna(method="bfill")
            ).to_numpy()
            std_peaks = scp.signal.argrelmin(std_array, order=150)[0]
            mean_of_peaks = np.mean(self.df_f[std_peaks])
            std = np.std(self.df_f[std_peaks])
            self.z_array = (self.df_f - mean_of_peaks) / std
        else:
            self.z_array = np.zeros(self.df_f.shape)

    def ewma_linear_filter(self, array, window, sum_proportion):
        alpha = 1 - np.exp(np.log(1 - sum_proportion) / window)
        b = [alpha]
        a = [1, alpha - 1]
        return signal.filtfilt(b, a, array)

    def savgol(self, array, order, polyorder):
        filt = signal.savgol_filter(array, int(order), polyorder=int(polyorder))
        return filter

    def filter_array_func(self, array):
        if self.filter_type == "savgol":
            filt_array = self.savgol(array, self.input_1, self.input_2)
        elif self.filter_type == "ewma":
            filt_array = self.ewma_linear_filter(array, self.input_1, self.input_2)
        return filt_array

    def find_peaks(self):
        if self.z_score:
            array = self.z_array
        else:
            array = self.df_f
        if self.peak_direction == "positive":
            multiplier = 1
            # comparator = np.greater
        elif self.peak_direction == "negative":
            multiplier = -1
            # comparator = np.less
        if self.find_tau:
            width = 0
        else:
            width = None
        if self.analyze_peaks:
            self.peaks_x, properties = scp.signal.find_peaks(
                x=array * multiplier,
                distance=15,
                prominence=self.peak_threshold,
                rel_height=(1 - (1 / np.e)),
                width=width,
            )

            # self.peak_amps = array[self.peaks_x] - properties["prominences"]
            # self.tau = properties["widths"]

        else:
            self.peaks_x = np.array([])
            self.peak_amps = np.array([])

    # def find_peak_amp(self, array):
    #     prominences, left_bases, right_bases = scp.signal.peak_prominences(
    #         array, self.peaks_x
    #     )
    #     peak_heights = array[self.peaks_x] - prominences  # Amplitude, essentially
    #     return peak_heights, (prominences, left_bases, right_bases)

    # def find_tau_func(self, array):
    #     widths, h_eval, left_ips, right_ips = scp.signal.peak_widths(
    #         array,
    #         self.peaks_x,
    #         rel_height=1 - (1 / e),
    #         prominence_data=(prominences, left_bases, right_bases),
    #     )
    #     taus = rights_ips - self.peaks_x


class Quantile(Acquisition, analysis_method="quantile"):
    def split_array(self):
        self.len_chunks = int(len(self.array) / (self.smooth * 30))
        self.chunks = np.array_split(self.array, self.len_chunks)

    def calc_quantiles(self):
        self.baseline_pts_y = []
        for c in self.chunks:
            bottom5 = np.quantile(c, 0.05)
            self.baseline_pts_y.append(bottom5)

    def poly_fitting(self):
        quants_x = np.linspace(
            start=0, stop=(self.array.shape[0]), num=self.len_chunks, endpoint=False
        )  # Do NOT use arange for this, bad handling of stop location

        """ Fitting 5° polynomial models chunked 5th percentiles """
        poly = Polynomial.fit(
            quants_x,
            self.baseline_pts_y,
            deg=5,
            rcond=None,
            full=False,
            w=None,
        )

        """ Fitted lines, to be used as baselines for photometry signal de-trending and for calculating relative """
        baseline_x = np.arange(1, self.array.shape[0] + 1, 1)
        self.temp_baseline = poly(baseline_x)
        self.baseline_pts_x = quants_x / 30

    def corrected_baseline(self):
        X = sm.add_constant(self.temp_baseline)
        model = sm.RLM(self.array, X, M=sm.robust.norms.HuberT())
        fit = model.fit()
        intercept = fit.params[0]
        slope = fit.params[1]
        self.baseline = (slope * self.temp_baseline) + intercept

    def analyze(self):
        self.split_array()
        self.calc_quantiles()
        self.poly_fitting()
        self.corrected_baseline()
        self.deltaf_f()
        if self.filter_array == "df_f":
            self.df_f = self.filter_array_func(self.df_f)
        self.z_score_df_f()
        if self.filter_array == "z_array":
            self.z_array = self.filter_array_func(self.z_array)
        self.find_peaks()
        comps = AnalyzedComps(self.df_f, self.baseline, self.z_array, self.peaks_x)
        return comps


class CubicSpline(Acquisition, analysis_method="cubic_spline"):
    def analyze(self):
        spl = scp.interpolate.UnivariateSpline(
            self.time_array, self.array, s=self.smooth
        )
        self.baseline = spl(self.sec_array)
        self.deltaf_f()
        if self.filter_array == "df_f":
            self.df_f = self.filter_array_func(self.df_f)
        self.z_score_df_f()
        if self.filter_array == "z_array":
            self.z_array = self.filter_array_func(self.z_array)
        self.find_peaks()
        comps = AnalyzedComps(
            self.df_f,
            self.baseline,
            self.z_array,
            self.peaks_x,
        )
        return comps


class NatSpline(Acquisition, analysis_method="nat_spline"):
    def analyze(self):
        x_basis = cr(self.time_array, df=self.smooth, constraints="center")

        # Fit model to the data
        model = LinearRegression().fit(x_basis, self.array)

        # Get estimates
        self.baseline = model.predict(x_basis)
        self.deltaf_f()
        if self.filter_array == "df_f":
            self.df_f = self.filter_array_func(self.df_f)
        self.z_score_df_f()
        if self.filter_array == "z_array":
            self.z_array = self.filter_array_func(self.z_array)
        self.find_peaks()
        comps = AnalyzedComps(self.df_f, self.baseline, self.z_array, self.peaks_x)
        return comps


class MinPeak(Acquisition, analysis_method="min_peak"):
    def min_points(self):
        self.sec_array = np.arange(self.array.size)
        x_values = signal.argrelmin(self.array, order=self.smooth)[0]
        self.baseline_pts_x = self.sec_array[x_values]
        self.baseline_pts = self.array[x_values]

    def poly_fitting(self):
        """Fitting 5° polynomial models chunked 5th percentiles"""
        poly = Polynomial.fit(
            self.baseline_pts_x,
            self.baseline_pts,
            deg=5,
            rcond=None,
            full=False,
            w=None,
        )

        """ Fitted lines, to be used as baselines for photometry signal de-trending and for calculating relative """
        self.baseline = poly(self.sec_array)

    def analyze(self):
        self.min_points()
        self.poly_fitting()
        self.deltaf_f()
        if self.filter_array == "df_f":
            self.df_f = self.filter_array_func(self.df_f)
        self.z_score_df_f()
        if self.filter_array == "z_array":
            self.z_array = self.filter_array_func(self.z_array)
        self.find_peaks()
        comps = AnalyzedComps(self.df_f, self.baseline, self.z_array, self.peaks_x)
        return comps


class Plexxon(Acquisition, analysis_method="plexxon"):
    def roll_ave(self, y):
        self.baseline = (
            pd.Series(y).rolling(self.smooth[0], min_periods=1).mean().to_numpy()
        )

    def roll_min(self):
        self.baseline = (
            pd.Series(self.baseline)
            .rolling(self.smooth[1], min_periods=1)
            .mean()
            .to_numpy()
        )

    def analyze(self):
        self.roll_ave()
        self.roll_min()
        self.deltaf_f()
        if self.filter_array == "df_f":
            self.df_f = self.filter_array_func(self.df_f)
        self.z_score_df_f()
        if self.filter_array == "z_array":
            self.z_array = self.filter_array_func(self.z_array)
        self.find_peaks()
        comps = AnalyzedComps(self.df_f, self.baseline, self.z_array, self.peaks_x)
        return comps
