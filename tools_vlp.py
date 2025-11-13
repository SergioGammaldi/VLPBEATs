import os
import pandas as pd
from obspy.core import read, Stream,UTCDateTime
from random import random
import plotly.graph_objects as go
import scipy.signal as signal
import math
import numpy as np
from scipy.signal import find_peaks
import io
from obspy.signal.polarization import eigval, flinn
import datetime
import os, errno
import obspy
import time
from plot_repository import attribute_pol_plot,frequency_plot_and_waveform
from obspy import read, Stream, Trace
import os
import io
import errno
import time
from random import random
import math
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import obspy
from obspy import Stream, Trace, UTCDateTime
from scipy import signal
from scipy.signal import find_peaks


def check_if_exist_pass_over(station: str,
                             time_to_analyze: str,
                             pwd_csv: str,
                             pwd_result: str) -> Tuple[pd.Timestamp, int]:
    """
    Check CSV for previous VLP passes for a station and adjust the time_to_analyze
    to avoid re-processing the same event. Uses a simple lockfile to avoid concurrent edits.

    Parameters:
    - station: str
        Station name to look for in CSV.
    - time_to_analyze: str
        Time (ISO string or pandas-parsable) to analyze.
    - pwd_csv: str
        Path to the CSV file storing previous events (must contain 'station' and 'start_vlp_iso' columns).
    - pwd_result: str
        Directory path used as base for the lockfile.

    Returns:
    - time_to_analyze: pd.Timestamp
        Possibly-updated time to analyze (last saved event for that station).
    - seconds: int
        Seconds remaining until a full day (86400) since the last saved event (used by caller).
    """
    saved = False
    exists = os.path.isfile(pwd_csv)
    seconds = 86400  # default: one full day

    # Ensure time_to_analyze is a pandas Timestamp for comparisons
    time_to_analyze = pd.to_datetime(time_to_analyze)

    if not exists:
        # Nothing saved yet; just return original time and full seconds
        return time_to_analyze, seconds

    # Lock file path (unique per analyzed time to avoid collisions)
    file_lock = os.path.join(pwd_result, f'lock_file_{time_to_analyze.isoformat()}')

    # Try to create lockfile exclusively; if it exists, wait small random time and retry
    while not saved:
        try:
            # open with 'x' will fail if file exists
            with open(file_lock, 'x') as lockfile:
                # write pid for debugging if needed
                lockfile.write(str(os.getpid()))

                # Read CSV and remove Unnamed columns if present
                csv_data = pd.read_csv(pwd_csv)
                csv_data = csv_data.loc[:, ~csv_data.columns.str.contains('^Unnamed')]

                # Filter rows by station
                df_time = csv_data[csv_data['station'] == station].copy()
                if df_time.empty:
                    # Nothing to adjust
                    saved = True
                    break

                # Ensure the time column is datetime
                df_time['start_vlp_iso'] = pd.to_datetime(df_time['start_vlp_iso'])
                df_time = df_time.sort_values('start_vlp_iso')

                # Compare only by date: keep events that have same date as time_to_analyze
                # If you want exact comparisons use equality of timestamps instead.
                same_date_mask = df_time['start_vlp_iso'].dt.date == time_to_analyze.date()
                df_time_same_date = df_time.loc[same_date_mask]

                if not df_time_same_date.empty:
                    # Take the most recent saved event that has same date
                    last_saved = df_time_same_date['start_vlp_iso'].iloc[-1]
                    delta_seconds = (time_to_analyze - last_saved).total_seconds()
                    # If last_saved precedes time_to_analyze by more than 120s, adjust:
                    if (86400 - delta_seconds) > 120:
                        seconds = int(86400 - delta_seconds)
                        # Set analysis time to the last saved event (so we won't re-process)
                        time_to_analyze = last_saved

                saved = True

        except OSError:
            # Lockfile exists — sleep a bit and retry (small random jitter)
            wait = int(random() * 5) + 1
            time.sleep(wait)

    # remove lockfile if it exists
    try:
        if os.path.exists(file_lock):
            os.remove(file_lock)
    except Exception:
        # best effort: ignore
        pass

    return pd.to_datetime(time_to_analyze), int(seconds)


def detrending(trace: Trace) -> Trace:
    """
    Remove leading/trailing zero padding and apply constant + linear detrend.

    Parameters:
    - trace: obspy.Trace

    Returns:
    - trace: obspy.Trace
        The modified trace (same object returned for convenience).
    """
    # Remove leading zeros
    data = trace.data
    if len(data) == 0:
        return trace

    # Find indices of first and last non-zero samples
    nonzero_idx = np.nonzero(data)[0]
    if nonzero_idx.size > 0:
        first, last = nonzero_idx[0], nonzero_idx[-1]
        # Trim zeros only if they exist at extremes
        if first > 0 or last < len(data) - 1:
            trimmed = data[first:last + 1].astype(float)
            # replace data in-place (keep original trace.stats)
            trace.data = trimmed
            # Do detrending on the trimmed data
            trace.detrend('constant')
            trace.detrend('linear')
            # Put back zeros at end if originally there were trailing zeros
            # (to preserve original length if desired). Here we append zeros back so length increases.
            # If you prefer to preserve original length exactly, modify as needed.
            if last < len(data) - 1:
                zeros_tail = np.zeros(len(data) - (last + 1))
                trace.data = np.concatenate([trace.data, zeros_tail])
        else:
            trace.detrend('constant')
            trace.detrend('linear')
    else:
        # All zeros — just keep as-is but still detrend (no-op)
        trace.detrend('constant')
        trace.detrend('linear')

    return trace


def delete_spike(trace: Trace, max_thr_spike: Optional[float] = 1e6) -> Trace:
    """
    Remove large spikes from a trace by setting points exceeding `max_thr_spike` to zero.

    Parameters:
    - trace: obspy.Trace
    - max_thr_spike: float or None
        Threshold above which a sample is considered a spike. If None, a large default is used.

    Returns:
    - trace: obspy.Trace
        Trace with spikes removed (in-place).
    """
    if max_thr_spike is None:
        max_thr_spike = 1e6

    # Use absolute values
    absdata = np.abs(trace.data)
    while True:
        max_val = absdata.max() if absdata.size > 0 else 0
        if max_val < max_thr_spike:
            break
        idx = int(np.argmax(absdata))
        trace.data[idx] = 0.0
        absdata[idx] = 0.0  # update local copy to avoid recomputing whole array each loop

    return trace


def filters_new(stz: Trace, stn: Trace, ste: Trace,
                fmin: float, fmax: float, integrate: bool, crn: int) -> Tuple[Trace, Trace, Trace]:
    """
    Apply detrending, spike removal, bandpass filtering and optional integration
    to three component traces. Operates in-place on Trace.data and returns the traces.

    Parameters:
    - stz, stn, ste: obspy.Trace
        Z, N, E traces.
    - fmin: float
        Bandpass low corner (Hz).
    - fmax: float
        Bandpass high corner (Hz).
    - integrate: bool
        If True, integrate the traces (displacement from velocity for example).
    - crn: int
        Number of filter corners.

    Returns:
    - stz, stn, ste: Tuple[Trace, Trace, Trace]
        Processed traces.
    """
    processed = []
    for tr in [stz, stn, ste]:
        # Ensure float data type for processing
        tr.data = tr.data.astype(float)
        # detrend (removes mean and linear trend)
        tr = detrending(tr)
        # remove spikes
        tr = delete_spike(tr, None)
        # bandpass filter (in-place)
        try:
            tr.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=crn, zerophase=False)
        except Exception:
            # If filter fails (e.g. tiny number of samples), skip filtering
            pass
        # integrate if requested
        if integrate:
            try:
                tr.integrate()
            except Exception:
                # integration may fail if metadata missing; skip
                pass
        processed.append(tr)

    return tuple(processed)  # (stz, stn, ste)


def pol_detect(stn: Trace, ste: Trace, stz: Trace,
               nsample_window: int,
               cosin_taper_perc: float,
               perc_overlap: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Polarization detection by sliding-window covariance eigen-decomposition.

    For each sliding window the 3x3 covariance matrix (E, N, Z) is formed and its
    eigenvalues/eigenvectors computed. The principal eigenvector gives direction;
    eigenvalues provide measures of signal energy, rectilinearity and planarity.

    Parameters:
    - stn, ste, stz: obspy.Trace  (N, E, Z components)
    - nsample_window: int
        Number of samples per analysis window.
    - cosin_taper_perc: float
        Cosine taper percentage applied to each window (0-1).
    - perc_overlap: float
        Fractional overlap between windows (0-1).

    Returns:
    - t_centers: np.ndarray
        Array of center times (UTCDateTime converted to seconds since start) for each window.
    - comp_amp: np.ndarray
        Total component amplitude per window (sqrt of sum of eigenvalues).
    - incidence: np.ndarray
        Incidence angles (degrees).
    - azimuth: np.ndarray
        Azimuth angles (degrees, clockwise from North).
    - rectilinearity: np.ndarray
        Rectilinearity measure per window.
    - planarity: np.ndarray
        Planarity measure per window.
    - success: bool
        True if any windows were processed, False otherwise.
    """
    # Basic checks
    if any(tr is None or len(tr.data) == 0 for tr in [stn, ste, stz]):
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), False

    df = round(stz.stats.sampling_rate)
    starttime = stz.stats.starttime
    endtime = stz.stats.endtime
    total_samples = int(round((endtime - starttime) * df))

    # Derived parameters
    step = int(round(nsample_window * (1.0 - perc_overlap)))
    if step <= 0:
        step = 1
    n_windows = max(0, (total_samples - nsample_window) // step + 1)

    # Prepare taper window
    taper = signal.tukey(nsample_window, alpha=cosin_taper_perc) if 0 < cosin_taper_perc < 1 else np.ones(nsample_window)

    t_centers = []
    azimuths = []
    comp_amps = []
    incidences = []
    rects = []
    plans = []

    for w in range(n_windows):
        # sample indices
        start_idx = w * step
        end_idx = start_idx + nsample_window
        # slice traces. Use numpy arrays for speed
        try:
            seg_n = stn.data[start_idx:end_idx].astype(float) * taper
            seg_e = ste.data[start_idx:end_idx].astype(float) * taper
            seg_z = stz.data[start_idx:end_idx].astype(float) * taper
        except Exception:
            continue

        if len(seg_n) < nsample_window:
            # skip incomplete tail window
            continue

        # Form 3 x Ns matrix (E, N, Z)
        X = np.vstack([seg_e, seg_n, seg_z])  # shape (3, nsample)
        # Covariance matrix (3x3). Use unbiased estimator (ddof=1) if enough samples
        C = np.cov(X, bias=True)

        # eigen-decomposition (returns eigenvalues ascending)
        try:
            eigvals, eigvecs = np.linalg.eigh(C)
        except np.linalg.LinAlgError:
            continue

        # Sort eigenvalues descending and reorder eigenvectors
        idx_sort = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx_sort]
        eigvecs = eigvecs[:, idx_sort]

        # Prevent negative eigenvalues due to numerical issues
        eigvals = np.clip(eigvals, a_min=0, a_max=None)

        # total amplitude (sqrt of sum of eigenvalues)
        comp_amp = math.sqrt(np.sum(eigvals))
        comp_amps.append(comp_amp)

        # Rectilinearity and planarity (common measures)
        # rectilinearity: (λ1 - λ2) / λ1  (range [0,1])
        # planarity: (λ2 - λ3) / λ1
        lam1, lam2, lam3 = eigvals
        rect = (lam1 - lam2) / lam1 if lam1 > 0 else 0.0
        plan = (lam2 - lam3) / lam1 if lam1 > 0 else 0.0
        rects.append(rect)
        plans.append(plan)

        # Principal eigenvector (direction of particle motion) -> components [E, N, Z]
        v = eigvecs[:, 0]
        # Normalize
        norm = np.linalg.norm(v)
        if norm == 0:
            v = np.array([0.0, 0.0, 0.0])
        else:
            v = v / norm

        # Compute azimuth (deg): clockwise from North, atan2(E, N)
        az = math.degrees(math.atan2(v[0], v[1]))  # v[0]=E, v[1]=N
        if az < 0:
            az += 360.0
        azimuths.append(az)

        # Incidence (deg): angle from vertical (Z). If v_z is component for Z
        v_z = v[2]  # because ordering is [E, N, Z]
        inc = math.degrees(math.acos(abs(v_z))) if abs(v_z) <= 1.0 else 0.0
        incidences.append(inc)

        # center time of window
        center_time = starttime + ((start_idx + nsample_window / 2) / df)
        t_centers.append(center_time.timestamp)  # seconds since epoch (float)

    if len(comp_amps) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), False

    return (np.array(t_centers), np.array(comp_amps),
            np.array(incidences), np.array(azimuths),
            np.array(rects), np.array(plans), True)


def get_attributes(stz: Trace, stn: Trace, ste: Trace,
                   cosin_taper_perc: float,
                   overlap: float, win_pol: float, mtl: int, plot: bool):
    """
    Wrapper that computes polarization attributes and derives picks/thresholds.

    Parameters:
    - stz, stn, ste: obspy.Trace (Z, N, E)
    - cosin_taper_perc: float
        Fraction (0-1) for Tukey/cosine taper.
    - overlap: float
        Fractional overlap between windows (0-1).
    - win_pol: float
        Window length in seconds for polarization analysis.
    - mtl: int
        Unused here (kept for compatibility).
    - plot: bool
        Whether to generate plots (not implemented here).

    Returns:
    - final_tp: np.ndarray
        Times of detected peaks (UTC timestamps).
    - ts_pick: np.ndarray
        Same as final_tp (kept for compatibility).
    - time_pick: np.ndarray
        All time centers from pol_detect (seconds since epoch).
    - comp_amp: np.ndarray
        Amplitude per window.
    - inc: np.ndarray
        Incidence angles (deg).
    - azimuth: np.ndarray
        Azimuth angles (deg).
    - rectilinearity: np.ndarray
        Rect per window.
    - planarity: np.ndarray
        Plan per window.
    - success_attrib: bool
        True if attributes found.
    - [thr]: list
        Single-element list containing threshold used for peak picking.
    """
    df = round(stz.stats.sampling_rate)
    nsamples = int(round(win_pol * df))

    # Defensive: ensure nsamples >= 3
    if nsamples < 3:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), False, []

    (time_pick, comp_amp, inc, azimuth,
     rectilinearity, planarity, success) = pol_detect(stn, ste, stz,
                                                    nsample_window=nsamples,
                                                    cosin_taper_perc=cosin_taper_perc,
                                                    perc_overlap=overlap)
    if not success:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), False, []

    # threshold based on RMS-like value of comp_amp
    thrmin = np.sqrt(np.mean(comp_amp ** 2)) if comp_amp.size > 0 else 0.0
    if thrmin != 0:
        # round thrmin to same magnitude as thrmin
        thr = round(thrmin, -int(math.floor(math.log10(abs(thrmin)))))
    else:
        thr = 0.0

    # find peaks above threshold
    if thr > 0:
        peaks, _ = find_peaks(comp_amp, height=thr)
    else:
        peaks = np.array([], dtype=int)

    ts_pick = time_pick[peaks] if peaks.size > 0 else np.array([])
    final_tp = ts_pick.copy()

    success_attrib = final_tp.size > 0 and thr != 0
    return final_tp, ts_pick, time_pick, comp_amp, inc, azimuth, rectilinearity, planarity, success_attrib, [thr]




def detect(stream: obspy.Stream, inner_overlap: float, low_frequency: float, high_frequency: float,
           low_frequency_detect: float, high_frequency_detect: float, window_polarization: int,integrate: bool, plot_attr:bool, plot_freq:bool,mtl: int, pwd_plot:str) -> [{}]:
    """
    Detects Very Low-Frequency (VLP) events within a specified time window.

    Parameters:
    - stream: Obspy Stream, the input stream to process
    - inner_overlap: Percentage of window overlap
    - low_frequency: Low bandpass frequency
    - high_frequency: High bandpass frequency
    - low_frequency_detect: Low frequency range for detection
    - high_frequency_detect: High frequency range for detection
    - window_polarization: Time in seconds for the extraction of eigenvalues

    Returns:
    - List of dictionaries with attributes of VLP events
    """
    stz = stream.select(component='Z')[0]
    stn = stream.select(component='N')[0]
    ste = stream.select(component='E')[0]


    result = []

    try:

        stz1, stn1, ste1 = filters_new(stz, stn, ste, low_frequency, high_frequency, integrate, 4)

    except ValueError as e:
        print(e)
        return None

    
    df = round(stz.stats.sampling_rate)
    # nsample_smooth=0
    tp_pick, ts_pick, time_pick, comp_amp, inc, azimuth,rectilinearity,planarity, found, thr = get_attributes(stz1, stn1, ste1,
                                                                                    0.03, inner_overlap,
                                                                                    window_polarization,mtl, 
                                                                                        plot=False)

    if not found:
        return None
    # once having chosen the time of the trim we further characterize the event in frqeuency
    ts_updated = [] 
    min_list_tca = []
    max_list_tca = []
    min_list_tca_amp=[]
    max_list_tca_amp=[]
    idx_max_comp_amp=[]
    index_loop=0
    for x in ts_pick:
        # HERE CHECK THE PICK ACCORDING WITH THE TIME AND THE WE CUT AND CHARACTERIZE THE EVENT +- 6 SEC
        idxs = get_index_from_pick(time_pick, x, df)
 
        idx1 = np.argwhere(np.diff(np.sign(sorted(thr * len(comp_amp)) - comp_amp))).flatten()

        try:
            min_closest_tca = min(idx1, key = lambda x: idxs - x if idxs >= x else idxs)
            min_closest_tca_amp=comp_amp[min_closest_tca]

        except ValueError as e:
            print(e,'is the min_closest_tca')
            continue
        try:
            max_closest_tca = min(idx1, key=lambda x: x-idxs if idxs <= x else idxs)

            max_closest_tca_amp=comp_amp[max_closest_tca]   
        except:
            continue


        
        startp = time_pick[min_closest_tca]
        endp = time_pick[max_closest_tca]
        if startp>endp:
            continue
        stzb = stz.slice(startp, endp)
        stnb = stn.slice(startp, endp)
        steb = ste.slice(startp, endp)

        try:
            stzb, stnb, steb = filters_new(stzb, stnb, steb, low_frequency, high_frequency, integrate, 4)
        except ValueError as e:
            print(e,'is the filter')
            continue
        successfreq, fcharact = frequency_characterization_vlp(stzb, stnb, steb, df,low_frequency_detect,
                                                                  high_frequency_detect, pwd_plot,
                                                                  x.strftime("%Y%m%d_%H_%M_%S"), plot_freq)
        if not successfreq:
            print('successfreq not true')

            continue
        # finally here save the data in hdf5 files but before we check the length of the waveform


        stz_arr = stzb.data
        stn_arr = stnb.data
        ste_arr = steb.data
        ts_updated.append(x)


        try:
            azimuth_tca=(azimuth[idxs])
            azimuth_std = (np.std(azimuth[min_closest_tca:max_closest_tca]))
        except:
            continue
        rsamz = np.sqrt(np.mean(np.square(stz_arr)))
        rsamn = np.sqrt(np.mean(np.square(stn_arr)))
        rsame = np.sqrt(np.mean(np.square(ste_arr)))
        
        rppz = max(stz_arr) - min(stz_arr).item()
        rppn = max(stn_arr) - min(stn_arr).item()
        rppe = max(ste_arr) - min(ste_arr).item()
        incidence_std = (np.std(inc[min_closest_tca:max_closest_tca]))
        incidence_tca = (inc[idxs])
        rectilinearity_tca = (rectilinearity[idxs])
        rectilinearity_std = (np.std(rectilinearity[min_closest_tca:max_closest_tca]))
        planarity_tca = (planarity[idxs])
        planarity_std = (np.std(planarity[min_closest_tca:max_closest_tca]))
        freq = np.round(float(fcharact), 2).item()
        wsize_s = int(endp-startp)
        # TCA COHERENCE AND TIME
        max_tca = comp_amp[idxs]
        tca = float(x.timestamp)
        stat = stz.stats.station
    
        min_list_tca.append(min_closest_tca)
        min_list_tca_amp.append(min_closest_tca_amp)

        max_list_tca.append(max_closest_tca)
        max_list_tca_amp.append(max_closest_tca_amp)
        idx_max_comp_amp.append(idxs)
        start_vlp = float(startp.timestamp)

        result.append({'station': stat, 
                       'start_vlp': start_vlp,
                       'start_vlp_iso': datetime.datetime.fromtimestamp(start_vlp,tz=datetime.timezone.utc).isoformat(timespec='milliseconds'),
                       'tca': tca,  # int(tca * 1_000),  # milliseconds
                       'tca_iso': datetime.datetime.fromtimestamp(tca,tz=datetime.timezone.utc).isoformat(timespec='milliseconds'),
                       'freq': freq, 'azimuth': azimuth_std, 'azimuth_tca': azimuth_tca,
                        'rsam': math.sqrt(rsamz**2 + rsame**2 + rsamn**2),
                       'rsamz': rsamz, 'rsame': rsame, 'rsamn': rsamn,
                       'rpp': math.sqrt(rppz**2 + rppe*2 + rppn**2),
                       'rppz': rppz, 'rppe': rppe, 'rppn': rppn, 'max_tca': float(max_tca),
                       'incidence': float(incidence_std), 'incidence_tca': float(incidence_tca),'rectilinearity_tca':float(rectilinearity_tca),
                       'rectilinearity':float(rectilinearity_std), 'planarity_tca' :float(planarity_tca),'planarity_std':float(planarity_std),
                       'window_size': wsize_s})
        index_loop=index_loop+1

        if plot_attr and index_loop==len(ts_pick):
            attribute_pol_plot(stz1,stn1,ste1,inc, comp_amp,azimuth,rectilinearity,planarity,time_pick,pwd_plot,stat,ts_pick,thr,min_list_tca,max_list_tca,min_list_tca_amp,max_list_tca_amp,comp_amp[idx_max_comp_amp])

    return result if len(result) > 0 else None

def get_index_from_pick(array_date: list, pick: obspy.core.UTCDateTime, df: float):

    """
    Function to get the index in the time array using UTCDateTime

    Parameters:
    - array_date: array of the time used for the polarization analysis in UTCDateTime
    - pick: UTCDateTime to be detected in term of index
    - df: frequency sampling rate

    Returns:
    - indx_pick: index of the time in the array_date
    """

    utc_pick = UTCDateTime(pick)

    pick_n = [i for i in array_date if abs(utc_pick - i) < 1 / df]

    if len(pick_n) > 1:
        pick_n = min(pick_n, key=lambda x: abs(x - utc_pick))

    if pick_n:
        indx_pick = np.where(array_date == pick_n)
        indx_pick = indx_pick[0].tolist()
        indx_pick = indx_pick[0]
    elif len(np.where(array_date == utc_pick)[0]) > 0:
        indx_pick = np.where(array_date == utc_pick)
        indx_pick = indx_pick[0].tolist()
        indx_pick = int(indx_pick[0])

    else:

        print('not founded')
        indx_pick = 'NaT'

    return indx_pick


def station_to_be_analised(station_to_include: list, station_to_exclude: list) -> list:
    """
    Determine which stations should be analyzed based on inclusion and exclusion lists.

    Parameters:
    - station_to_include: list
        List of station names to include.
    - station_to_exclude: list
        List of station names to exclude.

    Returns:
    - stats: list
        Sorted list of stations to be analyzed.
    """
    # If no stations are explicitly included, start from an empty set
    if len(station_to_include) == 0:
        stats = list(set() - set(station_to_exclude))
    else:
        stats = list(set(station_to_include) - set(station_to_exclude))

    stats = sorted(stats)
    return stats


def makeadir(pwd: str) -> None:
    """
    Create a directory if it does not already exist.

    Parameters:
    - pwd: str
        Path of the directory to create.

    Returns:
    - None
    """
    try:
        os.makedirs(pwd)
        print(f'{pwd} created')
    except OSError as er:
        if er.errno == errno.EEXIST:
            print(f'{pwd} already exists')
        else:
            raise
    return


def stream_to_trace(stream: Stream, vc: str) -> tuple:
    """
    Extract traces for the given components ('Z' or 'V'), 'N', and 'E' from a stream.

    Parameters:
    - stream: obspy.Stream
        Input ObsPy stream object.
    - vc: str
        Vertical component ('Z' or 'V').

    Returns:
    - stz_copy, stn_copy, ste_copy: tuple
        Tuple of Trace objects (or empty lists if missing).
    """
    stz = stream.select(component=vc)
    stn = stream.select(component='N')
    ste = stream.select(component='E')

    stz_copy = stz[0] if len(stz) == 1 else []
    stn_copy = stn[0] if len(stn) == 1 else []
    ste_copy = ste[0] if len(ste) == 1 else []

    return stz_copy, stn_copy, ste_copy


def check_vc(stream: Stream) -> tuple:
    """
    Check which vertical component ('Z' or 'V') exists in the stream.

    Parameters:
    - stream: obspy.Stream
        Input ObsPy stream object.

    Returns:
    - vc: str
        Vertical component label ('Z' or 'V').
    - success: bool
        True if a vertical component exists, False otherwise.
    """
    success = False
    vc = None

    if len(stream.select(component='Z')) > 0:
        vc = 'Z'
        success = True
    elif len(stream.select(component='V')) > 0:
        vc = 'V'
        success = True

    return vc, success


def get_minimum_sample(stream: Stream, total_nsample: int, percent: float) -> bool:
    """
    Verify that each component ('Z', 'N', 'E') has at least a given percentage of samples.

    Parameters:
    - stream: obspy.Stream
        Input ObsPy stream object.
    - total_nsample: int
        Expected total number of samples.
    - percent: float
        Minimum percentage (0–1) of samples required per component.

    Returns:
    - success: bool
        True if all components have enough samples, False otherwise.
    """
    for c in ['Z', 'N', 'E']:
        trace = stream.select(component=c)
        nrsample = sum(len(s) for s in trace)

        if len(trace) == 0:
            return False

        try:
            _ = round(trace[0].stats.sampling_rate)
        except Exception:
            return False

        if nrsample < total_nsample * percent:
            return False

    return True


def read_chunk(pwd: str) -> Stream:
    """
    Read a waveform file in chunks to handle large files efficiently.

    Parameters:
    - pwd: str
        Path to the file.

    Returns:
    - st: obspy.Stream
        Combined ObsPy Stream object containing all data read.
    """
    reclen = 512
    chunksize = 100000 * reclen  # ~50 MB per chunk
    st_total = Stream()

    with io.open(pwd, "rb") as fh:
        while True:
            c = fh.read(chunksize)
            if not c:
                break
            with io.BytesIO(c) as buf:
                try:
                    st = read(buf)
                    st_total += st
                except Exception as e:
                    print(f"Error reading chunk: {e}")
                    continue

    return st_total


def find_one_max(maxf_index: list, freq: np.ndarray, freq_amp: np.ndarray) -> int:
    """
    Determine the correct frequency index corresponding to a spectral maximum.

    Parameters:
    - maxf_index: list
        List of indices where local maxima occur.
    - freq: np.ndarray
        Frequency array.
    - freq_amp: np.ndarray
        Amplitude array corresponding to frequencies.

    Returns:
    - maxf_index: int
        Selected index of the dominant frequency.
    """
    count = 0

    if np.isnan(np.sum(freq_amp)) or len(maxf_index) == 0:
        return 0

    for i in maxf_index:
        if 0 < i < len(freq) and count == 0:
            maxf_index = i
            count += 1
            break

    if count == 0:
        maxf_index = max(maxf_index)

    return maxf_index


def frequency_characterization_vlp(stz: Trace, stn: Trace, ste: Trace, df: float,
                                   fcut0: float, fcut1: float, pwd_plot: str,
                                   name_plot: str, pltfreq: bool) -> tuple:
    """
    Check if a specified frequency range is present in the spectrum of the signal.

    Parameters:
    - stz: Trace
        Vertical component
    - stn: Trace
        North-South component
    - ste: Trace
        East-West component
    - df: float
        Sampling rate (Hz)
    - fcut0: float
        Minimum frequency range
    - fcut1: float
        Maximum frequency range
    - pwd_plot: str
        Directory to save plots
    - name_plot: str
        Plot file name
    - pltfreq: bool
        Whether to save frequency spectrum plot

    Returns:
    - success: bool
        True if frequency exists in the range
    - fcharact: float or np.nan
        Detected characteristic frequency
    """
    freq_thr = 0.9
    success = False
    fz, t, Zxx = signal.stft(stz.data, df, window='hann', nperseg=len(stz))
    fn, t, Nxx = signal.stft(stn.data, df, window='hann', nperseg=len(stn))
    fe, t, Exx = signal.stft(ste.data, df, window='hann', nperseg=len(ste))

    fz = fz.round(3)
    valuemin = find_nearest(fz, fcut0)
    valuemax = find_nearest(fz, fcut1)
    fzmin = np.where(fz == valuemin)[0][0]
    fzmax = np.where(fz == valuemax)[0][0]

    # Limit frequency arrays to the desired range
    fz, fn, fe = fz[fzmin:fzmax + 1], fn[fzmin:fzmax + 1], fe[fzmin:fzmax + 1]
    Zxx, Nxx, Exx = abs(Zxx)[fzmin:fzmax + 1], abs(Nxx)[fzmin:fzmax + 1], abs(Exx)[fzmin:fzmax + 1]

    # Normalize
    for arr in [Zxx, Nxx, Exx]:
        max_val = np.max(arr)
        if max_val > 1e-10:
            arr /= max_val
        else:
            return False, np.nan

    maxf_index_z, _ = np.where(Zxx > freq_thr)
    maxf_index_e, _ = np.where(Exx > freq_thr)
    maxf_index_n, _ = np.where(Nxx > freq_thr)

    fcharact = np.nan

    if len(maxf_index_z) > 0 and len(maxf_index_e) > 0 and len(maxf_index_n) > 0:
        maxf_index_z = find_one_max(maxf_index_z, fz, Zxx)
        maxf_index_e = find_one_max(maxf_index_e, fe, Exx)
        maxf_index_n = find_one_max(maxf_index_n, fn, Nxx)

        # Corrected comparison to include North component
        if (maxf_index_z == maxf_index_e == maxf_index_n or
            fcut0 <= fz[maxf_index_z] <= fcut1 or
            fcut0 <= fe[maxf_index_e] <= fcut1 or
            fcut0 <= fn[maxf_index_n] <= fcut1):
            success = True
            fcharact = fz[maxf_index_z]

            if pltfreq:
                frequency_plot_and_waveform(fz, fcut0, fcut1, t, Zxx, Exx, Nxx,
                                           stz, stn, ste, pwd_plot,
                                           f'{stz.stats.station}_{name_plot}')
        elif pltfreq:
            frequency_plot_and_waveform(fz, fcut0, fcut1, t, Zxx, Exx, Nxx,
                                       stz, stn, ste, pwd_plot,
                                       f'NO_{stz.stats.station}_{name_plot}')

    return success, fcharact

def find_nearest(array: np.ndarray, value: float) -> float:
    """
    Return nearest value in array to the given value.

    Parameters:
    - array: np.ndarray
    - value: float

    Returns:
    - nearest value
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def pol_detect(stn: obspy.Trace, ste: obspy.Trace, stz: obspy.Trace,
               nsample_window: int,
               cosin_taper_perc: float,
               perc_overlap: float):
    """
    Polarization detection using ObsPy's eigval and flinn functions.

    Parameters:
    - stn: obspy.Trace
        North-South component trace.
    - ste: obspy.Trace
        East-West component trace.
    - stz: obspy.Trace
        Vertical component trace.
    - nsample_window: int
        Number of samples per analysis window.
    - cosin_taper_perc: float
        Fraction (0-1) of cosine taper applied to window.
    - perc_overlap: float
        Fraction (0-1) of overlap between consecutive windows.

    Returns:
    - t1: np.ndarray
        Array of window center times.
    - comp_amp: np.ndarray
        Total component amplitude per window.
    - incidence: np.ndarray
        Incidence angles (degrees).
    - azimuth: np.ndarray
        Azimuth angles (degrees from North).
    - rectilinearity: np.ndarray
        Rectilinearity per window.
    - planarity: np.ndarray
        Planarity per window.
    - success: bool
        True if at least one window processed successfully.
    """
    df = round(stz.stats.sampling_rate)
    starttime, endtime = stz.stats.starttime, stz.stats.endtime
    total_samples = round((endtime - starttime) * df)
    step = int(round(nsample_window * (1.0 - perc_overlap)))
    if step <= 0:
        step = 1
    n_windows = max(0, (total_samples - nsample_window) // step + 1)

    t1, azimuth, comp_amp, incidence, rectilinearity, planarity = [], [], [], [], [], []

    for w in range(n_windows):
        start = starttime + (w * step) / df
        end = start + nsample_window / df

        stz_win, stn_win, ste_win = stz.slice(start, end), stn.slice(start, end), ste.slice(start, end)

        try:
            # Use ObsPy's eigval and flinn functions
            leigenv1, leigenv2, leigenv3, r, p, *_ = eigval(ste_win, stn_win, stz_win, fk=[1]*5)
            azi, inc, rect, plan = flinn(Stream(traces=[stz_win, stn_win, ste_win]))
        except Exception:
            continue

        t1.append(end - (nsample_window / (2*df)))
        comp_amp.append(np.sqrt(leigenv1[0] + leigenv2[0] + leigenv3[0]))
        azimuth.append(azi)
        incidence.append(round(inc))
        rectilinearity.append(r[0])
        planarity.append(p[0])

    if comp_amp:
        success = True
        return (np.array(t1), np.array(comp_amp), np.array(incidence),
                np.array(azimuth), np.array(rectilinearity), np.array(planarity), success)
    else:
        return (np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), np.array([]), False)


def get_attributes(stz: obspy.Trace, stn: obspy.Trace, ste: obspy.Trace,
                   cosin_taper_perc: float,
                   overlap: float, win_pol: float, mtl: int, plot: bool):
    """
    Compute polarization attributes and find time picks using ObsPy eigval/flinn.

    Parameters:
    - stz, stn, ste: obspy.Trace
        Z, N, E component traces.
    - cosin_taper_perc: float
        Cosine taper fraction for each window.
    - overlap: float
        Fractional overlap between consecutive windows.
    - win_pol: float
        Window length in seconds for polarization analysis.
    - mtl: int
        Currently unused, kept for compatibility.
    - plot: bool
        Whether to generate plots (not implemented here).

    Returns:
    - final_tp: np.ndarray
        Times of detected peaks.
    - ts_pick: np.ndarray
        Same as final_tp (kept for compatibility).
    - time_pick: np.ndarray
        All window center times.
    - comp_amp: np.ndarray
        Component amplitude per window.
    - inc: np.ndarray
        Incidence angles (deg).
    - azimuth: np.ndarray
        Azimuth angles (deg).
    - rectilinearity: np.ndarray
        Rectilinearity per window.
    - planarity: np.ndarray
        Planarity per window.
    - success_attrib: bool
        True if at least one pick was found.
    - [thr]: list
        Threshold used for peak picking.
    """
    df = round(stz.stats.sampling_rate)
    nsamples = int(win_pol * df)

    time_pick, comp_amp, inc, azimuth, rectilinearity, planarity, success = pol_detect(
        stn, ste, stz, nsample_window=nsamples,
        cosin_taper_perc=cosin_taper_perc, perc_overlap=overlap
    )

    if not success:
        return (np.array([]), np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), np.array([]), np.array([]), False, [])

    thrmin = np.sqrt(np.mean(comp_amp ** 2)) if len(comp_amp) > 0 else 0
    thr = round(thrmin, -int(math.floor(math.log10(abs(thrmin))))) if thrmin != 0 else 0
    peaks, _ = find_peaks(comp_amp, height=thr)
    ts_pick = time_pick[peaks] if len(peaks) > 0 else np.array([])
    final_tp = ts_pick.copy()
    success_attrib = len(final_tp) > 0 and thr != 0

    return final_tp, ts_pick, time_pick, comp_amp, inc, azimuth, rectilinearity, planarity, success_attrib, [thr]
def pol_detect(stn: obspy.Trace, ste: obspy.Trace, stz: obspy.Trace,
               nsample_window: int,
               cosin_taper_perc: float,
               perc_overlap: float):
    """
    Polarization detection using ObsPy's eigval and flinn functions.

    Parameters:
    - stn: obspy.Trace
        North-South component trace.
    - ste: obspy.Trace
        East-West component trace.
    - stz: obspy.Trace
        Vertical component trace.
    - nsample_window: int
        Number of samples per analysis window.
    - cosin_taper_perc: float
        Fraction (0-1) of cosine taper applied to window.
    - perc_overlap: float
        Fraction (0-1) of overlap between consecutive windows.

    Returns:
    - t1: np.ndarray
        Array of window center times.
    - comp_amp: np.ndarray
        Total component amplitude per window.
    - incidence: np.ndarray
        Incidence angles (degrees).
    - azimuth: np.ndarray
        Azimuth angles (degrees from North).
    - rectilinearity: np.ndarray
        Rectilinearity per window.
    - planarity: np.ndarray
        Planarity per window.
    - success: bool
        True if at least one window processed successfully.
    """
    df = round(stz.stats.sampling_rate)
    starttime, endtime = stz.stats.starttime, stz.stats.endtime
    total_samples = round((endtime - starttime) * df)
    step = int(round(nsample_window * (1.0 - perc_overlap)))
    if step <= 0:
        step = 1
    n_windows = max(0, (total_samples - nsample_window) // step + 1)

    t1, azimuth, comp_amp, incidence, rectilinearity, planarity = [], [], [], [], [], []

    for w in range(n_windows):
        start = starttime + (w * step) / df
        end = start + nsample_window / df

        stz_win, stn_win, ste_win = stz.slice(start, end), stn.slice(start, end), ste.slice(start, end)

        try:
            # Use ObsPy's eigval and flinn functions
            leigenv1, leigenv2, leigenv3, r, p, *_ = eigval(ste_win, stn_win, stz_win, fk=[1]*5)
            azi, inc, rect, plan = flinn(Stream(traces=[stz_win, stn_win, ste_win]))
        except Exception:
            continue

        t1.append(end - (nsample_window / (2*df)))
        comp_amp.append(np.sqrt(leigenv1[0] + leigenv2[0] + leigenv3[0]))
        azimuth.append(azi)
        incidence.append(round(inc))
        rectilinearity.append(r[0])
        planarity.append(p[0])

    if comp_amp:
        success = True
        return (np.array(t1), np.array(comp_amp), np.array(incidence),
                np.array(azimuth), np.array(rectilinearity), np.array(planarity), success)
    else:
        return (np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), np.array([]), False)


def get_attributes(stz: obspy.Trace, stn: obspy.Trace, ste: obspy.Trace,
                   cosin_taper_perc: float,
                   overlap: float, win_pol: float, mtl: int, plot: bool):
    """
    Compute polarization attributes and find time picks using ObsPy eigval/flinn.

    Parameters:
    - stz, stn, ste: obspy.Trace
        Z, N, E component traces.
    - cosin_taper_perc: float
        Cosine taper fraction for each window.
    - overlap: float
        Fractional overlap between consecutive windows.
    - win_pol: float
        Window length in seconds for polarization analysis.
    - mtl: int
        Currently unused, kept for compatibility.
    - plot: bool
        Whether to generate plots (not implemented here).

    Returns:
    - final_tp: np.ndarray
        Times of detected peaks.
    - ts_pick: np.ndarray
        Same as final_tp (kept for compatibility).
    - time_pick: np.ndarray
        All window center times.
    - comp_amp: np.ndarray
        Component amplitude per window.
    - inc: np.ndarray
        Incidence angles (deg).
    - azimuth: np.ndarray
        Azimuth angles (deg).
    - rectilinearity: np.ndarray
        Rectilinearity per window.
    - planarity: np.ndarray
        Planarity per window.
    - success_attrib: bool
        True if at least one pick was found.
    - [thr]: list
        Threshold used for peak picking.
    """
    df = round(stz.stats.sampling_rate)
    nsamples = int(win_pol * df)

    time_pick, comp_amp, inc, azimuth, rectilinearity, planarity, success = pol_detect(
        stn, ste, stz, nsample_window=nsamples,
        cosin_taper_perc=cosin_taper_perc, perc_overlap=overlap
    )

    if not success:
        return (np.array([]), np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]), np.array([]), np.array([]), False, [])

    thrmin = np.sqrt(np.mean(comp_amp ** 2)) if len(comp_amp) > 0 else 0
    thr = round(thrmin, -int(math.floor(math.log10(abs(thrmin))))) if thrmin != 0 else 0
    peaks, _ = find_peaks(comp_amp, height=thr)
    ts_pick = time_pick[peaks] if len(peaks) > 0 else np.array([])
    final_tp = ts_pick.copy()
    success_attrib = len(final_tp) > 0 and thr != 0

    return final_tp, ts_pick, time_pick, comp_amp, inc, azimuth, rectilinearity, planarity, success_attrib, [thr]

def check_if_exist_pass_over(station: str,
                             time_to_analyze: str,
                             pwd_csv: str,
                             pwd_result: str) -> Tuple[pd.Timestamp, int]:
    """
    Check CSV for previous VLP passes for a station and adjust the time_to_analyze
    to avoid re-processing the same event. Uses a simple lockfile to avoid concurrent edits.

    Parameters:
    - station: str
        Station name to look for in CSV.
    - time_to_analyze: str
        Time (ISO string or pandas-parsable) to analyze.
    - pwd_csv: str
        Path to the CSV file storing previous events (must contain 'station' and 'start_vlp_iso' columns).
    - pwd_result: str
        Directory path used as base for the lockfile.

    Returns:
    - time_to_analyze: pd.Timestamp
        Possibly-updated time to analyze (last saved event for that station).
    - seconds: int
        Seconds remaining until a full day (86400) since the last saved event (used by caller).
    """
    saved = False
    exists = os.path.isfile(pwd_csv)
    seconds = 86400  # default: one full day

    # Ensure time_to_analyze is a pandas Timestamp for comparisons
    time_to_analyze = pd.to_datetime(time_to_analyze)

    if not exists:
        # Nothing saved yet; just return original time and full seconds
        return time_to_analyze, seconds

    # Lock file path (unique per analyzed time to avoid collisions)
    file_lock = os.path.join(pwd_result, f'lock_file_{time_to_analyze.isoformat()}')

    # Try to create lockfile exclusively; if it exists, wait small random time and retry
    while not saved:
        try:
            # open with 'x' will fail if file exists
            with open(file_lock, 'x') as lockfile:
                # write pid for debugging if needed
                lockfile.write(str(os.getpid()))

                # Read CSV and remove Unnamed columns if present
                csv_data = pd.read_csv(pwd_csv)
                csv_data = csv_data.loc[:, ~csv_data.columns.str.contains('^Unnamed')]

                # Filter rows by station
                df_time = csv_data[csv_data['station'] == station].copy()
                if df_time.empty:
                    # Nothing to adjust
                    saved = True
                    break

                # Ensure the time column is datetime
                df_time['start_vlp_iso'] = pd.to_datetime(df_time['start_vlp_iso'])
                df_time = df_time.sort_values('start_vlp_iso')

                # Compare only by date: keep events that have same date as time_to_analyze
                # If you want exact comparisons use equality of timestamps instead.
                same_date_mask = df_time['start_vlp_iso'].dt.date == time_to_analyze.date()
                df_time_same_date = df_time.loc[same_date_mask]

                if not df_time_same_date.empty:
                    # Take the most recent saved event that has same date
                    last_saved = df_time_same_date['start_vlp_iso'].iloc[-1]
                    delta_seconds = (time_to_analyze - last_saved).total_seconds()
                    # If last_saved precedes time_to_analyze by more than 120s, adjust:
                    if (86400 - delta_seconds) > 120:
                        seconds = int(86400 - delta_seconds)
                        # Set analysis time to the last saved event (so we won't re-process)
                        time_to_analyze = last_saved

                saved = True

        except OSError:
            # Lockfile exists — sleep a bit and retry (small random jitter)
            wait = int(random() * 5) + 1
            time.sleep(wait)

    # remove lockfile if it exists
    try:
        if os.path.exists(file_lock):
            os.remove(file_lock)
    except Exception:
        # best effort: ignore
        pass

    return pd.to_datetime(time_to_analyze), int(seconds)

