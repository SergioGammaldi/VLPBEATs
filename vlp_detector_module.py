import datetime
import math
import numpy as np
from scipy.signal import find_peaks
from scipy import signal
import obspy
from obspy import UTCDateTime, Stream
from obspy.signal.polarization import eigval, flinn
from tools_vlp import stream_to_trace, check_vc
from plot_repository import attribute_pol_plot_ps, frequency_plot_and_waveform

def event_trim2(tp_pick: list, min_trim_length: int):
    success = False
    tp_pick2 = []
    tcheck = UTCDateTime(1000)
    for tx in tp_pick:
        if abs(tx - tcheck) > min_trim_length:
            tcheck = UTCDateTime(tx)
            tp_pick2.append(tcheck)
    if len(tp_pick2) > 0:
        success = True
    return tp_pick2, success

def detrending(trace: obspy.Trace) -> obspy.Trace:
    count = 0
    arr_zeros = []
    for data in trace.data[::-1]:
        if data != 0:
            break
        arr_zeros.append(0)
        count += 1
    if arr_zeros:
        trace.data = trace.data[:-count]
    trace.detrend('constant')
    trace.detrend('linear')
    if arr_zeros:
        trace.data = np.append(trace.data, arr_zeros)
    return trace

def delete_spike(trace: obspy.Trace, max_thr_spike: float = 1e6) -> obspy.Trace:
    max_data = max(abs(trace.data))
    while max_data >= max_thr_spike:
        indx_data = np.argmax(np.abs(trace.data))
        trace.data[indx_data] = 0
        max_data = max(abs(trace.data))
    return trace

def filters_new(stz: obspy.Trace, stn: obspy.Trace, ste: obspy.Trace, 
                fmin: float, fmax: float, integrate: bool, crn: int):
    for tr in [stz, stn, ste]:
        tr = detrending(tr)
        tr = delete_spike(tr, None)
        tr.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=crn, zerophase=False)
        if integrate:
            tr.integrate()
    return stz, stn, ste

def smooth_data_convolve_my_average(arr: np.ndarray, span: int):
    re = np.convolve(arr, np.ones(span * 2 + 1) / (span * 2 + 1), mode="same")
    re[0] = np.average(arr[:span])
    for i in range(1, span + 1):
        re[i] = np.average(arr[:i + span])
        re[-i] = np.average(arr[-i - span:])
    return re

def pol_detect(stn: obspy.Trace, ste: obspy.Trace, stz: obspy.Trace, nsample_window: int,
               cosin_taper_perc: float, perc_overlap: float):
    df = round(stz.stats.sampling_rate)
    startcut, endcut = stz.stats.starttime, stz.stats.endtime
    total_time = round(endcut - startcut) * df
    win = (1 / df) * nsample_window
    t1, azimuth, comp_amp, incidence, rectilinearity, planarity = [], [], [], [], [], []
    num = total_time - (nsample_window - (nsample_window * perc_overlap))
    den = nsample_window * perc_overlap
    k = win * perc_overlap
    success = False

    for sl in range(round(num / den)):
        start = startcut + k * sl
        end = start + win
        stz1, stn1, ste1 = stz.slice(start, end), stn.slice(start, end), ste.slice(start, end)
        try:
            leigenv1, leigenv2, leigenv3, r, p, *_ = eigval(ste1, stn1, stz1, fk=[1]*5)
            azi, inc, rect, plan = flinn(Stream(traces=[stz1, stn1, ste1]))
        except:
            continue
        t1.append(end - win / 2)
        comp_amp.append(np.sqrt(leigenv1[0] + leigenv2[0] + leigenv3[0]))
        azimuth.append(azi)
        incidence.append(round(inc))
        rectilinearity.append(r[0])
        planarity.append(p[0])

    if comp_amp:
        success = True
        return (np.array(t1), np.array(comp_amp), np.array(incidence), np.array(azimuth),
                np.array(rectilinearity), np.array(planarity), success)
    return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), False

def get_attributes(stz: obspy.Trace, stn: obspy.Trace, ste: obspy.Trace, cosin_taper_perc: float,
                   overlap: float, win_pol: float, mtl: int, plot: bool):
    df = round(stz.stats.sampling_rate)
    nsamples = win_pol * df
    time_pick, comp_amp, inc, azimuth, rectilinearity, planarity, success = pol_detect(stn, ste, stz, nsamples, cosin_taper_perc, overlap)
    if not success:
        return [], [], [], [], [], [], [], [], False, []
    thrmin = np.sqrt(np.mean(comp_amp ** 2))
    thr = round(thrmin, abs(math.floor(math.log10(abs(thrmin))))) if thrmin != 0 else 0
    peakp, _ = find_peaks(comp_amp, height=thr)
    ts_pick, final_tp = time_pick[peakp], time_pick[peakp]
    success_attrib = len(final_tp) > 0 and thr != 0
    return final_tp, ts_pick, time_pick, comp_amp, inc, azimuth, rectilinearity, planarity, success_attrib, [thr]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def get_index_from_pick(array_date: list, pick: UTCDateTime, df: float):
    utc_pick = UTCDateTime(pick)
    pick_n = [i for i in array_date if abs(utc_pick - i) < 1 / df]
    if pick_n:
        indx_pick = np.where(array_date == min(pick_n, key=lambda x: abs(x - utc_pick)))[0][0]
    else:
        indx_pick = 'NaT'
    return indx_pick

def find_one_max(maxf_index, freq, freq_amp):
    if len(maxf_index) == 0 or np.isnan(np.sum(freq_amp)):
        return 0
    return maxf_index[0]

def frequency_characterization_ps_vlp(stz: obspy.Trace, stn: obspy.Trace, ste: obspy.Trace, df: float,
                                      fcut0: float, fcut1: float, pwd_plot: str, name_plot: str, pltfreq: bool):
    freq_thr = 0.9
    success = False
    fz, t, Zxx = signal.stft(stz.data, df, window='hann', nperseg=len(stz))
    fn, _, Nxx = signal.stft(stn.data, df, window='hann', nperseg=len(stn))
    fe, _, Exx = signal.stft(ste.data, df, window='hann', nperseg=len(ste))
    fz = fz.round(3)
    idx_min = np.argmin(abs(fz - fcut0))
    idx_max = np.argmin(abs(fz - fcut1)) + 1
    fz, fn, fe = fz[idx_min:idx_max], fn[idx_min:idx_max], fe[idx_min:idx_max]
    Zxx, Nxx, Exx = abs(Zxx[idx_min:idx_max]), abs(Nxx[idx_min:idx_max]), abs(Exx[idx_min:idx_max])
    for arr in [Zxx, Exx, Nxx]:
        mx = np.max(arr)
        if mx > 1e-10:
            arr /= mx
        else:
            return None, None
    maxf_index_z, _ = np.where(Zxx > freq_thr)
    maxf_index_e, _ = np.where(Exx > freq_thr)
    maxf_index_n, _ = np.where(Nxx > freq_thr)
    fcharact = np.nan
    if maxf_index_z.size and maxf_index_e.size and maxf_index_n.size:
        maxf_index_z = find_one_max(maxf_index_z, fz, Zxx)
        fcharact = fz[maxf_index_z]
        success = True
        if pltfreq:
            frequency_plot_and_waveform(fz, fcut0, fcut1, t, Zxx, Exx, Nxx, stz, stn, ste, pwd_plot, name_plot)
    return success, fcharact
