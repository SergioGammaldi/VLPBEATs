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

        stz1, stn1, ste1 = filters_new(stz, stn, ste, low_frequency, high_frequency, False, 4)

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
        # if max_closest_tca <= min_closest_tca:

        #     max_closest_tca=idxs+2
        #     min_closest_tca=idxs-2
        #     try:
        #         min_closest_tca_amp=comp_amp[min_closest_tca]
        #         max_closest_tca_amp=comp_amp[max_closest_tca]
        #     except:
        #         continue
            stat = stz.stats.station

        
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
        successfreq, fcharact = frequency_characterization_ps_vlp(stzb, stnb, steb, df,low_frequency_detect,
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
            attribute_pol_plot_ps(stz1,stn1,ste1,inc, comp_amp,azimuth,rectilinearity,planarity,time_pick,pwd_plot,stat,ts_pick,thr,min_list_tca,max_list_tca,min_list_tca_amp,max_list_tca_amp,comp_amp[idx_max_comp_amp])

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

def station_to_be_analised(station_to_include,station_to_exclude):

    list_station_considered=station_to_include
    list_station_not_considered=station_to_exclude


    stats = list(set(list_station_considered))
    if len(list_station_considered)==0 :
        
        stats=list(set(stats) - set(list_station_not_considered))
    else:
        stats=list(set(list_station_considered) - set(list_station_not_considered))
        
    stats=sorted(stats)
    return stats
def makeadir(pwd):
    try:
        os.makedirs(pwd)
        print(f'{pwd} created')
    except OSError as er:
        if er.errno != errno.EEXIST:
            print('already exist')
            print(pwd)
    return
def stream_to_trace(stream,vc):
    stz=stream.select(component=vc)
    stn=stream.select(component='N')
    ste=stream.select(component='E')
    # if len(stz)==2:
    #     stz=stream.select(component=vc)
    #     stn=stream.select(component='N')
    #     ste=stream.select(component='E')
    if len(stz)==1:
       stz_copy=stz[0]
    else:
        stz_copy=[]
    if len(stn)==1:
       stn_copy=stn[0]
    else:
        stn_copy=[]
    if len(ste)==1:
       ste_copy=ste[0]
    else:
        ste_copy=[]
    return stz_copy,stn_copy,ste_copy

def check_vc(stream):
    success=False
    vc=[]


    if len(stream.select(component='Z'))>0:
        vc='Z'
        success=True
    elif len(stream.select(component='V'))>0:     
        vc='V'
        success=True

    return vc,success

def get_minimum_sample(stream,total_nsample,percent):
    success=True
    
    for c in ['Z','N','E']:
        
        trace=stream.select(component=c)
        nrsample=0
        for s in trace:
            nrsample=nrsample+len(s)
        try:
            df=round(trace[0].stats.sampling_rate)
        except:
            success=False
            return success
        if nrsample>=total_nsample*percent:
            success=True
        else:
            success=False

            return success
    return success
def read_chunk(pwd):
    reclen = 512
    chunksize = 100000 * reclen # Around 50 MB

    with io.open(pwd, "rb") as fh:
        while True:
            with io.BytesIO() as buf:
                c = fh.read(chunksize)
                if not c:
                    break
                buf.write(c)
                buf.seek(0, 0)
                st = read(buf)
            # Do something useful!
    return st

def find_one_max(maxf_index, freq, freq_amp):
    count = 0

    array_sum = np.sum(freq_amp)
    array_has_nan = np.isnan(array_sum)
    if array_has_nan or len(maxf_index) == 0:
        maxf_index = 0
        count = count + 1
        return maxf_index

    for i in maxf_index:
        if len(freq) > i > 0 == count:
            maxf_index = i
            count = count + 1
            break
    if count == 0:
        maxf_index = max(maxf_index)
    return maxf_index



def frequency_characterization_ps_vlp( stz: obspy.Trace, stn: obspy.Trace, ste: obspy.Trace, df: float,
                                      fcut0: float, fcut1: float, pwd_plot: str, name_plot: str, pltfreq: bool):
    """
    This function checks if a specified frequency range is present in the spectrum of the signal.

    Parameters:
    - stz: Obspy trace of the vertical component
    - stn: Obspy trace of the North-South component
    - ste: Obspy trace of the East-West component
    - df: Frequency sampling rate
    - fcut0: Minimum frequency range
    - fcut1: Maximum frequency range
    - pwd_plot: Directory for saving the frequency spectrum plot
    - name_plot: Name for saving the frequency spectrum plot
    - pltfreq: Boolean to save or not save the frequency spectrum plot

    Returns:
    - success: Boolean indicating the success of the frequency characterization
    - fcharact: The detected frequency, representing the maximum amplitude within the specified frequency range
    """

    freq_thr=0.9
    success = False
    nsample_freq = len(stz)
    window = 'hann'
    fz, t, Zxx = signal.stft(stz.data, df, window=window, nperseg=len(stz))
    fn, t, Nxx = signal.stft(stn.data, df, window=window, nperseg=len(stn))
    fe, t, Exx = signal.stft(ste.data, df, window=window, nperseg=len(ste))
    stat = stz.stats.station
    fz = fz.round(3)
    valuemin = find_nearest(fz, value=fcut0)
    valuemax = find_nearest(fz, value=fcut1)
    fzmin = np.where(fz == valuemin)[0].tolist()[0]
    fzmax = np.where(fz == valuemax)[0].tolist()[0]
    fz = fz[fzmin:fzmax + 1]
    fn = fn[fzmin:fzmax + 1]
    fe = fe[fzmin:fzmax + 1]

    Zxx = abs(Zxx)[fzmin:fzmax + 1]
    Exx = abs(Exx)[fzmin:fzmax + 1]
    Nxx = abs(Nxx)[fzmin:fzmax + 1]
    Zmax = np.max(Zxx)
    Emax = np.max(Exx)
    Nmax = np.max(Nxx)

    if Zmax > 1e-10:
        Zxx /= Zmax
    else:
        return None,None  # or handle safely

    if Emax > 1e-10:
        Exx /= Emax
    else:
        return None,None

    if Nmax > 1e-10:
        Nxx /= Nmax
    else:
        return None,None


    maxf_index_z, cols = np.where(Zxx > freq_thr)
    maxf_index_e, cols = np.where(Exx > freq_thr)
    maxf_index_n, cols = np.where(Nxx > freq_thr)

    fcharact = np.nan

    if len(maxf_index_z) > 0 and len(maxf_index_e) > 0 and len(maxf_index_n) > 0:

        maxf_index_z = find_one_max(maxf_index_z, fz, Zxx)
        maxf_index_e = find_one_max(maxf_index_e, fe, Exx)
        maxf_index_n = find_one_max(maxf_index_n, fn, Nxx)
        # update to be tested where the maxf_index_z==maxf_index_e==maxf_index_e and the frequency is equal for all the componetn
        if maxf_index_z==maxf_index_e==maxf_index_e or fcut0 <= fz[maxf_index_z] <= fcut1 or fcut0 <= fe[maxf_index_e] <= fcut1 or fcut0 <= fn[maxf_index_n] <= fcut1:
            success = True
            fcharact = fz[maxf_index_z]

            if pltfreq:
                frequency_plot_and_waveform(fz, fcut0, fcut1, t, Zxx, Exx, Nxx,stz,stn,ste, pwd_plot, f'{stat}_{name_plot}')

        else:
            if pltfreq:

                frequency_plot_and_waveform(fz, fcut0, fcut1, t, Zxx, Exx, Nxx,stz,stn,ste, pwd_plot, f'NO_{stat}_{name_plot}')
    return success, fcharact


def check_if_exist_pass_over(station,time_to_analyze,pwd_csv,pwd_result):
    saved=False
    exist=os.path.isfile(pwd_csv)
    seconds=86400
    if exist:

        file_lock=f'{pwd_result}lock_file_{time_to_analyze}'

        while not saved:
            try:
                with open(file_lock, 'x') as lockfile:


                    csv_data = pd.read_csv(pwd_csv)
                    csv_data = csv_data.loc[:, ~csv_data.columns.str.contains('^Unnamed')]
    
                    df_time=csv_data[csv_data['station'] == station]
                    time_to_analyze=pd.to_datetime(time_to_analyze)
                    # df_time=stat[stat['TCA'] == str(time_to_analyze)]
                    df_time['start_vlp_iso']=pd.to_datetime(df_time['start_vlp_iso'])
                    df_time=df_time.sort_values('start_vlp_iso')
                    df_time=df_time[df_time['start_vlp_iso'].dt.date== time_to_analyze]
                    if any(df_time['start_vlp_iso'].dt.date== time_to_analyze):
                        if (86400-(df_time['start_vlp_iso'].iloc[-1]-time_to_analyze).total_seconds())>120:

                            seconds=86400-(df_time['start_vlp_iso'].iloc[-1]-time_to_analyze).total_seconds()
                            time_to_analyze=df_time['start_vlp_iso'].iloc[-1]
                        
                    
                    # write the PID of the current process so you can debug
                    # later if a lockfile can be deleted after a program crash
                    lockfile.write(str(os.getpid()))
                    saved=True

            
            except OSError as er:
                time.sleep(int(random()*5))
        os.remove(file_lock)
        print('file removed 1')

    return time_to_analyze,seconds
def filters_new(stz: obspy.Trace, stn: obspy.Trace, ste: obspy.Trace, 
                fmin: float, fmax: float, integrate: bool, crn: int):
    for tr in [stz, stn, ste]:
        tr = detrending(tr)
        tr = delete_spike(tr, None)
        tr.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=crn, zerophase=False)
        if integrate:
            tr.integrate()
    return stz, stn, ste
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
    if max_thr_spike==None:
        max_thr_spike=1e6
    while max_data >= max_thr_spike:
        indx_data = np.argmax(np.abs(trace.data))
        trace.data[indx_data] = 0
        max_data = max(abs(trace.data))
    return trace

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
