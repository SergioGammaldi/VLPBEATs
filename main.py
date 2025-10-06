import os
import pathlib
import datetime
import argparse
import yaml
import pandas as pd
from termcolor import colored
from obspy import Stream, UTCDateTime
from obspy.clients.fdsn.client import Client
from obspy.clients.filesystem.sds import Client

from tools_vlp import ( detect,
    get_minimum_sample, check_vc, stream_to_trace, 
    makeadir, station_to_be_analised, check_if_exist_pass_over)

def main_pol_dataset(cfgs):
    # Paths and settings
    dataset_path = cfgs['Folder']['dataset']
    result_path = cfgs['Folder']['result']
    makeadir(result_path)
    # client1 = Client(dataset_path,timeout=60*10)
    client1 = Client('/home/sergio/miniseed/')

    # Parameters
    mtl = cfgs['Parameter']['min_trim_length']
    pltattr = cfgs['Parameter']['plot_attributes']
    pltfreq = cfgs['Parameter']['plot_frequency']
    integrate = cfgs['Parameter']['integrate']
    fcut0, fcut1 = cfgs['Parameter']['characterist_frequency']
    fmin, fmax = cfgs['Parameter']['frequency_band_filters_butterworth']
    overlap, overlap2 = cfgs['Parameter']['overlaps']
    tm1 = cfgs['Parameter']['time_window1']
    tm2 = cfgs['Parameter']['window_polarization']

    win = 60 * tm1

    # Time range
    if cfgs['Parameter']['time_to_be_analised']:
        startfromtime, totime = cfgs['Parameter']['time_to_be_analised']
    else:
        prev_date = datetime.datetime.today() - datetime.timedelta(days=1)
        startfromtime = prev_date.strftime('%Y-%m-%d')
        totime = datetime.datetime.today().strftime('%Y-%m-%d')

    startcut = UTCDateTime(startfromtime)
    endcut = UTCDateTime(totime)
    date_list = pd.date_range(startfromtime, totime, freq='D').strftime('%Y-%m-%dT%H:%M:%SZ').to_list()

    print(f'Processing days: {date_list}')

    # Stations
    station_to_include = cfgs['Parameter']['single_station']
    station_to_exclude = cfgs['Parameter']['no_station']

    stats = station_to_be_analised(station_to_include, station_to_exclude)

    for dat in date_list:
        initial_time = datetime.datetime.now()
        for stat in stats:
            csvfile = pathlib.Path(f'{result_path}/vlp_{startcut.year:02d}{startcut.month:02d}{startcut.day:02d}.txt')
            time, seconds = check_if_exist_pass_over(stat, dat, csvfile, result_path)

            # Skip if data already exists and less than 2h
            if seconds != 86400 and (UTCDateTime(time) + seconds - UTCDateTime(time)) / 60 < 120:
                print(colored('Skipping due to existing data', 'yellow'))
                continue

            startcut = UTCDateTime(time)
            total_time = round(seconds)
            nsamples = win
            k = win * overlap
            num = total_time - (nsamples - k)
            den = k

            print(f'Analyzing range for {stat}: {startcut} to {startcut + total_time}')

            for sl in range(round(num / den)):
                start = startcut + k * sl
                end = start + win


                try:
                    st = client1.get_waveforms('*', stat, "*", "HH*", start, end)
                except ValueError:
                    print(ValueError)
                    continue

                if len(st) != 3:
                    continue

                # Check uniform sampling rate
                if any(tr.stats.sampling_rate != st[0].stats.sampling_rate for tr in st):
                    continue

                if not get_minimum_sample(st, nsamples, 0.6):
                    continue

                st.merge(fill_value=0)
                vc, succ_vc = check_vc(st)
                stz, stn, ste = stream_to_trace(st, vc)
                st = Stream(traces=[stz, stn, ste])
                if st is None:
                    continue

                try:
                    st.resample(5, window='boxcar')
                except Exception as e:
                    print("Resample failed:", e)
                    continue

                result = detect(st, overlap2, fmin, fmax, fcut0, fcut1, tm2, integrate, pltattr, pltfreq, mtl, result_path)
                if result:
                    df = pd.DataFrame(result)

                    df.to_csv(csvfile, mode='a', index=False, header=not csvfile.exists())

        print(f'Time elapsed for {dat}: {datetime.datetime.now() - initial_time}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VLP Detection Utility')
    parser.add_argument('--config-file', dest='config_file', type=str, help='Path to Configuration file')
    args = parser.parse_args()

    cfgs = yaml.safe_load(open(args.config_file))
    makeadir(cfgs['Folder']['result'])
    main_pol_dataset(cfgs)
