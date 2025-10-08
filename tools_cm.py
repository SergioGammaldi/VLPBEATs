import plot_repository  
import importlib
import pandas as pd
import os
import glob
import numpy as np
from datetime import timedelta
def load_and_preprocess_data(pwd,folder_neolo, catalog_path, startfromtime, totime):
    
        
    txt_file=[]
    dti = pd.date_range(startfromtime,totime, freq='D')

    date=dti.strftime('%Y%m%d').to_list()
    for i in date:
        txt=glob.glob(f'{pwd}/{folder_neolo}/vlp_{i}.txt')
        if len(txt)>0:
            txt_file.append(txt[0])
    dataframes = []
    for file in txt_file:
        try:
            df = pd.read_csv(file)
            # print(f"✅ File OK: {file} - {df.shape[1]} colonne trovate")  # Controlla il numero di colonne
            dataframes.append(df)
        except pd.errors.ParserError as e:
            print(f"❌ ERRORE nel file: {file}\nDettaglio errore: {e}")
            continue  # Interrompe il ciclo appena trova un errore

    # Se tutti i file sono stati letti correttamente, concatenali
    if len(dataframes) > 0:
        df_neolo = pd.concat(dataframes, ignore_index=True)
        print("✅ DataFrame concatenato con successo!")
    df_neolo.to_csv(f'{pwd}/{folder_neolo}/vlp_neolo_analysis_{folder_neolo}.csv')
    print(f'{pwd}/{folder_neolo}/vlp_neolo_analysis_{folder_neolo}.csv')

    df_neolo['start_vlp_iso']=pd.to_datetime(df_neolo['start_vlp_iso']).apply(lambda t: t.replace(tzinfo=None))
    df_neolo['tca_iso'] = pd.to_datetime(df_neolo['tca_iso'], errors='coerce')  # Converte e gestisce errori
    df_neolo['tca_iso'] = df_neolo['tca_iso'].apply(lambda t: t.replace(tzinfo=None) if pd.notna(t) else t)
    df_neolo['tca_iso_sec']=df_neolo['tca_iso'].dt.round(freq='S')
    df_neolo=df_neolo.drop_duplicates(subset=['station','start_vlp_iso'],keep='last')
    df_neolo=df_neolo.drop_duplicates(subset=['station','tca_iso_sec'],keep='last')


    # df_neolo["diff"] = df_neolo["start_vlp_iso"].diff()
    # df_neolo = df_neolo[df_neolo["diff"] >= pd.Timedelta(seconds=5)]


    mask = (df_neolo['start_vlp_iso'] >=  startfromtime) & (df_neolo['start_vlp_iso'] < totime)
    df_neolo=df_neolo[mask]
    df_neolo.sort_values(by='start_vlp_iso')



    cat_str = pd.read_csv(catalog_path)

    # Convert datetime columns
    cat_str['datetime'] = pd.to_datetime(cat_str['datetime'])
    
    mask = (cat_str['datetime'] >=  startfromtime) & (cat_str['datetime'] < totime)
    cat_str=cat_str[mask]

    return df_neolo, cat_str
# Function to calculate f1_score, precision, recall
def f1_score(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall


# Function to compute confusion matrix for each station
def compute_confusion_matrix(dti, df_neolo, cat_str, tolerance, stations):
    results = []
    half_window = timedelta(seconds=tolerance / 2)

    for d in dti:
        day_start = pd.to_datetime(d)
        day_end = day_start + pd.DateOffset(days=1)

        for stat in stations:
            auto_df = df_neolo[
                (df_neolo['station'] == stat) &
                (df_neolo['start_vlp_iso'] >= day_start) &
                (df_neolo['start_vlp_iso'] < day_end)
            ].copy()

            manual_df = cat_str[
                (cat_str['datetime'] >= day_start) &
                (cat_str['datetime'] < day_end)
            ].copy()

            if auto_df.empty or manual_df.empty:
                continue

            auto_df['matched'] = False
            manual_df['matched'] = False

            TP = 0
            rms = []

            auto_df.sort_values('start_vlp_iso', inplace=True)
            manual_df.sort_values('datetime', inplace=True)

            for idx_auto, row_auto in auto_df.iterrows():
                t_auto = row_auto['start_vlp_iso']
                potential = manual_df[
                    (~manual_df['matched']) &
                    (manual_df['datetime'] >= t_auto - half_window) &
                    (manual_df['datetime'] <= t_auto + half_window)
                ]

                if not potential.empty:
                    diffs = (potential['datetime'] - t_auto).abs()
                    best_idx = diffs.idxmin()

                    manual_df.at[best_idx, 'matched'] = True
                    auto_df.at[idx_auto, 'matched'] = True
                    TP += 1
                    rms.append(diffs[best_idx].total_seconds())

            FP = (~auto_df['matched']).sum()
            FN = (~manual_df['matched']).sum()

            lenauto = len(auto_df)
            lenman = len(manual_df)

            rate_tp_manual = TP / lenman if lenman > 0 else 0
            rate_fn = FN / lenman if lenman > 0 else 0
            rate_tp_auto = TP / lenauto if lenauto > 0 else 0
            rate_fp = FP / lenauto if lenauto > 0 else 0

            f1, precision, recall = f1_score(TP, FP, FN)

            results.append({
                'date': day_start.strftime("%Y-%m-%d"),
                'station': stat,
                'TP': TP, 'FP': FP, 'FN': FN,
                'rate_tp_manual': round(rate_tp_manual, 3),
                'rate_fn': round(rate_fn, 3),
                'rate_tp_auto': round(rate_tp_auto, 3),
                'rate_fp': round(rate_fp, 3),
                'f1-score': round(f1, 3),
                'pr': round(precision, 3),
                'rc': round(recall, 3),
                'len_manual': lenman,
                'len_auto': lenauto,
                'manual_vlp_rate': lenman / (86400 / tolerance),
                'auto_vlp_rate': lenauto / (86400 / tolerance),
                'rms': np.mean(np.square(rms)) if rms else None
            })

    return pd.DataFrame(results)
