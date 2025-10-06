import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go

def frequency_plot_and_waveform(frequency,fmin,fmax,time,Zxx,Exx,Nxx,stz,stn,ste,pwd,test_frequency):
    fig = make_subplots(rows=6, cols=1,shared_xaxes=True,  vertical_spacing=0.02)
    date=stz.times('utcdatetime')
    #plot z waveform and frequency
    fig.add_trace(go.Scatter(x = date, y =stz.data,  marker=dict( color='black', size=20, line=dict(
        color='black', width=1 ) ), name='Waveform Z Component' ),row=1,col=1)
    fig.add_trace(go.Contour(  z=abs(Zxx),      x=time,  y=frequency, coloraxis='coloraxis' ),row=2,col=1)
    fig.update_coloraxes(showscale=False)

    #plot n waveform and frequency

    fig.add_trace(go.Scatter(x = date, y =stn.data,  marker=dict( color='black', size=20, line=dict(
        color='black', width=1 ) ), name='Waveform Z Component' ),row=3,col=1)
    fig.add_trace(go.Contour(  z=abs(Nxx),       x=time,  y=frequency, coloraxis='coloraxis'),row=4,col=1)
    fig.update_coloraxes(showscale=False)

    #plot e waveform and frequency

    fig.add_trace(go.Scatter(x = date, y =ste.data,  marker=dict( color='black', size=20, line=dict(
        color='black', width=1 ) ), name='Waveform Z Component' ),row=5,col=1)
    fig.add_trace(go.Contour(  z=abs(Exx),      x=time,  y=frequency, coloraxis='coloraxis'),row=6,col=1)
    fig.update_coloraxes(showscale=True)
    matr = np.maximum(np.maximum(stz,stn), ste)
    mitr = np.minimum(np.minimum(stz,stn), ste)
    fig.update_xaxes(range=[min(time), max(time)], title_text="<b>Time [s]</b>", row=6, col=1)
    fig.update_yaxes(  range=[fmin,fmax],title_text = "Frequency [Hz] Z",row=2,col=1)
    fig.update_yaxes( range=[fmin,fmax], title_text = "Frequency [Hz] N",row=4,col=1)
    fig.update_yaxes( range=[fmin,fmax], title_text = "Frequency [Hz] E",row=6,col=1)
    fig.update_yaxes(  range=[min(mitr),max(matr)], title_text = "count Z",row=1,col=1)
    fig.update_yaxes(   range=[min(mitr),max(matr)],title_text = "count N",row=3,col=1)
    fig.update_yaxes(   range=[min(mitr),max(matr)],title_text = "count E",row=5,col=1)

    fig.update_layout( coloraxis=dict(colorscale='RdBu', colorbar=dict(title=dict( text="Normalized amplitude spectra",
        side="right",  # You can set this to "top" to position it above the color bar
        font=dict(size=14)  # Adjust the font size or other properties as needed
    ))),showlegend=False)

    fig.update_layout(width=1200, height=800)
    print(f'{pwd}Spectra_analysis_and_waveform{test_frequency}.html')
    fig.write_html(f'{pwd}Spectra_analysis_and_waveform{test_frequency}.html')

    return
def attribute_pol_plot(stz,stn,ste,incidence,three_component_amplitude,azimuth,rectilinearity,planarity,time,pwd,add_name,ts_pick,thr,min_list_tca,max_list_tca,min_list_tca_amp,max_list_tca_amp,max_comp_amp):

    StaName = stz.stats.station
    Endtime = stz.stats.endtime
    timesave=ts_pick[0].strftime("%Y%m%d_%H_%M_%S")
    stz.resample(5)
    stn.resample(5)
    ste.resample(5)

    name_test=f'{add_name}_{StaName}_{timesave}'
    date=stz.times('utcdatetime')

    fig = make_subplots(rows=8, cols=1,shared_xaxes=True, vertical_spacing=0.02 )

    fig.add_trace(go.Scatter(mode='lines',x = time, y =three_component_amplitude,  marker=dict(  color='black',  size=20, line=dict(
        color='Black',width=1 ) ), name='TCA' ),row=1,col=1)
                
    fig.add_trace(go.Scatter(mode='markers',   marker_symbol='line-ew',x =time,
                y=sorted(thr*len(time)),marker=dict(color='red', size=5,line=dict(color='red', width=1)), name='Threshould'),row=1     ,col=1)


    fig.add_trace(go.Scatter(x = date, y =stz.data,  marker=dict( color='black', size=20, line=dict(
        color='black', width=1 ) ), name='Z' ),row=2,col=1)

    fig.add_trace(go.Scatter(x = date, y =stn.data,  marker=dict( color='black', size=20, line=dict(
        color='black', width=1 ) ), name='N' ),row=3,col=1)

    fig.add_trace(go.Scatter(x = date, y =ste.data,  marker=dict( color='black', size=20, line=dict(
        color='black', width=1 ) ), name='E' ),row=4,col=1)

    #calling attribute
    fig.add_trace(go.Scatter(mode='markers',x = time,   y =incidence, marker=dict( color='blue', size=3),   name='Incidence'
    ),row=5,col=1)

    fig.add_trace(go.Scatter(mode='markers',x = time, y =azimuth,  marker=dict(  color='green',  size=3, line=dict(
        color='green',width=1 ) ), name='Azimuth' ),row=6,col=1)
    fig.add_trace(go.Scatter(mode='markers',x = time, y =rectilinearity,  marker=dict(  color='green',  size=3, line=dict(
        color='green',width=1 ) ), name='rectilinearity' ),row=7,col=1)
    fig.add_trace(go.Scatter(mode='markers',x = time, y =planarity,  marker=dict(  color='green',  size=3, line=dict(
        color='green',width=1 ) ), name='planarity' ),row=8,col=1)
    marker='circle'
    marker1='circle-x'
    fig.add_trace(go.Scatter(mode='markers', marker_symbol=marker, x =time[min_list_tca]    ,
        y=min_list_tca_amp,marker=dict(color='red',
        size=10,line=dict(color='green', width=4)),name='Start' ),row=4       ,col=1)
    fig.add_trace(go.Scatter(mode='markers', marker_symbol=marker1, x =time[max_list_tca]    ,
        y=max_list_tca_amp,marker=dict(color='red',
        size=10,line=dict(color='red', width=4)),name='End'),row=4       ,col=1)
    fig.add_trace(go.Scatter(mode='markers', marker_symbol=marker, x =time[min_list_tca]    ,
        y=min_list_tca_amp,marker=dict(color='red',
        size=10,line=dict(color='green', width=4)),name='Start' ),row=1       ,col=1)
    fig.add_trace(go.Scatter(mode='markers', marker_symbol=marker1, x =time[max_list_tca]    ,
        y=max_list_tca_amp,marker=dict(color='red',
        size=10,line=dict(color='red', width=4)),name='End'),row=1       ,col=1)
    #calling
    fig.add_trace(go.Scatter(mode='markers', marker_symbol=marker, x =ts_pick    ,
        y=max_comp_amp,marker=dict(color='orange',
        size=10,line=dict(color='orange', width=4)), name=f'Prominence'),row=1       ,col=1)
    fig.add_trace(go.Scatter(mode='markers', marker_symbol=marker, x =ts_pick    ,
        y=max_comp_amp,marker=dict(color='orange',
        size=10,line=dict(color='orange', width=4)),name=f'Prominence' ),row=4       ,col=1)

    matr = np.maximum(np.maximum(stz,stn), ste)
    mitr = np.minimum(np.minimum(stz,stn), ste)

    fig.update_xaxes( range=[min(date),max(date)],tickfont=dict(size=16),title_text = "Time",row=6,col=1)
    fig.update_yaxes( range=[min(mitr),max(matr)],tickfont=dict(size=16),row=2,col=1)
    fig.update_yaxes( range=[min(mitr),max(matr)],tickfont=dict(size=16),row=3,col=1)
    fig.update_yaxes( range=[min(mitr),max(matr)],tickfont=dict(size=16),row=4,col=1)
    # fig.update_yaxes( range=[0,1],tickfont=dict(size=16),row=7,col=1)

    fig.update_yaxes( range=[0,180],title_text = 'Angle°',title_font=dict(size=26),row=6,col=1)
    fig.update_yaxes( range=[0,90],title_text = 'Angle°',title_font=dict(size=26),row=5,col=1)

    fig.update_layout(    width=1800,      # Imposta la larghezza della finestra
        height=1200, 
        paper_bgcolor="white",
        title_text=f'{StaName}',
        title_font=dict(size=38),                   # Titolo ingrandito
        legend=dict(font=dict(size=20)),            # Testo della legenda ingrandito
    )
    fig.write_html(f'{pwd}/{name_test}.html')
    print(f'html provided for attribute pol plot {pwd}/{name_test}.html')
    return

