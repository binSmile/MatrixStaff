##################################################################
#
#           StepByStepVacAnalize
#
##################################################################

import os
import pickle
import glob
from statsmodels.nonparametric.kernel_regression import KernelReg
import matplotlib.pyplot as plt

import access2thematrix
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def initdir(ed):
    ## Dirs initialization
    project_path = 'C:\\...\\Project name\\'
    DataDirIn = project_path + 'ExpData\\Omicron results\\%s\\' % ed
    DataDirOut = project_path + 'WorkWithData\\2022-02-03 IVworking\\'
    return DataDirIn, DataDirOut

def dostdfilter(IVdata, ser,nsigma):
    ''' IV filter. Check garbage or good IV. Based on std.dev.
    If point in the current IV in nsigma - it is ok'''
    ss = ser.std()*nsigma
    sm = ser.mean()
    fordrop = np.append(ser[ser > sm + ss].index,ser[ser < sm - ss].index)
    IVdata = IVdata.drop(fordrop, axis=1)

    return [IVdata,fordrop]


def calcR(IVdata):
    """Calculation of resistance on Current–voltage characteristic
    IVdata - is pd.Dataframe"""

    points = ['0','1']
    c=250
    variants = [[c-10,c+10],[c-20,c+20],[c-30,c+30],[c-40,c+40]]
    for v in variants:
        start, stop = v
        df = IVdata.loc[start:stop, ['V', *points]]
        x =df['V']
        pl = []
        for j in points:
            y = df[j]
            pl += [np.polyfit(x,y,1)]
        pl = pd.DataFrame(pl, index=points)
        (1 / pl[0]).plot(label='from %s to %s' % (start, stop))

        pl = pd.DataFrame(pl)
        R = (1 / pl[0])



def GetFileList(start, stop,DataDirIn,type='I(V)',addfilter=''):
    """ Give list with filelist with Run Number between start and stop, in DataDirIn. whith type I(V) or Aux(V) if you need"""
    intrestinglist = []
    for i in range(start, stop + 1):
        intrestinglist += ['*%s*--%d_*.%s_mtrx' % (addfilter,i,type)]
    filelist = []
    for m in intrestinglist:
        filelist += glob.glob(DataDirIn + m)
    return (filelist)

def GetDataAsDataset(filelist):
    """This function get list of files and transform it to pd.DataFrame"""""
    IVdata = pd.DataFrame()
    IVdata_info = {}

    for filename in filelist:
        Ptitle = filename.split('--')[-1][:-10]
        print(Ptitle)
        mtrx_data = access2thematrix.MtrxData()
        data_file = filename
        traces, message = mtrx_data.open(data_file)
        for i in traces.keys():
            trace, message = mtrx_data.select_curve(traces[i])
            if i == 0:
                IVdata['V']=trace.data[0]
            a = trace.data[1]
            l = mtrx_data.param['EEPA::Spectroscopy.Device_1_Points'][0]
            if len(a) != l:
                print(Ptitle, 'i', ' is broken. %i from %i points' % (len(a), l))
                z = np.zeros(l)
                z[0:a.shape[0]] = a
                a = z
            IVdata[Ptitle + '_' + str(i)] = a
        IVdata_info[Ptitle] = trace.referenced_by
        IVdata_info['param'] = mtrx_data.param

    return IVdata, IVdata_info

def testshow(iv):
    print('Max: %f.0' % iv.max())
    iv.plot(x='V', y=i)

def Do_didv(IVdata):
    """It calc deriative dI/dV for each CV in dataframe"""
    d = abs(np.diff(IVdata.index)[5])
    didv = pd.DataFrame(index=IVdata.index)
    # didv['V'] = IVdata['V']

    for iv in IVdata:
        didv[iv] = np.gradient(IVdata[iv], d)
    return didv

def Do_dvdlnj(IVdata):
    """It calc the vague deriative dV/d ln(j)"""
    d = abs(np.diff(IVdata.index)[5])
    for iii in ['1','0']:
        IVdata['absI'] = abs(IVdata[iii])
        IVdata['absI'] /= 7.853981633974483e-19
        IVdata['lnI']=np.log(IVdata['absI'])

        side = 'p'
        if side == 'p':
            PP = IVdata.iloc[:150].copy()
            dln0 = np.diff(IVdata['lnI'], append=0)
            dV = np.diff(IVdata.index.values, append=0)
        elif side == 'n':
            PP = IVdata.iloc[150:].copy()
            dln0 = np.diff(PP['lnI'], prepend=0)
            dV = np.diff(PP.index.values, prepend=0)

        PP['dvdlnI'] = np.gradient(dV, dln0)

        y = PP['lnI']
        x = np.linspace(PP.idxmax()[0], PP.idxmin()[0], len(y))
        kr = KernelReg(y, x, 'c')
        plt.plot(x, y, '+')
        y_pred, y_std = kr.fit(x)

        # plt.plot(x, y_pred)
        # plt.show()
        PP['lnIsmooth'] = y_pred
        PP['Iexp'] = np.exp(y_pred)
        PP['Vsmooth'] = x
        PP['dlnIsmooth'] = np.diff(PP['lnIsmooth'], prepend=0)
        PP['Vsmooth'] = np.diff(PP['VIsmooth'], prepend=0)
        PP['dv/dlni'] = PP['Vsmooth'] / PP['dlnIsmooth']
        # PP[['absI', 'dv/dlni']][:-1].plot(x='absI')

        plt.plot(PP['absI'][100:220], savgol_filter(PP['dv/dlni'], 31, 2)[100:220])
        plt.yscale('log')
        plt.xscale('log')
        # x=V, y=lnI

        from scipy.optimize import curve_fit
        def powerlaw(x, a, b):
            return a * np.power(x, b)
        #  y = a * x^b

        df = PP['absI','dv/dlni'][100:220].copy()
        popt, pcov = curve_fit(powerlaw, df['absI'].values,df['dv/dlni'])
        # plt.plot(xdata, powerlaw(xdata, *popt), '-',
        #      label='fit %s mW: a=%.2E, b=%5.3f' % (column, popt[0],popt[1]))
        # plt.scatter(xdata, ydata, s=30)
        # fitA[column] = popt[1]
        # print(column)


    return didv



def SmoothIVs(IVdata,sgi):
    """It smooth every column of Dataframe with CV by Savitzky-Golay filter. sgi - smooth frame, must be odd """
    SIV = pd.DataFrame(index=IVdata.index)
    for iv in IVdata:
        SIV[iv] = savgol_filter(IVdata[iv], sgi, 2)
    return SIV

def MergeIVs(IVdata,start,stop,toone=False):
    """It merges set of CV by average
    Input: dataframe with columns ..32_1,32_2,32_3,33_1,33_2,33_3...
    (where first number is run cycle, second - repetition)
    Output: dataframe,
    - if toone=False, return columns ...32,33...
        merge repetitions only
    - if tiine=True, return one column.
        merge repetitions and runcycles
    """
    MIV = pd.DataFrame()
    if toone:
        #tmp = IVdata.drop(['V'],axis=1)
        MIV['I'] = IVdata.sum(axis=1) / IVdata.shape[1]
    else:
        for j in range(start,stop+1):
            tmp = IVdata.filter(like=str(j),axis=1)
            MIV[j] = tmp.sum(axis=1)/tmp.shape[1]

    return MIV


def FingerPrintPlot(df):
    """Fingerprint map plot"""
    X = df.columns.values
    Y = df.index.values
    Z = df.values
    x,y=np.meshgrid(X, Y)
    plt.contourf(x, y, Z)


def ShowMap(IVdata_info, start,stop,fullrange=False):
    """ Shows a scan with points, where was measured CVC
    if fullrange=True, that means draw all points from IVData_info. Not from range start-stop.
    """
    t = list(IVdata_info.keys())[0]
    RefMap = DataDirIn + IVdata_info[t]['Data File Name']
    mtrx_data = access2thematrix.MtrxData()
    traces_im, message = mtrx_data.open(RefMap)
    im, message = mtrx_data.select_image(IVdata_info[t]['Trace'][1])
    N = mtrx_data.scan.shape[1]  # Maybe i confused
    fig, ax = plt.subplots(
        nrows=1, ncols=1,
        figsize=(8, 4)
    )
    ax.imshow(im.data[::-1])

    if fullrange:
        k = list(int(e.split('_')[0]) for e in IVdata_info.keys())
        start, stop = min(k), max(k)

    for j in range(start,stop+1):
        p = IVdata_info['%d_1' % j]
        [x, y] = p['Location (px)']
        plt.text(x, N - y+4, j+1-start,fontsize=6,c='w')
        ax.scatter(x, N - y, c='r')


    plt.title('Map from %d to %d' % (start, stop))
    fig.tight_layout()
    plt.savefig(DataDirOut + '%s_%d_%d_map.png' % (ed, start, stop), bbox_inches='tight')


def Calcd2idv2map():
    """ Read Data and shows CVCs filtred  data by 2sigma"""
# if __name__ == '__main__':
    ## Init ##
    ed = '2021-11-22'  # Date of experiment (it is need for dir initialisation)
    start, stop = 49,88  # range of CVC
    DataDirIn, DataDirOut = initdir(ed)
    sgi = 41 # Savitzky- Golay filter frame width
    NewData = True # Set True for new data. If changed some processing paramentras - set False. It will be faster, because i use picke for cache.

    if NewData:
        ftype = 'I(V)'
        # Step 1 ##
        print('Read new data %s for points from %d to %d' % (ed,start,stop))
        print('Work with: %s' % ftype)
        filelist = GetFileList(ed,start,stop,type=ftype)
        IVdata, IVdata_info = GetDataAsDataset(filelist)
        ## Exception
        for IVd_i in IVdata_info:
            if IVdata_info[IVd_i]['Data File Name'] == '2021Nov22-120542_STM-STM_Spectroscopy--6_2.Z_mtrx':
                IVdata_info[IVd_i]['Data File Name'] = '2021Nov22-120542_STM-STM_Spectroscopy--6_1.Z_mtrx'

        #### save checkpoint
        IVdata.to_csv(DataDirOut+'%s_%d_%d.csv' % (ed,start,stop),index=False)
        with open(DataDirOut+'%s_info_%d_%d.pickle' % (ed,start,stop), 'wb') as f:
               pickle.dump(IVdata_info, f)
    else:
        ## Step 2 ##
        IVdata = pd.read_csv(DataDirOut + '%s_%d_%d.csv' % (ed, start, stop))
        with open(DataDirOut+'%s_info_%d_%d.pickle' % (ed,start,stop), 'rb') as f:
            IVdata_info = pickle.load(f)

    # ManualCheckIV(IVdata)
    IVdata = SmoothIVs(IVdata, sgi)
    IVdata = IVdata.set_index('V')


    # Vshift = -0.6192372368232928
    # IVdatafm = IVdata.set_index(IVdata.index - Vshift)

#####################
# This is intresting filter by sigma. You can use breakpoints and to choose the best settings.
#####################

    IVdataf = IVdata.copy()
    IVdataf, fordrop1 = dostdfilter(IVdataf,IVdataf[:][-0.15:-0.1].mean(),1)
    IVdataf, fordrop2 = dostdfilter(IVdataf, IVdataf[:][0.1:0.15].mean(),1)
    IVdataf, fordrop3 = dostdfilter(IVdataf, IVdataf[:][-0.15:-0.1].mean(),2)
    IVdataf, fordrop4 = dostdfilter(IVdataf, IVdataf[:][0.1:0.15].mean(),2)
    IVdataf, fordrop5 = dostdfilter(IVdataf, IVdataf[:][0.1:0.15].mean(), 2)



    IVdatafm = MergeIVs(IVdataf, start, stop)
    c0 = int(IVdatafm.shape[0] / 2)
    Ishift = IVdatafm.iloc[c0 - 1:c0 + 1, :].mean().mean()
    IVdatafm -= Ishift ## USE it, because you can have I != 0, when V == 0.

    IVdatafm.to_csv(DataDirOut + '%s_smooth_%d_%d.csv' % (ed, start, stop))
    didv = Do_didv(IVdatafm)
    # didv.to_csv(DataDirOut + '%s_%d_%d_didv.csv' % (ed, start, stop))

    didv = SmoothIVs(didv, 11)
    d2idv2 = Do_didv(didv)


    d2idv2.set_axis([i for i in range(1, 41)], axis=1, inplace=True)
    d2idv2.to_csv(DataDirOut + '%s_%d_%d_d2idv2.csv' % (ed, start, stop))

    # FingerPrintPlot(d2idv2)
    print('Done')
    #
    # IVdata = IVdata.drop([123],axis=0)
    # calcR(IVdata)


if __name__ == '__main__':
    ## Init ##
    ed = '2022-04-13'  # Exp data. it need for datadir initialization
    start, stop = 14,14  # range of run cycles of CVC
    DataDirIn, DataDirOut = initdir(ed)
    sgi = 41 # Slovitsky-golay frame height
    NewData = False ## False for first read, and True - for picke cache.

    if NewData:  ##
        ftype = 'I(V)'
        # Step 1 ##
        print('Read new data %s for points from %d to %d' % (ed,start,stop))
        print('Work with: %s' % ftype)
        filelist = GetFileList(start,stop,DataDirIn,type=ftype)
        IVdata, IVdata_info = GetDataAsDataset(filelist)

        #### save checkpoint
        IVdata.to_csv(DataDirOut+'%s_%d_%d.csv' % (ed,start,stop),index=False)
        with open(DataDirOut+'%s_info_%d_%d.pickle' % (ed,start,stop), 'wb') as f:
               pickle.dump(IVdata_info, f)
    else:
        ## Step 2 ##
        IVdata = pd.read_csv(DataDirOut + '%s_%d_%d.csv' % (ed, start, stop))
        with open(DataDirOut+'%s_info_%d_%d.pickle' % (ed,start,stop), 'rb') as f:
            IVdata_info = pickle.load(f)

    # занимаюсь сопоставлением данных с Aux2(V) с данными с IDQ сделанных копипастом
    IVdata = IVdata.set_index('V')
    # ВАХ33_1 записана от +10 до -10 до +10

    print('Hello')

    IVdataf = IVdata.copy()
    sv = 2
    dv = 0.25
    IVdataf, fordrop1 = dostdfilter(IVdataf, IVdataf[:][-sv - dv:-sv + dv].mean(), 1)
    IVdataf, fordrop2 = dostdfilter(IVdataf, IVdataf[:][sv - dv:sv + dv].mean(), 1)
    IVdatafm = MergeIVs(IVdataf, start, stop)
    c0 = int(IVdatafm.shape[0] / 2)
    Ishift = IVdatafm.iloc[c0 - 1:c0 + 1, :].mean().mean()
    IVdatafm -= Ishift

    DataDirOut = 'C:\\....\\processing\\'

