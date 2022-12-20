import glob
import os
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import access2thematrix
import numpy as np



### The examples of file names.
# 2021Nov17-104210_STM-STM_Spectroscopy--3_1.Aux2_mtrx.gwy
# 2021May08-134535_STM-STM_Spectroscopy--11_2.Z_mtrx.gwy

#      IDQ    Optical                         solid angle
etta = 0.8 * (0.99 * 0.9 * 0.99 * 0.99 * 0.92 * 0.92) * (1 / 18) # light collection efficiency
e_per_nA = 6241509074460760000 / 1e9 # electron per 1 nA
magic = (etta * e_per_nA)

def QEbyMap(ed,NNN,DACmax=None,direct='forward/up'):
    """Calc STM-LE QE map corresponded to scan
    ed - day of the experiment,
    NNN - run number of intrested scan
    DACmax - correspond to maximum photons count when DAC give maximum (5V) on Aux
    direct - direction of scan.  """
    # ed = '2022-10-12'
    project_path = 'C:\\...\\STM-Electroluminescence\\'
    DataDirIn = project_path + 'ExpData\\Omicron results\\%s\\' % ed
    DataDirOut = project_path + 'WorkWithData\\2022-10-12 Calc efficiency\\'
    Pdir = project_path + 'ExpData\\Omicron results\\PulseGrabber\\' # Directory with pulses log

    def GetFileList(ed, start, stop, type='I(V)'):
        # type can by 'I(V)' or 'Aux2(V)' for example
        intrestinglist = []
        for i in range(start, stop + 1):
            intrestinglist += ['*--%d_*.%s_mtrx' % (i, type)]
        filelist = []
        for m in intrestinglist:
            filelist += glob.glob(DataDirIn + m)
        return (filelist)

    # Step 1 ##


    filelist = GetFileList(ed, NNN, NNN, type='Aux2')
    mtrx_data = access2thematrix.MtrxData()
    traces_im, message = mtrx_data.open(filelist[0])
    im, message = mtrx_data.select_image(direct)
    N = mtrx_data.scan.shape[1]  # обращаюсь к числу задонных строк/столбцов (мог перепутать размерность)
    Auxmap = im.data[::-1]

    filelist = GetFileList(ed, NNN, NNN, type='I')
    traces_im, message = mtrx_data.open(filelist[0])
    im, message = mtrx_data.select_image(direct)
    ImapRaw = im.data[::-1]

    if not DACmax:
        # Todo: FIX MapTime. It must be like CurveTime
        MapTime = IVdata_info['param']['BKLT']
        Aparams = GetAparams(MapTime, Pdir)
        DACmax = Aparams[1] - Aparams[0]

    Auxmap /= 4.98
    Auxmap *= DACmax
    # Auxmap += 150

    from gwyfile.objects import GwyContainer, GwyDataField
    obj = GwyContainer()
    ImapRaw *= 1e9
    shift_value = findTdelta(ImapRaw, Auxmap)
    Imap = np.where(ImapRaw < 0.1, 1e10, ImapRaw)  # if curren below 0.1, set big value for QE to 0

    ImapSh = shift5(Imap.T, shift_value,fill_value=1e10).T # необходимо сдвинуть на одну точку
    AuxNorm = Auxmap / ImapSh
    AuxNorm /= etta
    AuxNorm /= e_per_nA

    obj['/0/data/title'] = 'Aux with'
    data = AuxNorm
    obj['/0/data'] = GwyDataField(data, xreal=im.width, yreal=im.height, si_unit_xy='m')

    dn = ''.join([x[0] for x in direct.split('/')])
    obj.tofile(DataDirOut + "AuxNorm_%s_%s_%s_shift.gwy" % (ed, NNN, dn))
    print(1)
    if True:
        lineN = 4
        filelist = GetFileList(ed, NNN, NNN, type='Z')
        traces_im, message = mtrx_data.open(filelist[0])
        im, message = mtrx_data.select_image(direct)
        Z = im.data[::-1] * 1e9

        fig, (axup, ax, axb) = plt.subplots(
            nrows=3, ncols=1,
            figsize=(8, 4)
        )

        axup.plot(Z[lineN])
        axup.set_ylabel('Height, nm')

        ImapRawSh = shift5(ImapRaw.T, shift_value,fill_value=0).T
        ax.plot(ImapRawSh[lineN])
        ax.set_ylabel('Current, nA', color='blue')

        ax2 = ax.twinx()
        ax2.plot(Auxmap[lineN], color='red')
        ax2.set_ylabel('Intensity, Hz', color='red')

        axb.plot(AuxNorm[lineN], color='green')
        axb.set_ylabel('QE, ph./el.')
        axb.set_ylim(0, 1.2e-5)
        plt.show()
        print(1)


def shift5(arr, num, fill_value=np.nan):
    """Be prevented. If you use DAC such as mine, you may have a little delay between Z,I... and Aux-signal.
It was provided by time, which need for collecting of pulses from photon counter and give it for Omicron CU.
Usually, i use DAC's integration equal 10ms, and T_rast = 1...10 ms.
It provide zero-like filling on the edge of data array. If you are nerd, you can found data from scan with another scan dirrection.
"""
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def GetAparams(CurveTime, Pdir):
    """Try to find the correspondig pulse log, and read DAC params"""
    # Получить параметры для ЦАП
    # Найти файл от ЦАП соответствующий времени (ближайший по времени до)

    from datetime import datetime
    CT = CurveTime.split(', ')[1]
    CT = datetime.strptime(CT, '%d %B %Y %H:%M:%S')
    # if CT < datetime.fromisoformat('2022-02-22 13:11:00'): # Use it, if you fix time delta between matrix PC and couner PC
    #     from datetime import timedelta
    #     d = timedelta(minutes=19, seconds=2)
    #     CT -= d
    # get file list for this data
    pattern = CT.strftime('%Y-%m-%d*.gz')
    filelist = glob.glob(Pdir + pattern)
    # sort by time
    # Get file with maximal time before CurveTime
    if not filelist:
        return False
    else:
        f2o = False
        for f in filelist:
            ftime = f.split('\\')[-1][:-7]
            ftime = datetime.strptime(ftime, '%Y-%m-%d_%H-%M-%S')
            if ftime < CT:
                f2o = f
            else:
                break
        if not f2o:
            return False
    # unizp
    import gzip
    with gzip.open(f2o, 'rb') as f:
        file_content_h = f.readline()
        file_content_p = f.readline().decode()
        # check: curve data in file
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
    last_line_time = datetime.fromisoformat(last_line.split(' ')[0])
    if CT > last_line_time:
        print('log started before curve, but curve not in it')

    # encode params
    params = file_content_p[:-2].split(' ')
    params = list(map(int, params))
    return params




def CurveQEA():
    import StepByStepСVCAnalize as ML

    project_path = 'C:\\......\\GlobalProjects\\STM-Electroluminescence\\'
    Pdir = project_path + 'ExpData\\Omicron results\\PulseGrabber\\'
    Sample = 'Gold' # Name of dataset
    addpat = ''

    if Sample == 'Si220221':        ### Datasets
        ed = '2021-11-17'           # Date
        Amax = 3600 - 600 - 120     # DACs params
                                    # Dir's initialization
        DataDirIn = project_path + 'ExpData\\Omicron results\\%s\\' % ed
        DataDirOut = project_path + 'WorkWithData\\2022-03-09 Calc efficiency\\pimple\\'

        slist = [1, 2, 3]           # Run cycle of intrested I(V)s
    elif Sample == 'Gold':
        ed = '2022-02-18'
        # Amax = 3600 - 600 - 120
        DataDirIn = project_path + 'ExpData\\Omicron results\\%s\\' % ed
        slist = [7, 12, 13, 14, 15, 16, 19, 20]

    DataDirOut = project_path + 'WorkWithData\\2022-03-09 Calc efficiency\\%s\\' % Sample
    QEf = {}
    res = {}

    for j in slist:
        N = j
        pat = '*%s*--%d_*.%s_mtrx' % (addpat, N, 'Aux2(V)')
        filelist = glob.glob(DataDirIn + pat)

        AuxVdata0, IVdata_info = ML.GetDataAsDataset(filelist)
        AuxVdata0 = AuxVdata0.set_index('V')

        AuxVdata0 = AuxVdata0.sort_index()
        AuxVdata0.set_axis(['0', '1'], axis=1, inplace=True)

        CurveTime = IVdata_info['param']['BKLT']
        Aparams = GetAparams(CurveTime, Pdir)
        Amax = Aparams[1] - Aparams[0]
        AuxVdata = AuxVdata0 * Amax / 4.2

        filelist = ML.GetFileList(ed, N, N, DataDirIn, type='I(V)')
        IVdata, IVdata_info = ML.GetDataAsDataset(filelist)
        IVdata = IVdata.set_index('V')
        IVdata = IVdata.sort_index()
        IVdata.set_axis(['0', '1'], axis=1, inplace=True)
        IVdata *= 1e9

        for i in ['0', '1']:
            QEData = pd.concat((IVdata[i], AuxVdata0[i]), axis=1)
            QEData.set_axis(['I', 'Aux', ], axis=1, inplace=True)
            QEData['Intensity'] = QEData['Aux'] * Amax / 4.2
            QEData['IntNorm'] = QEData['Intensity'] / QEData['I'].abs()
            QEData['QE'] = QEData['IntNorm'] / magic
            res['%s%s' % (j, i)] = QEData

            Ul, Uh = 1.23, 3
            # Filtering by U. sly-filtering for separate worth data from garbage
            QEDataf = pd.concat([
                QEData[-Uh:-Ul],
                QEData[Ul:Uh]
            ])
            Ifl = pd.concat([
                QEData['I'][-Uh:-Ul] < -0.01,
                QEData['I'][Ul:Uh] > 0.01,
            ])
            Ifh = pd.concat([
                QEData['I'][-Uh:-Ul] > -200,
                QEData['I'][Ul:Uh] < 200,
            ])

            QEf['%s%s' % (j, i)] = QEDataf[Ifl]
            # print(QEDataf[Ifl][Ifh]['QE'].abs().describe())
            # QEDataf[Ifl][Ifh][['I', 'QE']].plot.scatter(x='I', y='QE')

            # V = QEDataf[Ifl][Ifh][['QE']].index
            # I = QEDataf[Ifl][Ifh][['I']].values
            # QE = QEDataf[Ifl][Ifh][['QE']].values
            # ax1.scatter(x=V,y=QE,label='%s-%s' % (j,i))
            # ax2.scatter(x=I, y=QE,label='%s-%s' % (j,o))
            #
            # print('!',end='')

    fig, ax = plt.subplots(
        nrows=1, ncols=1,
        figsize=(8, 4)
    )
    for K in QEf.keys():
        QEDataf = QEf[K]
        V = QEDataf[['QE']].index
        I = QEDataf[['I']].values
        QE = QEDataf[['QE']].values
        ax.scatter(x=V, y=QE, label=K, marker='+')
    plt.xlabel('Voltage (V)')
    plt.ylabel('QE')
    plt.legend()
    plt.grid()
    plt.title(Sample)
    plt.savefig(DataDirOut + 'QEvsV.png')
    plt.ylim(0, 4e-6)
    plt.savefig(DataDirOut + 'QEvsV zoom.png')

    fig, ax = plt.subplots(
        nrows=1, ncols=1,
        figsize=(8, 4)
    )
    for K in QEf.keys():
        QEDataf = QEf[K]
        V = QEDataf[['QE']].index
        I = QEDataf[['I']].values
        QE = QEDataf[['QE']].values
        ax.scatter(x=I, y=QE, label=K, marker='+')
    plt.xlabel('Current (nA)')
    plt.ylabel('QE')
    # plt.legend()
    plt.grid()
    x = np.linspace(-10, 10, 1000)
    y = Amax / x
    y = np.abs(Amax / (x * magic))
    yl = ax.get_ylim()
    # plt.plot(x, y,'--',label='DAC overload')
    plt.ylim(yl)
    plt.legend()
    plt.title(Sample)
    plt.savefig(DataDirOut + 'QEvsI.png')
    plt.xlim((-10, 10))
    plt.savefig(DataDirOut + 'QEvsI zoom.png')
    plt.xlim((-3, 3))
    plt.ylim(0, 1e-4)
    plt.savefig(DataDirOut + 'QEvsI zoom2.png')

    DataDirOut += 'data\\'
    for K in QEf.keys():
        QEf[K].to_csv(DataDirOut + 'QE filtred for %s.csv' % (K))
        res[K].to_csv(DataDirOut + 'QE raw for %s.csv' % (K))

    for K in QEf.keys():
        l = res[K]['I'][0:3][res[K]['I'][0:3] < 0].__len__()
        h = res[K]['I'][-3:0][res[K]['I'][-3:0] > 0].__len__()
        print(K, l, h)


def findTdelta(Imap,Auxmap,g=15):
    from scipy import signal, fftpack
    A = fftpack.fft(Imap[g])
    B = fftpack.fft(Auxmap[g])
    Ar = -A.conjugate()
    Br = -B.conjugate()
    sh = np.argmax(np.abs(fftpack.ifft(Ar * B)))
    # print(sh)
    # print(np.argmax(np.abs(fftpack.ifft(A * Br))))
    return sh

if __name__ == '__main__':
    # QEbyMap(ed='2022-10-06', NNN=16)
    QEbyMap(ed='2022-10-20', NNN=8)
    # QEbyMap(ed='2022-10-20', NNN=8,direct='backward/up')
    # QEbyMap(ed='2022-10-20', NNN=4)
