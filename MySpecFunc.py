#!/usr/bin/env python3
import glob
import access2thematrix
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
import csv
import os
from fnmatch import fnmatch
import  spc

import zipfile
import tarfile

settings = {
'delHotPeaks':1,    # try to delete spikes by hot-pixels
'bdir': 'C:\\...\\Specs\\',
'evmes':0,          # transform Inetsity vs nm, to vs eV
'norm':0,           # normalize to 1
'convolve':0,       # convolve smotth
'n':20,              # convolve smooth width
'useDark':0,        # deduct dark spectrum
'useConstantDark':0, # or use constant value of dark level
'ConstantDark':0,   # dark level
'Fitting':False,    # do you need in gaussian fitting
'useQE':0,          # using of Quantum Efficiency of Andour CCD
'useHF':1,          # using of Hardware efficiency function (it include Andour QE)
'useTIME':0,
'prefix0':'',       # prefix in save files
'title':'',         # tile for save file


}

def GetFolderStat(DataDirIn,DataDirOut,ed):
    filelist = []
    # for m in title:
    filelist += glob.glob(DataDirIn + '*_mtrx')
    filelist.sort()
    Data = [['S','Num','Chanel','SetName', 'channel_name', 'min','max',
                      'voltage,V',
             # 'voltage_unit',
                      'EEPA::Regulator.Setpoint_1,nA',
                      # 'EEPA::Regulator.Setpoint_1 unit',
                      'EEPA::Spectroscopy.Device_1_Points',
                      'EEPA::Spectroscopy.Enable_Feedback_Loop',
                      'BKLT',
                      'Angle',
                      'axis',
                      'Raster time (ms)',
                      'Points',
                      'lines',
             'Height, nm',
             'Width,nm'
                        ]]
    for filename in filelist:
        # print(filename)
        Ptitle = filename.split('--')[-1][:-5].split('.')
        S = int(Ptitle[0].split('_')[0])
        Num = int(Ptitle[0].split('_')[1])
        chan = Ptitle[1]
        try:
            mtrx_data = access2thematrix.MtrxData()
            data_file = filename
            traces, message = mtrx_data.open(data_file)
            # print(message)
            trace, message = mtrx_data.select_curve(traces[0])
        except:
            Data += [[filename, '', '', 'Open failed']]

        if mtrx_data.channel_name == 'I(V)':
            try:
                min, max =  trace.data.min(), trace.data.max()
                ref = str(mtrx_data.ref_by)
            except:
                print(filename)
        else:
            min, max = '',''
            ref = str(mtrx_data.axis)

        time = datetime.strptime(mtrx_data.param['BKLT'],'%A, %d %B %Y %H:%M:%S')
        timef = time.isoformat()
        Data += [[S,Num, chan,
                   mtrx_data.data_set_name, mtrx_data.channel_name, min, max,
                  mtrx_data.param['EEPA::GapVoltageControl.Voltage'][0],
                   # mtrx_data.param['EEPA::GapVoltageControl.Voltage'][1],
                  mtrx_data.param['EEPA::Regulator.Setpoint_1'][0]*1e9,
                  # mtrx_data.param['EEPA::Regulator.Setpoint_1'][1],
                  mtrx_data.param['EEPA::Spectroscopy.Device_1_Points'][0],
                  mtrx_data.param['EEPA::Spectroscopy.Enable_Feedback_Loop'][0],
                  time,
                  mtrx_data.param['EEPA::XYScanner.Angle'][0],
                  ref,
                  mtrx_data.param['EEPA::XYScanner.Raster_Time'][0]*1e3,
                  mtrx_data.param['EEPA::XYScanner.Points'][0],
                  mtrx_data.param['EEPA::XYScanner.Lines'][0],
                  mtrx_data.param['EEPA::XYScanner.Height'][0]*1e9,
                  mtrx_data.param['EEPA::XYScanner.Width'][0]*1e9,

                  ]]



    import xlsxwriter
    workbook = xlsxwriter.Workbook(DataDirOut+'%s stat.xlsx' % ed)
    worksheet = workbook.add_worksheet()
    for row_num, line in enumerate(Data):
        worksheet.write_row(row_num, 0, line)

    format5 = workbook.add_format({'num_format': 'dd/mm/yy hh:mm'})
    worksheet.set_column(first_col=11,last_col=11,cell_format=format5)

    workbook.close()


def workWithSpcZip(zf):
    """unpack ZIP-file with spc-files, and store it in /tmp"""
    zippach = '/tmp/' + zf.split('/')[-1][:11] + '/'
    if zipfile.is_zipfile(zf):
        if not os.path.isdir(zippach):
            z = zipfile.ZipFile(zf, 'r')
            z.extractall('/tmp/')
    elif tarfile.is_tarfile(zf):
        if not os.path.isdir(zippach):
            z = tarfile.open(zf,'r:gz')
            z.extractall('/tmp/')
    else:
        print('ZIP Error')
        exit(1)
    return zippach

def createBaseDir(zf, workspace):
    p = workspace + zf.split('/')[-2]
    if not os.path.isdir(p):
        os.mkdir(workspace + zf.split('/')[-2])
    return p + '/'

def get_match_filenames(root, pattern):
    """search files with pattern in name"""
    for path, _, files in os.walk(root):
        for filename in files:
            full_name = os.path.join(path, filename)
            if fnmatch(full_name, pattern):
                yield full_name

def loadspc(fname,SpcFolder, doCorrect = 0):
    """reader of spc-spectrum
    it return list with wavelenght, intensity and dict of params"""
    f = spc.File(SpcFolder+fname)
    yy = f.sub[0].y
    if doCorrect:
        count = 0
        yCor = []
        tmp = list(f.sub[0].y)
        med = np.median(tmp)
        l = len(tmp)
        # for i in range(l):
        #     if tmp[i] <= med * 1.75:
        #         yCor += [tmp[i]]
        #     else:
        #         yCor += [med]
        #         count += 1
        for i in range(l):
            neighbors_c = 10
            el = neighbors_c if i >= neighbors_c else i
            er = neighbors_c if i <= (l-neighbors_c) else l - i
            testvalue = np.median(tmp[i-el:i+er])*1.1
            if tmp[i] <= testvalue:
                yCor += [tmp[i]]
            else:
                yCor += [testvalue]
                count += 1
        if count > 0:
            print(fname[-10:-4], 'replaced values: ', count)
        yy = np.array(yCor)
    return [f.x,yy,f.log_dict]

def HFF(x):
    return  (353.9278 / (np.pi * 449.8394 * (1 + (x - 994.9758)**2 / 449.8394**2))) + \
            (763.209 / (np.pi * 63.5401 * (1 + (x - 525.6296)**2 / 63.5401**2))) + \
            (np.sqrt(np.log(2) / np.pi) * (244.3813 / 75.2405) * np.exp(-np.log(2) * (x - 636.594)**2 / 75.2405**2)) + \
            (np.sqrt(np.log(2) / np.pi) * (52.9293 / 53.1659) * np.exp(-np.log(2) * (x - 732.9094)**2 / 53.1659**2)) + \
            (np.sqrt(np.log(2) / np.pi) * (1.2818 / 9.847) * np.exp(-np.log(2) * (x - 476.1229)**2 / 9.847**2)) + \
            (233.1373 / (np.pi * 49.6589 * (1 + (x - 412.5384)**2 / 49.6589**2))) + \
            (np.sqrt(np.log(2) / np.pi) * ((-33.2698) / 63.3418) * np.exp(-np.log(2) * (x - 1010.491)**2 / 63.3418**2))



def main():
    # flist = list(get_match_filenames(BaseDir, '*.csv'))
    #
    # UsefullList = []
    # for f in flist:
    #     i = int(f[-10:-4])
    #     if i >= 163725 and i <= 163732:
    #         UsefullList += [[' ',f[-21:],'Descr']]

    UsefullList = [

    ]





    # names =  ['name','xc','area','FWHM','sigma','Std.Dev.xc','Std.Dev.area','Std.Dev.sigma']
    # ApproxData = [
    #     names,
    # ]


    ax = plt.subplot(111)
    for s in UsefullList:
        n = 100
        TunSpecRaw = np.array(loadxy(BaseDir + s[1]))
        if TunSpecRaw[0][0] > 1500:
            print('spectra ' + s[1] + ' > 1500nm')
        if np.average(TunSpecRaw[1][0]) > 400:
            print('average intensity ' + s[1] + ' > 400')
        if np.average(TunSpecRaw[1][0]) < 50:
            print('average intensity ' + s[1] + ' < 50')
        else:

            if convolve:
                PlotSpec = np.array(np.convolve(TunSpecRaw[1], np.ones((n,)) / n, mode='valid'))
                xar = TunSpecRaw[0][int(n / 2):-int(n / 2) + 1]  # Возможно правильнее [int(n/2)-1:-int(n/2)]. Разница в 0,2 нм
            else:
                PlotSpec = np.array(TunSpecRaw[1])
                xar = TunSpecRaw[0]

            if useDark:
                DarkSpecRaw = np.array(loadxy(BaseDir + s[0]))
                if convolve:
                    DarkSpec = np.convolve(DarkSpecRaw[1], np.ones((n,)) / n, mode='valid')
                    PlotSpec -= DarkSpec
                else:
                    PlotSpec -= DarkSpecRaw[1]
            if useConstantDark:
                PlotSpec -= ConstantDark


            if useQE:
                QE = np.array(loadxy(QEfile))

                SpecMqe = []
                for i in range(len(PlotSpec)):
                    SpecMqe += [PlotSpec[i] / get_nearest_value_QE(QE, xar[i]) ]
                PlotSpec = SpecMqe

            if evmes:
                xar = 4.135667662 * (10 ** -15) * 299792458 * 10 ** 9 / xar
                PlotSpec *= 4.135667662 * (10 ** -15) * 299792458 * 10 ** 9 / xar ** 2
                PlotSpec /= 4.13*100 ######### Да бы абстрактная интенсивность была эквивалентна интенсивности от длин волн

                SpecMaxCentre = xar[np.argmax(PlotSpec)]
                SpecMaxCentrenm = ( 4.135667662*(10**(-15))*299792458*10 ** 9)/ SpecMaxCentre
                descr = str(int((SpecMaxCentre)*100)/100)+'/'+str(int(SpecMaxCentrenm))
            else:
                # descr = str(int(xar[np.argmax(PlotSpec)]))
                descr = ' '



            LineLabel = s[1][-7:-4] + ' ' + s[2] + ' (' + descr + ')'




            if norm:
                PlotSpec /= np.max(PlotSpec)

            line, = plt.plot(xar,
                    PlotSpec,
                            # s[3],# l,
                    label=LineLabel,
                    ) # Возможно правильнее [int(n/2)-1:-int(n/2)]. Разница в 0,2 нм

            # plt.plot(TunSpecRaw[0],TunSpecRaw[1],label=s[2])
            if Fitting:
                x_data = xar
                y_data = PlotSpec
                parameter, covariance_matrix = curve_fit(gauss_function, x_data, y_data)
                perr = np.sqrt(np.diag(covariance_matrix)) # Std.dev for a,x0,w0
                FWHM = np.sqrt(2*np.log(2))*parameter[2]

                x = np.linspace(min(x_data), max(x_data), 1000)
                plt.plot(x, gauss_function(x, *parameter), 'b-', label='fit',alpha=0.5)  # the star is to unpack the parameter array
                S = np.sqrt(np.sum((x_data-np.average(x_data))**2)/len(x_data))
                S0 = np.sqrt(len(x_data)*S**2/(len(x_data)-1))
                            #['xc',         'area',     'FWHM','sigma',   'Std.Dev.xc','Std.Dev.area','Std.Dev.sigma']
                ApproxData += [[
                    LineLabel,
                    str(s[2]),parameter[1],parameter[0],FWHM,parameter[2]/2,perr[1],perr[0],perr[2]]]

            prefix = prefix0 + '_' +s[1][-7:-4]+'_'
            if evmes:
                prefix += 'ev_'
            if norm:
                prefix += 'n_'

            # with open(OutDir+prefix+s[2]+'.csv', 'w', newline='') as csvfile:
            #     spamwriter = csv.writer(csvfile, delimiter='\t',
            #                             quotechar='|', quoting=csv.QUOTE_MINIMAL)
            #     spamwriter.writerows(
            #         np.array([xar,PlotSpec]).T
            #     )


    if Fitting:
        print(ApproxData)

        with open(OutDir +  'Approx.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter='\t',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerows(
                np.array(ApproxData)
            )
    # fig = plt.figure(1)
    # ax = fig.add_subplot(111)
    # handles, labels = ax.get_legend_handles_labels()

    # plt.xlim(550,950)
    # plt.ylim(0,1)

    # plt.legend()

    # # Put a legend to the right of the current axis
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))



    plt.ylabel('Интенсивность, у.е.')
    if evmes:
        plt.xlabel('Energy, ev')
        ax2, ax3 = ax.twinx(), ax.twiny()
    # ax2.set_ylim((0,1))
    plt.xlabel('Длина волны, нм')

    if evmes:
        l1 = ( 4.135*(10**(-15))*299792458*1000000000)/ax.get_xlim()[0]
        l2 = ( 4.135*(10**(-15))*299792458*1000000000)/ax.get_xlim()[1]
        ax3.set_xlim(l1,l2)


    plt.suptitle(title)
    # plt.title('')
    # plt.tight_layout(rect=[0,0,1,0.95])

    plt.savefig(OutDir+prefix+title+'.png',dpi=400)

    # plt.tight_layout()я
    plt.show()
    # # # 

    # print('Approx data')
    # for i in [0,1,2]:
    #     idata = np.array(ApproxData).T[i]
    #     lmax = np.max(idata)
    #     idata /= lmax
        # plt.plot(idata,label=names[i+1])
    # plt.legend()
    # plt.show()

    # Ed = np.array(ApproxData).T
    # plt.plot(Ed[0], Ed[1], 'o', yerr=Ed[5])  # Ox:BV, Oy:Xc
    # plt.errorbar(Ed[0], Ed[1], yerr=Ed[5],fmt='o')  # Ox:BV, Oy:Xc
    # plt.ylabel('Напряжение смещения, В')
    # plt.xlabel('Xc, положение максимума спектральной линии, эВ')
    # plt.savefig(OutDir+prefix+'Xc_vs_BV.png',dpi=300)

    # print('Hello')

    plt.show()

def AndourQE(x):
    import math
    y =  (math.sqrt(math.log(2) / math.pi) * (7762.0764 / 83.2022) * math.exp(-math.log(2) * (x - 551.427) ** 2 / 83.2022 ** 2)) + \
    (math.sqrt(math.log(2) / math.pi) * (11660.7024 / 99.2902) * math.exp(-math.log(2) * (x - 755.6066) ** 2 / 99.2902 ** 2)) + \
    (math.sqrt(math.log(2) / math.pi) * (381.7162 / 32.2671) * math.exp(-math.log(2) * (x - 633.9992) ** 2 / 32.2671 ** 2)) + \
    (math.sqrt(math.log(2) / math.pi) * (134.378 / 19.8452) * math.exp(-math.log(2) * (x - 672.2374) ** 2 / 19.8452 ** 2)) + \
    (math.sqrt(math.log(2) / math.pi) * (818.1648 / 43.4927) * math.exp(-math.log(2) * (x - 893.8167) ** 2 / 43.4927 ** 2)) + \
    (math.sqrt(math.log(2) / math.pi) * (488.0161 / 44.0818) * math.exp(-math.log(2) * (x - 982.3676) ** 2 / 44.0818 ** 2)) + \
    (math.sqrt(math.log(2) / math.pi) * (10.6847 / 6.9719) * math.exp(-math.log(2) * (x - 925.4494) ** 2 / 6.9719 ** 2)) + \
    (math.sqrt(math.log(2) / math.pi) * (127.9523 / 10.5653) * math.exp(-math.log(2) * (x - 397.63) ** 2 / 10.5653 ** 2)) + \
    (math.sqrt(math.log(2) / math.pi) * (1773.1797 / 27.0118) * math.exp(-math.log(2) * (x - 345.2432) ** 2 / 27.0118 ** 2)) + \
    (math.sqrt(math.log(2) / math.pi) * (949.0042 / 18.665) * math.exp(-math.log(2) * (x - 298.2513) ** 2 / 18.665 ** 2)) + \
    (math.sqrt(math.log(2) / math.pi) * (126.9464 / 9.6252) * math.exp(-math.log(2) * (x - 251.3138) ** 2 / 9.6252 ** 2)) + \
    (math.sqrt(math.log(2) / math.pi) * (1743.6685 / 37.8297) * math.exp(-math.log(2) * (x - 239.6065) ** 2 / 37.8297 ** 2)) + \
    (math.sqrt(math.log(2) / math.pi) * (1112.2431 / 31.5082) * math.exp(-math.log(2) * (x - 416.9155) ** 2 / 31.5082 ** 2))
    return y/100





def gauss_function(x, a0, xc0, w0):
    """ Gaussian broadened spectral line function
    x - shift
    a0 - area
    xc0 - centre
    w0 - width"""
    return a0 * ((w0 * np.sqrt(np.pi / 2)) ** -1) * np.exp(-2 * ((x - xc0) ** 2) * (w0 ** -2))


def TrC(lamp, sample, dark, descr, dash='-', n=20, save=True):
    """
    Spectra processing
    :param lamp: lamp spectra
    :param sample: sample spectra
    :param dark: dark spectra
    :param descr: just description
    :param dash: style on plot
    :param n: average frame for smoth
    :param save: save or not
    :return: None
    """

    # (['lamp path'],['sample path'],int:dark lmesel, label)
    LampSpecAvg = np.array(loadxy(lamp[0]))
    for i in range(1, len(lamp)):
        LampSpecAvg += np.array(loadxy(lamp[i]))
    LampSpecAvg /= len(lamp)
    LampSpecAvg[1] = np.convolve(LampSpecAvg[1] - dark, np.ones((n,)) / n, mode='same')

    Spec = np.array(loadxy(sample[0]))
    for i in range(1, len(sample)):
        Spec += np.array(loadxy(sample[i]))
    Spec /= len(sample)
    Spec[1] = np.convolve(Spec[1] - dark, np.ones((n,)) / n, mode='same') / LampSpecAvg[1]
    plt.plot(Spec[0], Spec[1], dash, label=descr)

    if save:
        with open('/tmp/2018_05_11/' + descr + '.csv', 'w', newline='') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter='\t',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            spamwriter.writerows(Spec.T)

    return




def loadPulseLog(fname):
    from datetime import datetime

    s = 0
    N, Time, RawValue = [],[],[]
    with open(fname, newline='') as csvfile:
        pulsereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        infohead = next(pulsereader)
        infoline = next(pulsereader)
        info = {'Noise':int(infoline[0]),'MaxCoutntsScale':int(infoline[1]),'ArduinoPause':int(infoline[2])}
        header = next(pulsereader)
        for row in pulsereader:
            N += [int(row[0])]
            Time += [datetime.strptime(row[1],'%Y-%m-%dT%H:%M:%S.%f')]
            RawValue += [int(row[2])]
#            ArduinoOut += [int(row[4])]
    info['l']=int(row[0])

    x_list = []
    zero_time = Time[0].timestamp() * 1000.0
    for point in Time:
        x_list.append(point.timestamp() * 1000.0 - zero_time)

    return {'N': N, 'Time': Time, 'AbsTime': x_list, 'RawValue': RawValue, 'info': info}

def loadPulseLogOld(fname): #With ArduinoValue
    from datetime import datetime

    s = 0
    N, Time, RawValue,ArduinoOut = [],[],[],[]
    with open(fname, newline='') as csvfile:
        pulsereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        infohead = next(pulsereader)
        infoline = next(pulsereader)
        info = {'Noise':int(infoline[0]),'MaxCoutntsScale':int(infoline[1]),'ArduinoPause':int(infoline[2])}
        header = next(pulsereader)
        for row in pulsereader:
            N += [int(row[0])]
            Time += [datetime.strptime(row[1],'%Y-%m-%dT%H:%M:%S.%f')]
            RawValue += [int(row[2])]
            ArduinoOut += [int(row[4])]
    info['l']=int(row[0])
    return {'N':N,'Time':Time, 'RawValue':RawValue, 'ArduinoOut':ArduinoOut,'info':info}


def loadxy(fname, skip = 0, aist = False, AistMap = False, Gwyddion=False,OOptics=False, doCorrect=0):
    """Universal function for reading csv (xy) data from different sources
    :param fname: full path
    :param skip: skip first n-lines
    :param aist: set true if file from AistNT software
    :param AistMap: set true if in from AistMap (i don't realy remember what does it mean)
    :param Gwyddion: xy-file from gwiddion
    :param OOptics: spec from Ocean Optics spectrometer (Now it is Ocean Insight)
    :param doCorrect: try to remove spikes on spectra (may work incorrect, if you have narrow lines)
    :return: list with [wavelenght, intensity, dict with info]
    """
    if aist or AistMap:
        deli = '\t'
        iy = 3
    elif Gwyddion:
        deli=';'
        iy = 1
    elif OOptics:
        deli='\t'
        iy=1
        coma2point(fname)
    else:
        deli = ','
        iy = 1

    info = {}
    lod = [[], []]  # list of date
    skiplines = skip
    s = 0
    tmp1 = []
    count = 0
    with open(fname, newline='') as csvfile:
        spectrreader = csv.reader(csvfile, delimiter=deli, quotechar='|')
        for row in spectrreader:
            if s != skiplines:
                s += 1
                if AistMap and s <8:
                  info[row[0]] = row[3]
                # if s == 4:
                #     fname = row[1]
                if Gwyddion:
                    if s == 3:
                        info['x']=row[0][1:-1]
                        info['y']=row[1][1:-1]
                # if OOptics:
                #     if s == 7:
                #         info['IntTime'] = row
                continue

            if row == '':
                break

            lod[0] += [float(row[0])]
            if doCorrect:
                tmp1 += [float(row[iy])]
            else:
                lod[1] += [float(row[iy])]

    if doCorrect:
        tmp = np.array(tmp1)
        med = np.median(tmp)

        for i in range(len(tmp)):
            if tmp[i] <= med * 1.75:
                lod[1] += [tmp[i]]
            else:
                lod[1] += [med]
                count += 1
            #
            # # lod[1] =
        if count > 0:
            print(fname[-10:-4], 'replaced values: ', count)
    lod += [info]
    return lod

def loadCurve(fname):
    """ Read from AIST-plot"""
    return loadxy(fname,deli='\t')



def GaussFit(data):
    def gauss(x, *p):
        A, mu, sigma = p
        return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

    from scipy.optimize import curve_fit

    hist, bin_edges = np.histogram(data, density=True)
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    p0 = [1., 1., 1.]
    coeff, var_matrix = curve_fit(gauss, bin_centres, hist, p0=p0)
    hist_fit = gauss(bin_centres, *coeff)
    return ([bin_centres, hist_fit])


def PlotProcessing(PlotTurple,Settings):
    s = PlotTurple
    TunSpecRaw = loadspc(s[1], Settings['SpcFolder'], doCorrect=Settings['doCorrect'])
    if TunSpecRaw[0][0] > 1500:
        print('spectra ' + s[1] + ' > 1500nm')
    if np.average(TunSpecRaw[1][0]) > 400:
        print('average intensity ' + s[1] + ' > 400')
    if np.average(TunSpecRaw[1][0]) < 50:
        print('average intensity ' + s[1] + ' < 50')

    if Settings['convolve']: # smoothing by n points (but now i reccomend to use slovatsky-golay smooth
        PlotSpec = np.array(np.convolve(TunSpecRaw[1], np.ones((Settings['n'],)) / Settings['n'], mode='valid'))
        xar = TunSpecRaw[0][int(Settings['n'] / 2):-int(Settings['n'] / 2) + 1]  # maybe [int(n/2)-1:-int(n/2)]. different by 0.2 nm
    else:
        PlotSpec = np.array(TunSpecRaw[1])
        xar = TunSpecRaw[0]

    if Settings['useDark']:
        DarkSpecRaw = loadspc(s[0], Settings['SpcFolder'], doCorrect=Settings['doCorrect'])

        if Settings['convolve']:
            DarkSpec = np.convolve(DarkSpecRaw[1], np.ones((Settings['n'],)) / Settings['n'], mode='valid')
            PlotSpec -= DarkSpec
        else:
            PlotSpec -= DarkSpecRaw[1]
    if Settings['useConstantDark']:
        PlotSpec -= Settings['ConstantDark']

    if Settings['useTIME']:
        PlotSpec /= float(TunSpecRaw[2]['TIME'])

    if Settings['useQE']:
        SpecMqe = []
        for i in range(len(PlotSpec)):
            SpecMqe += [PlotSpec[i] / AndourQE(xar[i])]
        PlotSpec = SpecMqe

    if Settings['useHF']:
        SpecMhf = []
        for i in range(len(PlotSpec)):
            SpecMhf += [PlotSpec[i] / HFF(xar[i])]
        PlotSpec = SpecMhf

    if Settings['evmes']:
        xar = 4.135667662 * (10 ** -15) * 299792458 * 10 ** 9 / xar
        PlotSpec *= 4.135667662 * (10 ** -15) * 299792458 * 10 ** 9 / xar ** 2
        PlotSpec /= 4.13 * 100  ######### Да бы абстрактная интенсивность была эквивалентна интенсивности от длин волн

        SpecMaxCentre = xar[np.argmax(PlotSpec)]
        SpecMaxCentrenm = (4.135667662 * (10 ** (-15)) * 299792458 * 10 ** 9) / SpecMaxCentre
        descr = str(int((SpecMaxCentre) * 100) / 100) + '/' + str(int(SpecMaxCentrenm))
    else:
        # descr = str(int(xar[np.argmax(PlotSpec)]))
        descr = ' '

    LineLabel = s[1][-7:-4] + ' ' + s[2] #+ ' (' + descr + ')'

    if Settings['norm']:
        PlotSpec /= np.max(PlotSpec)

    return (xar,PlotSpec,LineLabel)

def coma2point(filename):
    with open(filename, 'r') as f:
        old_data = f.read()

    new_data = old_data.replace(',', '.')

    with open(filename, 'w') as f:
        f.write(new_data)

def p2c(filename):
    with open(filename, 'r') as f:
        old_data = f.read()

    new_data = old_data.replace('.', ',')

    with open(filename, 'w') as f:
        f.write(new_data)

