import glob
from datetime import datetime
import  re
from access2thematrix import access2thematrix


def GetFolderStat(DataDirIn):
    filelist = []
    # for m in title:
    filelist += glob.glob(DataDirIn + '*_mtrx')
    filelist.sort()
    ErData = []
    Data = [['Scan cycle','Run Cycle','Channel','Dataset',
             'sample_name', 'channel_name', 'Spectroscopy min','Spectroscopy max',
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
                      'Points scan',
                      'lines',
             'Height, nm',
             'Width,nm',
             'Exp start time from name',
             'Creation comment',
             'EEPA::Spectroscopy.Delay_T2_1 (ms)',
             'EEPA::Spectroscopy.Raster_Time_1 (ms)',
             'EEPA::I_V.Initial_Delay (ms)',
                        ]]
    for filename in filelist:
        fn = filename.split('\\')[-1]
        print(fn)
        Ptitle = filename.split('--')[-1][:-5].split('.')
        nametime = re.findall(r'\d{6}', fn)[0]
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
            fn = filename.split('\\')[-1]
            ErData += [[fn,'Open failed']]


        ivp1, ivp2, ivp3 = '','',''
        if  mtrx_data.channel_name in ['Aux2(V)','Aux1(V)','I(V)'] :
            trastr = mtrx_data.param['EEPA::Spectroscopy.Raster_Time_1'][0]*1e3
            fb = mtrx_data.param['EEPA::Spectroscopy.Enable_Feedback_Loop'][0]
            try:
                min, max =  trace.data.min(), trace.data.max()
                ref = str(mtrx_data.ref_by)
                SpecPoints = mtrx_data.param['EEPA::Spectroscopy.Device_1_Points'][0]
                if mtrx_data.channel_name == 'I(V)':
                    ivp1 = mtrx_data.param['EEPA::Spectroscopy.Delay_T2_1'][0] * 1000
                    ivp2 = mtrx_data.param['EEPA::Spectroscopy.Raster_Time_1'][0] * 1000
                    ivp3 = mtrx_data.param['EEPA::I_V.Initial_Delay'][0] * 1000
            except:
                print('Problem with min/max or ref_by in %s' % filename)
                min, max = '', ''
                ref = str(mtrx_data.axis)
                SpecPoints = ''

        else:
            trastr = mtrx_data.param['EEPA::XYScanner.Raster_Time'][0]*1e3
            fb = ''
            min, max = '', ''
            ref = str(mtrx_data.axis)
            SpecPoints = ''


        time = datetime.strptime(mtrx_data.param['BKLT'],'%A, %d %B %Y %H:%M:%S')
        timef = time.isoformat()
        Data += [[S,Num, chan,
                   mtrx_data.data_set_name,
                  mtrx_data.sample_name,
                  mtrx_data.channel_name, min, max,
                  mtrx_data.param['EEPA::GapVoltageControl.Voltage'][0],
                   # mtrx_data.param['EEPA::GapVoltageControl.Voltage'][1],
                  mtrx_data.param['EEPA::Regulator.Setpoint_1'][0]*1e9,
                  # mtrx_data.param['EEPA::Regulator.Setpoint_1'][1],
                  SpecPoints,
                  fb,
                  time,
                  mtrx_data.param['EEPA::XYScanner.Angle'][0],
                  ref,
                  trastr,
                  mtrx_data.param['EEPA::Spectroscopy.Maximum_Points_1'][0],
                  mtrx_data.param['EEPA::XYScanner.Lines'][0],
                  mtrx_data.param['EEPA::XYScanner.Height'][0]*1e9,
                  mtrx_data.param['EEPA::XYScanner.Width'][0]*1e9,
                  nametime,
                  mtrx_data.creation_comment,
                  ivp1, ivp2, ivp3,
                  ]]
    return (Data,ErData)





if __name__ == '__main__':
    # The main dir with project. Be careful with paths and read readme


    project_path = 'C:\\DIR WITH DATA\\' # BA

    # List with dates for analizing
    IntestingDates = ['2022-04-29']
    DataDirOut = project_path + 'WorkWithData\\'

    import xlsxwriter
    workbook = xlsxwriter.Workbook(DataDirOut+'stat.xlsx')
    format5 = workbook.add_format({'num_format': 'dd/mm/yy hh:mm'})
    head_format = workbook.add_format({'bold': True})



    AllErrors = []
    for ed in IntestingDates:
        DataDirIn = project_path + 'ExpData\\Omicron results\\%s\\' % ed
        Data, ErData = GetFolderStat(DataDirIn)
        AllErrors += ErData
        worksheet = workbook.add_worksheet(ed)
        Data[1:] = sorted(Data[1:], key=lambda dd: dd[12], reverse=False)

        for row_num, line in enumerate(Data):
            worksheet.write_row(row_num, 0, line)
            if row_num >= 2:
                for i in [8,9,14,10,11,15,16,17,18,19]: # highlight changes
                    if line[i] != Data[row_num-1][i]:
                        worksheet.write(row_num,i,line[i],head_format)


        worksheet.autofilter(0, 0, row_num, 25)  # Same as above.
        # worksheet.filter_column('B', 'x = 1')
        worksheet.filter_column('C', 'x = Aux2(V)')
        worksheet.filter_column('P', 'P >= 1')

        worksheet.freeze_panes(1, 0)  # Freeze the first row.


        worksheet.set_column(first_col=12, last_col=12, cell_format=format5, width=12)
        worksheet.set_column(first_col=1,last_col=2, width=4)
        worksheet.set_row(row=0,cell_format=head_format)

    if AllErrors:
        worksheet = workbook.add_worksheet('Errors')
        for row_num, line in enumerate(AllErrors):
            worksheet.write_row(row_num, 0, line)
        worksheet.set_column(first_col=0,last_col=0, width=60)



    workbook.close()
