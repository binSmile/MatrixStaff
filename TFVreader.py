import numpy as np

# bdir = 'C:\\..\\GlobalProjects\\STM-Electroluminescence\\WorkWithData\\2020-07-20 Shkoldin STM-LE Smart Princeton\\Specs\\'
# fname = 'Autosave_124697.tvf'


def readTFV(file):
    """
    reader of TFV-files with spectrum
    :param file: path
    :return: list with [wavelengths, intensity, dict with info]
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(file)
    root = tree.getroot()

    info = root[1][0].get('InfoSerialized')


    import xmltodict
    ii = xmltodict.parse(info,process_namespaces=True)
    infoDict = {
        'Record Time': ii['Info']['Groups']['Group'][0]['Items']['Item']['Value'],
        'Setup':ii['Info']['Groups']['Group'][1]['Items']['Item']['Value']

    }
    for i in ii['Info']['Groups']['Group'][3]['Items']['Item']:
        infoDict[i['Name']]=i['Value']
    infoDict['ExpTime'] = float(infoDict['Exposure_Time_(ms)'])
    cv = ii['Info']['Groups']['Group'][4]['Items']['Item'][0]['Value'].replace(',','.')
    infoDict['CentralWL'] = float(cv)

    wl = root[1][0][3][0].get('ValueArray').split('|')
    wl = np.array([float(x) for x in wl][1:])

    yDim = int(root[1][0][6][0].get('yDim'))
    xDim = int(root[1][0][6][0].get('xDim'))

    c = root[1][0][6][0].text.split(';')
    counts = np.array([int(x) for x in c])
    counts = counts.reshape((yDim,xDim))
    return [wl,counts,infoDict]