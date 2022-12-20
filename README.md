# MatrixStaff
Some tools for work with data from Matrix (Omicron SPM). I worked on Omicron VT650 (Scienta Omicron) and write some tools for analyze data.
I will be happy, if something will be useful for you.

- MatrixDirStat - generate table with all parameters for each data file in experiment.
- StepByStepСVCAnalize - I analyzed many Current–voltage characteristic, their derivative and plotted the maps of them.



Some of that code was used for my PhD Thesis ([PhD summary](https://cloud.physics.itmo.ru/s/bz6SCaFym7nEdWa), [Thesis in Ru](http://dissovet.itmo.ru/qr/?number=596524)), and for some publication:
for Example: Lebedev, Denis V., et al. "Nanoscale Electrically Driven Light Source Based on Hybrid Semiconductor/Metal Nanoantenna." The Journal of Physical Chemistry Letters 13 (2022): 4612-4620. DOI:10.1021/acs.jpclett.2c00986 IF:6.38 Q1



## MatrixDirStatGUI.py

This small program scan directory with experiment results of Matrix (Omicron SPM, Scienta Omicron) and save xlsx-table with names and parameters of all files. 
I prefer pure python, and use DirStat.py without GUI. Before the using, check all paths.
In my project I used such tree:
```
.
└── ProjectName
    ├── ...
    ├── ExpData
    │   └── OmicronResults
    │       ├── ...
    │       ├── 2022-04-29
    │       ├── 2022-07-28
    │       └── 2022-08-13
    ├── WorkWithData
    │   └── 2022-04-29
    └── ..
```

### Requirements
For this tool, you need to install:
- [access2thematrix](https://pypi.org/project/access2theMatrix/)
- [xlsxwriter](https://pypi.org/project/xlsxwriter)
- PyQT6 (for GUI)

## StepByStepСVCAnalize.py
I analyzed many Current–voltage characteristic, their derivative and plotted the maps of them. Maybe, you can find some useful code for Current–voltage characteristic analyze.
I try to highlight some functions:

- GetDataAsDataset() - work in pair with GetFileList(). It transforms measurements to pandas's Dataframe.
- ShowMap() - Shows a scan with points, where was measured CVC
- Calcd2idv2map() - Calculate the 2nd derivative and plot "finger-print" map

## ScanQEAnalize
This script need for evaluation of estimated quantum efficiency of STM-Light Emission process.
1. It get topography (Z channel), Current-maps (I channel), Light-emission map (Aux chanel).
2. Convert Volts of Aux to counts 
3. Divide Aux map in I-map
4. Mupltiply on specific value (etta), which correspond to light collection efficiency

#### QEbyMap(ed,NNN,DACmax=5000,direct='forward/up') - main function
ed - day of the experiment,
NNN - run number of intrested scan
DACmax - correspond to maximum photons count when DAC give maximum (5V) on Aux
    if not specified, the script try read log of Pulses (see another project)
    (it need some fixes)
direct - direction of scan.

This function create Gwiddion file with results.

#### shift5(...) and findTdelta(...)
##### shift
Be prevented. If you use DAC such as mine, you may have a little delay between Z,I... and Aux-signal.
It was provided by time, which need for collecting of pulses from photon counter and give it for Omicron CU.
Usually, i use DAC's integration equal 10ms, and T_rast = 1...10 ms.
It provide zero-like filling on the edge of data array. If you are nerd, you can found data from scan with another scan dirrection.
##### findTdelt
Search of delay by fft.

#### GetAparams(..)
Try to find the corresponding pulse log, and read DAC params

#### CurveQEA()
Script to analize QE during of current-voltage characteristics (or V-sweeep).





