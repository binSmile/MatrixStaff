# MatrixStaff
Some tools for work with data from Matrix (Omicron SPM). I worked on Omicron VT650 (Scienta Omicron) and write some tools for analyze data.
I will be happy, if something will be useful for you.

- MatrixDirStat - generate table with all parameters for each data file in experiment.
- StepByStepСVCAnalize - I analyzed many Current–voltage characteristic, their derivative and plotted the maps of them.



Some of that code was used for my PhD Thesis (PhD summary)[https://cloud.physics.itmo.ru/s/bz6SCaFym7nEdWa] (Thesis in Ru)[http://dissovet.itmo.ru/qr/?number=596524], and for some publication:
for Example: Lebedev, Denis V., et al. "Nanoscale Electrically Driven Light Source Based on Hybrid Semiconductor/Metal Nanoantenna." The Journal of Physical Chemistry Letters 13 (2022): 4612-4620. DOI:10.1021/acs.jpclett.2c00986 IF:6.38 Q1



## MatrixDirStatGUI.py

This small program scan directory with experiment results of Matrix (Omicron SPM, Scienta Omicron) and save xlsx-table with names and parameters of all files. 
I prefer pure python, and use DirStat.py without GUI. Before the using, check all paths.
In my project I used such tree:
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

