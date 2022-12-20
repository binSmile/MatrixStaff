import sys
from pathlib import Path

import xlsxwriter
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QFileDialog
from PyQt6.QtGui import QPalette, QColor

class Color(QWidget):

    def __init__(self, color):
        super(Color, self).__init__()
        self.setAutoFillBackground(True)

        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(color))
        self.setPalette(palette)
class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Matrix exp to xlsx")

        layout = QVBoxLayout()

        buttonL = QPushButton("Select directory with Matrix experiment")


        buttonL.clicked.connect(self.the_buttonL_was_clicked)



        layout.addWidget(buttonL)
        # layout.addWidget(buttonS)
        # layout.addWidget(buttonR)

        # layout.addWidget(Color('blue'))




        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def the_buttonL_was_clicked(self):
        print("Select directory with Matrix experiment")
        directoryL = str(QFileDialog.getExistingDirectory(self, "Select Omicron Data"))+'/'
        print('Selected: ' + str(directoryL))
        directoryS = QFileDialog.getSaveFileName(self,caption='Save report',filter='*.xlsx')
        print('Path for report: ' + str(directoryS))

        directoryS = directoryS[0]
        directoryL = str(Path(directoryL))+'\\'

        workbook = xlsxwriter.Workbook(directoryS)
        format5 = workbook.add_format({'num_format': 'dd/mm/yy hh:mm'})
        head_format = workbook.add_format({'bold': True})

        print('Reading....',end=' ')
        Data, ErData = MatrixDirStat.GetFolderStat(directoryL)
        worksheet = workbook.add_worksheet('Stat')
        Data[1:] = sorted(Data[1:], key=lambda dd: dd[12], reverse=False)

        for row_num, line in enumerate(Data):
            worksheet.write_row(row_num, 0, line)
            if row_num >= 2:
                for i in [8, 9]:  # highlight changes
                    if line[i] != Data[row_num - 1][i]:
                        worksheet.write(row_num, i, line[i], head_format)

        worksheet.autofilter(0, 0, row_num, 24)  # Same as above.
        # worksheet.filter_column('B', 'x = 1')
        worksheet.freeze_panes(1, 0)  # Freeze the first row.

        worksheet.set_column(first_col=12, last_col=12, cell_format=format5, width=12)
        worksheet.set_column(first_col=1, last_col=2, width=4)
        worksheet.set_row(row=0, cell_format=head_format)

        if ErData:
            worksheet = workbook.add_worksheet('Errors')
            for row_num, line in enumerate(ErData):
                worksheet.write_row(row_num, 0, line)
            worksheet.set_column(first_col=0, last_col=0, width=60)

        workbook.close()
        print('Done')

import MatrixDirStat
app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()