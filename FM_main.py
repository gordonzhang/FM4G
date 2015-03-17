# -*- coding:utf-8 -*-
from __future__ import division

import sys
import os
import random
import copy
import time
from itertools import izip_longest, cycle

import numpy as np
import scipy.ndimage as ndimage
from matplotlib import pyplot as plt
from matplotlib import mlab as ml
import matplotlib.cm as cm
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib as mpl
import seaborn as sns

from PyQt4.QtCore import *
from PyQt4.QtGui import *
mpl.rc("figure", facecolor="white")
DPI_default = 93

class boxResult(QFrame):
    def __init__(self, text=None, result=None, parent=None):
        super(boxResult, self).__init__(parent)
        self.setFixedSize(240,150)
        self.setStyleSheet('''color: white; background-color: #5e6e8c; font: 18px; border-radius: 15px;''')
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignVCenter)

        titleLabel = QLabel(text)
        titleLabel.setAlignment(Qt.AlignHCenter)
        self.resultLabel = QLabel(result)
        self.resultLabel.setAlignment(Qt.AlignHCenter)    

        layout.addWidget(titleLabel)
        layout.addSpacing(15)
        layout.addWidget(self.resultLabel)

    def updatePlot(self, result):
        self.resultLabel.setText(result)


class NPVHist(FigureCanvas):
    def __init__(self, priceList=None, parent=None):
        self.fig = plt.figure(figsize=(8,5), dpi=DPI_default)
        FigureCanvas.__init__(self, self.fig)
        self.title = 'Field Net Cash Flow with Oil Pirces at\$%d, %d, %d, %d and %d \
                % (priceList[self.n[0]],priceList[self.n[1]],priceList[self.n[2]],priceList[self.n[3]],priceList[self.n[4]])'
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Fixed,
                                   QSizePolicy.Fixed)
        FigureCanvas.updateGeometry(self)
        sns.set_style('white')
        # sns.despine()

        self.ax = self.fig.add_subplot(111)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['left'].set_visible(False)

        # self.ax.get_xaxis().tick_bottom()

        self.ax.hold(True)

        self.ax.set_xlabel('Net Present Value ($mm.)', fontsize=12)
        self.ax.set_ylabel('Frequency', fontsize=12)
        self.ax.set_title('Net Present Value Distribution of Simulated Results', fontsize=14, fontweight='bold')

        self.draw()

    def updatePlot(self, NPVdata, lowerLimit, upperLimit):
        self.data = [x/1000000 for x in NPVdata]
        positiveNPVratio = sum([x>0 for x in NPVdata])/len(NPVdata)*100
        sns.set_style('white')
        self.ax.cla()

        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['left'].set_visible(False)

        self.ax.hold(True)
        self.ax.set_xlabel('Net Present Value ($mm.)', fontsize=12)
        self.ax.set_ylabel('Frequency', fontsize=12)
        self.ax.set_title('Net Present Value Distribution of Simulated Results', fontsize=14, fontweight='bold')

        self.ax.hist(self.data, bins=40, normed=1, facecolor='#3F5D7D', alpha=0.9)
        self.ax.text(0.65, 0.85, 'Ratio of Positive NPV: %0.1f' %positiveNPVratio +'%', fontweight='bold', transform=self.ax.transAxes)
        self.ax.text(0.65, 0.80, '95%% Conf. Interval: [%0.1f, %0.1f] $mm.' % (lowerLimit, upperLimit), fontweight='bold', transform=self.ax.transAxes)
        # draw new figure
        self.draw()


class AllNetCashflowChart(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = plt.figure(figsize=(8,5), dpi=DPI_default)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Fixed,
                                   QSizePolicy.Fixed)
        FigureCanvas.updateGeometry(self)
        sns.set_style('white')

        self.ax = self.fig.add_subplot(111)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.hold(True)

        self.ax.set_xlabel('Year', fontsize=12)
        self.ax.set_ylabel('Cash Flow', fontsize=12)
        self.ax.set_title('All Simulated Yearly Net Cash Flows', fontsize=14, fontweight='bold')

        self.draw()

    def updatePlot(self, yearlyAverageNetCashflow, yearlyNetCashflowTableAllSamplesCurrentPrice):
        sns.set_style('white')
        self.ax.cla()

        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)

        self.ax.hold(True)
        self.ax.set_xlabel('Year', fontsize=12)
        self.ax.set_ylabel('Cash Flow', fontsize=12)
        self.ax.set_title('All Simulated Yearly Net Cash Flows', fontsize=14, fontweight='bold')

        self.ax.plot([0 for _ in xrange(len(yearlyAverageNetCashflow))], "--", lw=1.5, color="black", alpha=0.3)
        self.ax.plot(yearlyAverageNetCashflow, lw=2.5, alpha=0.9, color='#253649')
        for cashflow in yearlyNetCashflowTableAllSamplesCurrentPrice:
            self.ax.plot(cashflow, lw=.4, alpha=0.2, color='#3F5D7D')
        self.draw()


class oilProdChart(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = plt.figure(figsize=(8,5), dpi=DPI_default)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Fixed,
                                   QSizePolicy.Fixed)
        FigureCanvas.updateGeometry(self)
        sns.set_style('whitegrid')
        self.ax = self.fig.add_subplot(111)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        # self.ax.spines['bottom'].set_visible(False)
        self.ax.hold(True)

        self.ax.set_xlabel('Month', fontsize=12)
        self.ax.set_ylabel('Production Volume', fontsize=12)
        self.ax.set_title('Averaged Porducing Volume of Each Well', fontsize=14, fontweight='bold')

        self.draw()

    def updatePlot(self, monthlyEachWellProdVolTable):
        monthlyEachWellProdVolTable_stack = np.asarray(np.cumsum(monthlyEachWellProdVolTable, axis=0))
        while monthlyEachWellProdVolTable_stack[-1,-1] == 0:
            monthlyEachWellProdVolTable_stack = monthlyEachWellProdVolTable_stack[:,:-1]

        sns.set_style('whitegrid')
        self.ax.cla()
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        # self.ax.spines['bottom'].set_visible(False)

        self.colorList = sns.color_palette(["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"])
        colorIter = cycle(self.colorList)

        self.ax.hold(True)
        self.ax.set_ylim(0,max(monthlyEachWellProdVolTable_stack[-1]))
        self.ax.set_xlabel('Month', fontsize=12)
        self.ax.set_ylabel('Production Volume', fontsize=12)
        self.ax.set_title('Averaged Porducing Volume of Each Well', fontsize=14, fontweight='bold')

        c = colorIter.next()
        self.ax.plot(monthlyEachWellProdVolTable_stack[0], color=c, lw=1)
        self.ax.fill_between(np.arange(len(monthlyEachWellProdVolTable_stack[0])), 0, monthlyEachWellProdVolTable_stack[0,:], facecolor=c, alpha=.3)
        for i in xrange(1, len(monthlyEachWellProdVolTable_stack)):
            c = colorIter.next()
            self.ax.plot(monthlyEachWellProdVolTable_stack[i], color=c, lw=1)
            self.ax.fill_between(np.arange(len(monthlyEachWellProdVolTable_stack[0])), monthlyEachWellProdVolTable_stack[i-1,:], monthlyEachWellProdVolTable_stack[i,:], facecolor=c, alpha=.3)
        
        self.draw()


class IntervalChart(FigureCanvas):
    def __init__(self, priceList, parent=None):
        self.fig = plt.figure(figsize=(8,5), dpi=DPI_default)
        self.priceList = priceList
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Fixed,
                                   QSizePolicy.Fixed)
        FigureCanvas.updateGeometry(self)
        sns.set_style('white')
        # sns.despine()
        self.alp = 0.5
        self.colorList = [(255, 187, 120), (174, 199, 232), (23, 190, 207), (197, 176, 213),  (196, 156, 148)]
        for i in range(len(self.colorList)):
            r, g, b = self.colorList[i]
            self.colorList[i] = (r / 255., g / 255., b / 255.)

        self.ax = self.fig.add_subplot(111)
        self.ax.hold(True)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)

        self.ax.set_xlabel('Year', fontsize=12)
        self.ax.set_ylabel('Net Cash Flow ($mm.)', fontsize=12)
        self.title = 'Field Net Cash Flow with Oil Pirces at $%d, %d, %d, %d and %d' \
                % (self.priceList[0],self.priceList[1],self.priceList[2],self.priceList[3],self.priceList[4])
        self.ax.set_title(self.title, fontsize=14, fontweight='bold')
        self.draw()

    def updatePlot(self, priceList, data, NPVdifferentPrices):
        self.priceList = priceList
        self.data = data

        # $ to $mm.
        self.data = [[item/1000000 for item in itemlist] for itemlist in self.data]

        sns.set_style('white')
        # clear current figure
        self.ax.cla()

        ind = np.arange(len(self.data[-1]))
        self.ax.set_xlabel('Year', fontsize=12)
        self.ax.set_ylabel('Net Cashflow ($mm.)', fontsize=12)
        self.title = 'Field Net Cash Flow with Oil Pirces at $%d, %d, %d, %d and %d' \
                % (self.priceList[0],self.priceList[1],self.priceList[2],self.priceList[3],self.priceList[4])
        self.ax.set_title(self.title, fontsize=16, fontweight='bold')
        self.ax.set_xticks(ind)

        self.ax.plot([0 for _ in xrange(len(self.data[-1]))], "--", lw=1.5, color="black", alpha=0.3)
        for i in xrange(5):
            self.ax.plot(self.data[i], color=self.colorList[i], linewidth=2.5)
            j = int(len(self.data[i])*3/4)
            if i%2 == 0:
                ypos = self.data[i][j]+.6
            else:
                ypos = self.data[i][j]-.6
            self.ax.text(j+.3, ypos, 'OP:$%0.0f\nNPV:%0.1fmm' % (self.priceList[i], NPVdifferentPrices[i]),
                color=self.colorList[i], fontsize=9, fontweight='bold', horizontalalignment='left', verticalalignment='center')
        # draw new figure
        self.draw()


class stackedBarchart(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = plt.figure(figsize=(8,5), dpi=DPI_default)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Fixed,
                                   QSizePolicy.Fixed)
        FigureCanvas.updateGeometry(self)
        sns.set_style("white")


        self.colorList = sns.color_palette("cubehelix", 6)
        self.colorList = sns.cubehelix_palette(6, start=.5, rot=-.65, light=0.95)

        self.ax = self.fig.add_subplot(111)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['left'].set_visible(False)

        self.ax.set_xlabel('Year', fontsize=12)
        self.ax.set_ylabel('Outflow ($mm.)', fontsize=12)
        self.ax.set_title('Outflow Breakdown', fontsize=16, fontweight='bold')
        self.ax.hold(True)
        
        self.draw()

    def updatePlot(self, royaltyList, capitalList, opertList, bonusList, rentalList, incomeTaxList):
        self.royaltyList = [item/1000000 for item in royaltyList]
        self.capitalList = [item/1000000 for item in capitalList]
        self.opertList = [item/1000000 for item in opertList]
        self.bonusList = [item/1000000 for item in bonusList]
        self.rentalList = [item/1000000 for item in rentalList]
        self.incomeTaxList = [item/1000000 for item in incomeTaxList]

        d0 = self.royaltyList
        d1 = [a+b for a,b in zip(self.royaltyList, self.capitalList)]
        d2 = [a+b for a,b in zip(d1, self.opertList)]
        d3 = [a+b for a,b in zip(d2, self.bonusList)]
        d4 = [a+b for a,b in zip(d3, self.rentalList)]

        # clear current figure
        self.ax.cla()
        ind = np.arange(len(self.royaltyList))
        width = 0.6

        p1 = self.ax.bar(ind, self.royaltyList, width, color='#0e244e')
        p2 = self.ax.bar(ind, self.capitalList, width, color='#143470', bottom=d0)
        p3 = self.ax.bar(ind, self.opertList, width, color='#425c8c', bottom=d1)
        p4 = self.ax.bar(ind, self.bonusList, width, color='#6e7b94', bottom=d2)
        p5 = self.ax.bar(ind, self.rentalList, width, color='#ced3db', bottom=d3)
        p6 = self.ax.bar(ind, self.incomeTaxList, width, color='#fcfcfc', bottom=d4)

        self.ax.set_xlabel('Year', fontsize=12)
        self.ax.set_ylabel('Outflow ($mm.)', fontsize=12)
        self.ax.set_title('Outflow Breakdown', fontsize=16, fontweight='bold')
        self.ax.set_xticks(ind)
        # self.ax.yticks(np.arange(0,81,10))
        self.ax.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]), ('Royalty', 'Capital', 'Operating Cost', 'Bonus', 'Rental', 'Tax'))

        # draw new figure
        self.draw()


class breakevenPriceChart(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = plt.figure()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        sns.set_style("whitegrid")
        # self.colorList = sns.color_palette("cubehelix", 6)
        self.colorList = sns.cubehelix_palette(6, start=.5, rot=-.65, light=0.95)
        self.ax = self.fig.add_subplot(111)
        self.ax.hold(True)
        self.draw()

    def updatePlot(self, data):
        self.ax.cla()
        self.data = data
        self.ax.plot(range(len(self.data)), self.data, linewidth=2.5)
        self.draw()


class Heatmap(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = plt.figure(figsize=(8, 6), dpi=DPI_default)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Fixed,
                                   QSizePolicy.Fixed)
        FigureCanvas.updateGeometry(self)
        self.ax = self.fig.add_subplot(111)
        self.ax.hold(True)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(False)

        self.ax.set_xlabel('Oil Price', fontsize=12)
        self.ax.set_ylabel('Success Rate %', fontsize=12)
        self.ax.set_title('Cross Analysis of Oil Price and Drilling Success Rate', fontsize=16, fontweight='bold')
        self.draw()

    def updatePlot(self, x, y, z, contour):
        y = [(a*100) for a in y]
        z = [[a/1000000 for a in b] for b in z]
        xx = ndimage.zoom(x, 6, order=1)
        yy = ndimage.zoom(y, 6, order=1)
        zz = ndimage.zoom(z, 6, order=1)

        self.ax.cla()
        self.ax.set_xlabel('Oil Price $', fontsize=12)
        self.ax.set_ylabel('Success Rate %', fontsize=12)
        self.ax.set_title('Cross Analysis: Oil Price vs Drilling Succ. Rate', fontsize=16, fontweight='bold')

        im = self.ax.imshow(z, extent=[x[0],x[-1],y[0],y[-1]], interpolation='bilinear', cmap=cm.RdBu, origin='lower')
        if contour:
            self.ax.contour(xx, yy, zz, [0], colors='#3F5D7D', alpha=.9, linewidths=1.5)

        if not hasattr(self, 'cb'):
            self.cb = self.fig.colorbar(im)
            self.cb.set_label('Net Present Value ($mm.)')

        # draw new figure
        self.draw()


####################################################################################################

####################################################################################################

####################################################################################################

__version__ = "1.0.0"

class FModel(QMainWindow):
    def __init__(self, parent=None):
        super(FModel, self).__init__(parent)
        self.initUI()
        

    def initUI(self):
        self.setStyleSheet('''
                QMainWindow{
                background-color: white;
                }

                QSplitter{
                background-color: white;
                padding-top: 5px;
                }

                QPushButton{
                padding-top: 6px;
                padding-bottom: 6px;
                border-radius: 6px;
                background-color: #445669;
                color: #fcfeff;
                border: 2px solid #ababab
                }

                QPushButton:pressed{
                background-color: #538bc2;
                }

                QDoubleSpinBox{
                border: 2px solid #fcfeff;
                background-color: #fcfeff;
                color: #4a4a4a;
                font: 14px;
                }

                QSpinBox{
                border: 2px solid #fcfeff;
                background-color: #fcfeff;
                color: #4a4a4a;
                font: 14px;
                }

                QLabel{
                color: #4a4a4a;
                font: 14px;
                }

                QTableView{
                color: #4a4a4a;
                font: 14px;
                background: #fcfeff;
                border: 0px outset red;
                }

                QHeaderView{
                background-color: #fcfeff;
                }




                QTabWidget::pane { /* The tab widget frame */
                border-top: 2px solid #9B9B9B;
                position: absolute;
                top: -2px;
                }

                QTabWidget::tab-bar {
                alignment: center;
                }

                QTabBar::tab {
                background: #d5dbe3;
                color: #9ba3ab;
                font: bold 16px;
                border: 2px solid #C2C7CB;
                border-bottom-color: #C2C7CB; /* same as the pane color */
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                min-width: 170px;
                min-height: 23px;
                padding: 2px;
                }

                QTabBar::tab:selected, QTabBar::tab:hover {
                background: white;
                color: #404347;
                font: bold 16px;
                }

                QTabBar::tab:selected {
                    border-color: #9B9B9B;
                    border-bottom-color: white; /* same as pane color */
                }
            ''')
        
        self.setWindowTitle('AE&E Financial Model v0.0.2')
        self.setWindowState(Qt.WindowMaximized | Qt.WindowActive)
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setStretchFactor(0,0)
        self.splitter.setStretchFactor(1,1)
        self.parameterBox = ParameterBox()
        self.resultBox = QTabWidget()

        self.tab1 = ResultTab1()
        self.tab2 = ResultTab2()
        self.resultBox.addTab(self.tab1, 'Gross Field Report')
        self.resultBox.addTab(self.tab2, 'AE&&E Report')

        reportBtn = QPushButton('Generate Report')
        reportBtn.setFixedWidth(150)

        reportBtn.clicked.connect(self.generateReport)
        leftWidget = QWidget()
        leftWidget.setFixedWidth(620)
        leftLayout = QVBoxLayout()
        leftWidget.setLayout(leftLayout)
        leftLayout.addWidget(reportBtn)
        leftLayout.addWidget(self.parameterBox)
        self.splitter.addWidget(leftWidget)
        self.splitter.addWidget(self.resultBox)

        self.setCentralWidget(self.splitter)

    def generateReport(self):
        currentPrice, priceList, taxRate, royaltyRate, rental, bonus, yearlyCPI, monthlyCPI, fixedOpertExpense, varOpertExpense,\
            capitalCost, tangibleCapitalPerc, abandonCost, wellData, lowestPrice, highestPrice, recoverableVol, maxNumOfProducingWells = self.parameterBox.getParameters()

        priceListCross = [30., 45., 60., 75., 90., 105.]
        succRateList = [.15, .30, .45, .60, .75, .90]

        NPVTableResult_diffPrices, yearlyNetCashflowTableResult, averageBreakevenYear, yearlyRoyaltyCurrentPrice, yearlyCapitalCurrentPrice, \
        yearlyOpertExpenseCurrentPrice, yearlyBonusCurrentPrice, yearlyRentalCurrentPrice, yearlyIncomeTaxCurrentPrice, NPVTableCross, \
        yearlyNetCashflowTableAllSamplesCurrentPrice, monthlyEachWellProdVolTable, maxExposureList, \
        AEE_NPVTableResult_diffPrices, AEE_yearlyNetCashflowTableResult, AEE_TableCross, AEE_yearlyNetCashflowTableAllSamplesCurrentPrice \
        = self.calculateAllResults(currentPrice, priceList, taxRate, royaltyRate, rental, bonus, yearlyCPI, monthlyCPI, fixedOpertExpense, varOpertExpense,
        capitalCost, tangibleCapitalPerc, abandonCost, wellData, lowestPrice, highestPrice, recoverableVol, maxNumOfProducingWells, priceListCross, succRateList)

        self.tab1.updateGraphs(NPVTable_diffPrices=NPVTableResult_diffPrices, IntervalData=yearlyNetCashflowTableResult, priceList=priceList, averageBreakevenYear=averageBreakevenYear,
                               yearlyRoyaltyCurrentPrice=yearlyRoyaltyCurrentPrice, yearlyCapitalCurrentPrice=yearlyCapitalCurrentPrice,
                               yearlyOpertExpenseCurrentPrice=yearlyOpertExpenseCurrentPrice, yearlyBonusCurrentPrice=yearlyOpertExpenseCurrentPrice,
                               yearlyRentalCurrentPrice=yearlyOpertExpenseCurrentPrice, yearlyIncomeTaxCurrentPrice=yearlyOpertExpenseCurrentPrice,
                               priceListCross=priceListCross, succRateList=succRateList, NPVTableCross=NPVTableCross,
                               yearlyNetCashflowTableAllSamplesCurrentPrice=yearlyNetCashflowTableAllSamplesCurrentPrice, monthlyEachWellProdVolTable=monthlyEachWellProdVolTable,
                               maxExposureList = maxExposureList)
        self.tab2.updateGraphs(NPVTable_diffPrices=NPVTableResult_diffPrices ,priceList=priceList, priceListCross=priceListCross, succRateList=succRateList, AEE_NPVTableResult_diffPrices=AEE_NPVTableResult_diffPrices,
                               AEE_yearlyNetCashflowTableResult=AEE_yearlyNetCashflowTableResult, AEE_TableCross=AEE_TableCross,
                               AEE_yearlyNetCashflowTableAllSamplesCurrentPrice=AEE_yearlyNetCashflowTableAllSamplesCurrentPrice)

    def calculateAllResults(self, currentPrice, priceList, taxRate, royaltyRate, rental, bonus, yearlyCPI, monthlyCPI, fixedOpertExpense, varOpertExpense, capitalCost,
                            tangibleCapitalPerc, abandonCost, wellData, lowestPrice, highestPrice, recoverableVol, maxNumOfProducingWells, priceListCross, succRateList):
        yearlyNetCashflowTableResult = [0 for _ in xrange(5)]
        AEE_yearlyNetCashflowTableResult = [0 for _ in xrange(5)]
        NPVTableResult_diffPrices = [0 for _ in xrange(5)]
        AEE_NPVTableResult_diffPrices = [0 for _ in xrange(5)]
        NPVTableCross = [[0 for i in xrange(len(priceListCross))] for j in xrange(len(succRateList))]
        AEE_NPVTableCross = [[0 for i in xrange(len(priceListCross))] for j in xrange(len(succRateList))]
        # Result of current Price
        NPVTableResult_diffPrices[2], yearlyNetCashflowTableResult[2], averageBreakevenYear, yearlyRoyaltyCurrentPrice, yearlyCapitalCurrentPrice, yearlyOpertExpenseCurrentPrice, \
            yearlyBonusCurrentPrice, yearlyRentalCurrentPrice, yearlyIncomeTaxCurrentPrice, yearlyNetCashflowTableAllSamplesCurrentPrice, monthlyEachWellProdVolTable, maxExposureList, \
            AEE_NPVTableResult_diffPrices[2], AEE_yearlyNetCashflowTableResult[2], AEE_yearlyNetCashflowTableAllSamplesCurrentPrice \
        = self.simulateResultCurrentPrice(nSimulation=200, currentPrice=currentPrice, taxRate=taxRate, royaltyRate=royaltyRate, rental=rental, bonus=bonus,
                                          yearlyCPI=yearlyCPI, monthlyCPI=monthlyCPI, fixedOpertExpense=fixedOpertExpense, varOpertExpense=varOpertExpense,
                                          capitalCost=capitalCost, tangibleCapitalPerc=tangibleCapitalPerc, abandonCost=abandonCost, wellData=wellData,
                                          recoverableVol=recoverableVol, maxNumOfProducingWells=maxNumOfProducingWells)
        # loop the priceList
        for i in [0,1,3,4]:
            price = priceList[i]
            NPVTableResult_diffPrices[i], yearlyNetCashflowTableResult[i] ,_,_,_,_,_,_,_,_,_,_, AEE_NPVTableResult_diffPrices[i], AEE_yearlyNetCashflowTableResult[i],_\
            = self.simulateResultCurrentPrice(nSimulation=100, currentPrice=price, taxRate=taxRate, royaltyRate=royaltyRate, rental=rental, bonus=bonus,
                                              yearlyCPI=yearlyCPI, monthlyCPI=monthlyCPI, fixedOpertExpense=fixedOpertExpense, varOpertExpense=varOpertExpense,
                                              capitalCost=capitalCost, tangibleCapitalPerc=tangibleCapitalPerc, abandonCost=abandonCost, wellData=wellData,
                                              recoverableVol=recoverableVol, maxNumOfProducingWells=maxNumOfProducingWells)

        # loop the succRateList and priceListCross
        for i, succRate in enumerate(succRateList):
            for j, price in enumerate(priceListCross):
                NPVTemp,_,_,_,_,_,_,_,_,_,_,_,AEE_NPVTemp,_,_\
                = self.simulateResultCurrentPrice(nSimulation=100, currentPrice=price, taxRate=taxRate, royaltyRate=royaltyRate, rental=rental, bonus=bonus,
                                                  yearlyCPI=yearlyCPI, monthlyCPI=monthlyCPI, fixedOpertExpense=fixedOpertExpense, varOpertExpense=varOpertExpense,
                                                  capitalCost=capitalCost, tangibleCapitalPerc=tangibleCapitalPerc, abandonCost=abandonCost, wellData=wellData,
                                                  recoverableVol=recoverableVol, maxNumOfProducingWells=maxNumOfProducingWells, succRate=succRate)
                NPVTableCross[i][j] = np.mean(NPVTemp)
                AEE_NPVTableCross[i][j] = np.mean(AEE_NPVTemp)


        return NPVTableResult_diffPrices, yearlyNetCashflowTableResult, averageBreakevenYear, yearlyRoyaltyCurrentPrice, yearlyCapitalCurrentPrice, yearlyOpertExpenseCurrentPrice,\
               yearlyBonusCurrentPrice, yearlyRentalCurrentPrice, yearlyIncomeTaxCurrentPrice, NPVTableCross, yearlyNetCashflowTableAllSamplesCurrentPrice, \
               monthlyEachWellProdVolTable, maxExposureList, \
               AEE_NPVTableResult_diffPrices, AEE_yearlyNetCashflowTableResult, AEE_NPVTableCross, AEE_yearlyNetCashflowTableAllSamplesCurrentPrice

    def simulateResultCurrentPrice(self, nSimulation, currentPrice, taxRate, royaltyRate, rental, bonus, yearlyCPI, monthlyCPI, fixedOpertExpense,
                                   varOpertExpense, capitalCost, tangibleCapitalPerc, abandonCost, wellData, recoverableVol, maxNumOfProducingWells, succRate=None):
        nWells = sum(np.asarray(wellData)[:,2])
        monthlyEachWellProdVolTable = [[] for _ in range(nWells)]
        NPVList = [0 for _ in xrange(nSimulation)]
        AEE_NPVList = [0 for _ in xrange(nSimulation)]
        maxExposureList = [0 for _ in xrange(nSimulation)]
        yearlyAverageNetCashflow = []
        AEE_yearlyAverageNetCashflow = []
        yearlyNetCashflowTableAllSamplesCurrentPrice = []
        AEE_yearlyNetCashflowTableAllSamplesCurrentPrice = []
        yearlyRoyaltyCurrentPrice = []
        yearlyCapitalCurrentPrice = []
        yearlyOpertExpenseCurrentPrice = []
        yearlyBonusCurrentPrice = []
        yearlyIncomeTaxCurrentPrice = []
        breakevenYearList = [0 for _ in xrange(nSimulation)]
        for i in xrange(nSimulation):
            if not succRate:
                monthlyEachWellProdVolTableTemp, monthlyFieldVolList, yearlyNetCashflowTemp, yearlyFieldRevenueList, yearlyFieldCapitalCostListTemp, \
                yearlyFieldRoyaltyListTemp, yearlyFieldBonusListTemp, yearlyFieldOpertCostListTemp, yearlyFieldAbandonCostList, yearlyTaxListTemp, \
                monthlyFieldRevenueList, monthlyFieldCapitalCostList, monthlyFieldRoyaltyList, monthlyFieldBonusList, \
                monthlyFieldOpertCostList, monthlyFieldAbandonCostList, NPVList[i], breakevenYearList[i], maxExposureList[i], AEE_yearlyNetCashflowTemp, AEE_NPVList[i] \
                = self.calculateOneResultCurrentPrice(currentPrice=currentPrice, taxRate=taxRate, royaltyRate=royaltyRate, rental=rental, bonus=bonus, yearlyCPI=yearlyCPI, monthlyCPI=monthlyCPI,
                                                      fixedOpertExpense=fixedOpertExpense, varOpertExpense=varOpertExpense, capitalCost=capitalCost, tangibleCapitalPerc=tangibleCapitalPerc,
                                                      abandonCost=abandonCost, wellData=wellData, recoverableVol=recoverableVol, maxNumOfProducingWells=maxNumOfProducingWells)
            else:
                monthlyEachWellProdVolTableTemp, monthlyFieldVolList, yearlyNetCashflowTemp, yearlyFieldRevenueList, yearlyFieldCapitalCostListTemp,\
                yearlyFieldRoyaltyListTemp, yearlyFieldBonusListTemp, yearlyFieldOpertCostListTemp, yearlyFieldAbandonCostList, yearlyTaxListTemp,\
                monthlyFieldRevenueList, monthlyFieldCapitalCostList, monthlyFieldRoyaltyList, monthlyFieldBonusList,\
                monthlyFieldOpertCostList, monthlyFieldAbandonCostList, NPVList[i], breakevenYearList[i], maxExposureList[i], AEE_yearlyNetCashflowTemp, AEE_NPVList[i] \
                = self.calculateOneResultCurrentPrice(currentPrice=currentPrice, taxRate=taxRate, royaltyRate=royaltyRate, rental=rental, bonus=bonus, yearlyCPI=yearlyCPI, monthlyCPI=monthlyCPI,
                                                      fixedOpertExpense=fixedOpertExpense, varOpertExpense=varOpertExpense, capitalCost=capitalCost, tangibleCapitalPerc=tangibleCapitalPerc,
                                                      abandonCost=abandonCost, wellData=wellData, recoverableVol=recoverableVol, maxNumOfProducingWells=maxNumOfProducingWells, succRate=succRate)
            yearlyNetCashflowTableAllSamplesCurrentPrice.append(yearlyNetCashflowTemp)
            AEE_yearlyNetCashflowTableAllSamplesCurrentPrice.append(AEE_yearlyNetCashflowTemp)
            yearlyNetCashflowTemp = [x/nSimulation for x in yearlyNetCashflowTemp]
            yearlyAverageNetCashflow = [x+y for x,y in izip_longest(yearlyAverageNetCashflow, yearlyNetCashflowTemp, fillvalue=0)]
            yearlyFieldRoyaltyListTemp = [x/nSimulation for x in yearlyFieldRoyaltyListTemp]
            yearlyRoyaltyCurrentPrice = [x+y for x,y in izip_longest(yearlyRoyaltyCurrentPrice, yearlyFieldRoyaltyListTemp, fillvalue=0)]
            yearlyFieldCapitalCostListTemp = [x/nSimulation for x in yearlyFieldCapitalCostListTemp]
            yearlyCapitalCurrentPrice = [x+y for x,y in izip_longest(yearlyCapitalCurrentPrice, yearlyFieldCapitalCostListTemp, fillvalue=0)]
            yearlyFieldOpertCostListTemp = [x/nSimulation for x in yearlyFieldOpertCostListTemp]
            yearlyOpertExpenseCurrentPrice = [x+y for x,y in izip_longest(yearlyOpertExpenseCurrentPrice, yearlyFieldOpertCostListTemp, fillvalue=0)]
            yearlyFieldBonusListTemp = [x/nSimulation for x in yearlyFieldBonusListTemp]
            yearlyBonusCurrentPrice = [x+y for x,y in izip_longest(yearlyBonusCurrentPrice, yearlyFieldBonusListTemp, fillvalue=0)]
            yearlyTaxListTemp = [x/nSimulation for x in yearlyTaxListTemp]
            yearlyIncomeTaxCurrentPrice = [x+y for x,y in izip_longest(yearlyIncomeTaxCurrentPrice, yearlyTaxListTemp, fillvalue=0)]
            yearlyRentalCurrentPrice = [rental for _ in xrange(len(yearlyAverageNetCashflow))]
            AEE_yearlyNetCashflowTemp = [x/nSimulation for x in AEE_yearlyNetCashflowTemp]
            AEE_yearlyAverageNetCashflow = [x+y for x,y in izip_longest(AEE_yearlyAverageNetCashflow, AEE_yearlyNetCashflowTemp, fillvalue=0)]

            for i, prodList in enumerate(monthlyEachWellProdVolTableTemp):
                prodList = [x/nSimulation for x in prodList]
                monthlyEachWellProdVolTable[i] = [x+y for x,y in izip_longest(prodList, monthlyEachWellProdVolTable[i], fillvalue=0)]

        breakevenYearList = [x for x in breakevenYearList if x>0]
        averageBreakevenYear = np.mean(breakevenYearList)

        return NPVList, yearlyAverageNetCashflow, averageBreakevenYear, yearlyRoyaltyCurrentPrice, yearlyCapitalCurrentPrice, yearlyOpertExpenseCurrentPrice, \
               yearlyBonusCurrentPrice, yearlyRentalCurrentPrice, yearlyIncomeTaxCurrentPrice, yearlyNetCashflowTableAllSamplesCurrentPrice, monthlyEachWellProdVolTable, \
               maxExposureList, \
               AEE_NPVList, AEE_yearlyAverageNetCashflow, AEE_yearlyNetCashflowTableAllSamplesCurrentPrice

    def calculateOneResultCurrentPrice(self, **kwargs):
        # unpack parameters from **kwargs
        currentPrice = kwargs.pop('currentPrice'); taxRate = kwargs.pop('taxRate'); royaltyRate = kwargs.pop('royaltyRate'); rental = kwargs.pop('rental'); bonus = kwargs.pop('bonus');
        yearlyCPI = kwargs.pop('yearlyCPI'); monthlyCPI = kwargs.pop('monthlyCPI'); fixedOpertExpense = kwargs.pop('fixedOpertExpense'); varOpertExpense = kwargs.pop('varOpertExpense');
        capitalCost = kwargs.pop('capitalCost'); tangibleCapitalPerc = kwargs.pop('tangibleCapitalPerc'); abandonCost = kwargs.pop('abandonCost'); wellData = kwargs.pop('wellData');
        recoverableVol = kwargs.pop('recoverableVol'); maxNumOfProducingWells = kwargs.pop('maxNumOfProducingWells')
        # if kwargs:
        #     raise TypeError('Unexpected **kwargs: %r' % kwargs)

        try:
            succRate = kwargs['succRate']
        except:
            succRate = 0

        depreciationLength = 25  # years
        taxLossCarryforwardYears = 7  # years
        developMonths = 6  # months
        averageWellVol = recoverableVol/maxNumOfProducingWells
        monthlyEachWellProdVolTable = []
        declineParameter = 0.7
        b = 1/2
        lastInstallMonth = 0
        installPlanList_installmonth_nWell = []

        def prodFunction(q0, b, declineParameter):
            def funct(t):
                return q0/(1+b*declineParameter*t)**(1/b)
            return funct

        # ELT of each well
        # returns a table monthlyEachWellProdVolTable, containing monthly production of each well
        for data in wellData:
            if succRate:
                installYear, installMonth, numPlannedWells, _ = data
            else:
                installYear, installMonth, numPlannedWells, succRate = data
                succRate /= 100
            installMonthTotal = installYear*12 + installMonth
            installPlanList_installmonth_nWell.append((installMonthTotal, numPlannedWells))
            lastInstallMonth = max(lastInstallMonth, installMonthTotal)
            firstoilMonth = installMonthTotal + developMonths
            for i in range(numPlannedWells):
                if random.uniform(0,1) >= succRate:
                    monthlyEachWellProdVolTable.append([0 for _ in range(installMonth)] + [-1])
                else:
                    monthlyEachWellProdVolTable.append([0 for _ in range(firstoilMonth)])
                    wellVol = averageWellVol * random.uniform(0.5, 1.5)
                    q0 = wellVol*declineParameter*(1-b)
                    funAtFly = prodFunction(q0, b, declineParameter)
                    i = 0
                    while True:
                        monthlyVol = funAtFly((i+0.5)/12) * (1/12)
                        monthlyRevenue = currentPrice * monthlyVol * (1 - royaltyRate)
                        monthlyOpertExpense = fixedOpertExpense/12 + varOpertExpense*monthlyVol
                        if monthlyRevenue > monthlyOpertExpense:
                            monthlyEachWellProdVolTable[-1].append(monthlyVol)
                            i += 1
                        else: break

        ### ELT of the field (all wells together), including rental of field
        # extend every well's production list to the same length
        maxMonth = max((lastInstallMonth+36), max([len(x) for x in monthlyEachWellProdVolTable]))
        for x in monthlyEachWellProdVolTable: x += [0 for _ in range(maxMonth - len(x))]
        maxYear = int(np.ceil(maxMonth/12))

        installYears = set()
        for x in wellData: installYears.add(x[0])
        monthlyFieldProdVolList = [0 for _ in range(maxMonth)]
        monthlyNumOfWorkingWells = [0 for _ in range(maxMonth)]
        
        # calculate producing volume and # of working wells of each month
        for i in range(maxMonth):
            for x in monthlyEachWellProdVolTable:
                monthlyFieldProdVolList[i] += x[i]
                monthlyNumOfWorkingWells[i] += np.sign(x[i])
        yearlyELTFlagList = [1 for _ in range(maxYear)]

        monthlyOpertCashflow = [0 for _ in range(maxMonth)]
        for i in range(maxMonth):
            monthlyRevenue = monthlyFieldProdVolList[i] * currentPrice * (1 - royaltyRate) * monthlyCPI**i
            monthlyOpertExpense = (monthlyNumOfWorkingWells[i]*fixedOpertExpense/12 + varOpertExpense*monthlyFieldProdVolList[i])\
                                  *monthlyCPI**i
            monthlyOpertCashflow[i] = monthlyRevenue - monthlyOpertExpense

        for i in range(maxYear):
            yearlyCashflowPreInvestment = sum(monthlyOpertCashflow[i*12:i*12+12]) - rental
            if yearlyCashflowPreInvestment < 0 and i in (set(range(maxYear)) - installYears):
                yearlyELTFlagList[i] = 0

        maxYear = sum(yearlyELTFlagList)
        maxMonth = maxYear*12

        ### convert ELT Flag from yearly to monthly
        monthlyELTFlagList = []
        for flag in yearlyELTFlagList:
            monthlyELTFlagList += [flag]*12

        # set monthly production volume to 0 when oil field is OFF (ELT Flag = 0)
        for i in range(len(monthlyEachWellProdVolTable)):
            for j in range(len(monthlyEachWellProdVolTable[0])):
                monthlyEachWellProdVolTable[i][j] *= monthlyELTFlagList[j]
        

        def wellVol2Cash(volList):
            firstOil = 1
            revenueTemp = [0 for _ in range(len(volList))]
            capitalCostTemp = copy.copy(revenueTemp)
            intangibleCapitalCostTemp = copy.copy(revenueTemp)
            tangibleCapitalCostTemp = copy.copy(revenueTemp)
            yearlyTangibleCapitalCostTemp = [0 for _ in range(maxYear)]
            yearlyTangibleCapitalDepreciationTemp = copy.copy(yearlyTangibleCapitalCostTemp)
            bonusTemp = copy.copy(revenueTemp)
            opertCostTemp = copy.copy(revenueTemp)
            royaltyTemp = copy.copy(revenueTemp)
            abandonCostTemp = copy.copy(revenueTemp)

            for i in range(len(volList)):
                # dry wells: negative cashflow in developing months
                if volList[i] < 0:
                    capitalCostTemp[i:i+developMonths] = [capitalCost[0]/12 for _ in range(developMonths)]
                    for i in range(len(capitalCostTemp)):
                        capitalCostTemp[i] = capitalCostTemp[i] * monthlyCPI**i
                    return revenueTemp, capitalCostTemp, yearlyTangibleCapitalDepreciationTemp, intangibleCapitalCostTemp, royaltyTemp, bonusTemp, opertCostTemp, abandonCostTemp

                # producing wells' first oil
                elif volList[i] > 0 and firstOil:
                    bonusTemp[i] = bonus
                    capitalCostTemp[i:i+12] = [capitalCost[0]/12 for _ in range(12)]
                    intangibleCapitalCostTemp[i:i+12] = [x*y for x,y in zip(capitalCostTemp[i:i+12], [1-tangibleCapitalPerc[0]]*12)]
                    capitalCostTemp[i+12:i+24] = [capitalCost[1]/12 for _ in range(12)]
                    intangibleCapitalCostTemp[i+12:i+24] = [x*y for x,y in zip(capitalCostTemp[i+12:i+24], [1-tangibleCapitalPerc[1]]*12)]
                    capitalCostTemp[i+24:i+36] = [capitalCost[2]/12 for _ in range(12)]
                    intangibleCapitalCostTemp[i+24:i+36] = [x*y for x,y in zip(capitalCostTemp[i+24:i+36], [1-tangibleCapitalPerc[2]]*12)]
                    opertCostTemp[i] = (fixedOpertExpense/12 + volList[i]*varOpertExpense) * monthlyCPI**i
                    revenueTemp[i] = volList[i]*currentPrice*monthlyCPI**i
                    royaltyTemp[i] = revenueTemp[i]*royaltyRate
                    firstoilMonth = i
                    firstOil = 0
                # producing wells' last oil
                elif volList[i] > 0 and (i == len(volList)-1 or volList[i+1] == 0):
                    opertCostTemp[i] = (fixedOpertExpense/12 + volList[i]*varOpertExpense) * monthlyCPI**i
                    revenueTemp[i] = volList[i]*currentPrice*monthlyCPI**i
                    royaltyTemp[i] = revenueTemp[i]*royaltyRate
                    abandonCostTemp[i] = abandonCost * monthlyCPI**i
                # producing wells after first oil
                elif volList[i] > 0:
                    opertCostTemp[i] = (fixedOpertExpense/12 + volList[i]*varOpertExpense) * monthlyCPI**i
                    revenueTemp[i] = volList[i]*currentPrice*monthlyCPI**i
                    royaltyTemp[i] = revenueTemp[i]*royaltyRate

            tangibleCapitalCostTemp = copy.copy(capitalCostTemp)
            for i in range(len(capitalCostTemp)):
                capitalCostTemp[i] = capitalCostTemp[i] * monthlyCPI**i
                intangibleCapitalCostTemp[i] = intangibleCapitalCostTemp[i] * monthlyCPI**i
                tangibleCapitalCostTemp[i] = capitalCostTemp[i] - intangibleCapitalCostTemp[i]

            for i in range(maxYear):
                yearlyTangibleCapitalCostTemp[i] = sum(tangibleCapitalCostTemp[i*12:i*12+12])  

            for i,x in enumerate(yearlyTangibleCapitalCostTemp):
                if x > 0:
                    listTemp = [0] * i + [x/depreciationLength] * depreciationLength
                    yearlyTangibleCapitalDepreciationTemp = [a+b for a,b in zip(yearlyTangibleCapitalDepreciationTemp+[0]*(len(listTemp)-len(yearlyTangibleCapitalDepreciationTemp)), listTemp+[0]*(len(yearlyTangibleCapitalDepreciationTemp)-len(listTemp)))]

            return revenueTemp, capitalCostTemp, yearlyTangibleCapitalDepreciationTemp, intangibleCapitalCostTemp, royaltyTemp, bonusTemp, opertCostTemp, abandonCostTemp


        monthlyFieldRevenueList = [0 for _ in range(maxMonth)]
        monthlyFieldVolList = copy.copy(monthlyFieldRevenueList)
        monthlyFieldCapitalCostList = copy.copy(monthlyFieldRevenueList)
        monthlyFieldIntangibleCapitalCostList = copy.copy(monthlyFieldRevenueList)
        monthlyFieldRoyaltyList = copy.copy(monthlyFieldRevenueList)
        monthlyFieldBonusList = copy.copy(monthlyFieldRevenueList)
        monthlyFieldOpertCostList = copy.copy(monthlyFieldRevenueList)
        monthlyFieldAbandonCostList = copy.copy(monthlyFieldRevenueList)

        yearlyFieldRevenueList = [0 for _ in range(maxYear)]
        yearlyFieldProdVolList = copy.copy(yearlyFieldRevenueList)
        yearlyFieldCapitalCostList = copy.copy(yearlyFieldRevenueList)
        yearlyFieldIntangibleCapitalCostList = copy.copy(yearlyFieldRevenueList)
        yearlyTangibleCapitalDepreciationList = copy.copy(yearlyFieldRevenueList)
        yearlyFieldRoyaltyList = copy.copy(yearlyFieldRevenueList)
        yearlyFieldBonusList = copy.copy(yearlyFieldRevenueList)
        yearlyFieldOpertCostList = copy.copy(yearlyFieldRevenueList)
        yearlyFieldAbandonCostList = copy.copy(yearlyFieldRevenueList)
        yearlyNetCashflowList = copy.copy(yearlyFieldRevenueList)
        yearlyRealNetCashflowList = copy.copy(yearlyFieldRevenueList)
        yearlyCost = copy.copy(yearlyFieldRevenueList)
        yearlyTaxList = copy.copy(yearlyFieldRevenueList)
        accumulatedTaxLossList = copy.copy(yearlyFieldRevenueList)
        yearlyCostList = copy.copy(yearlyFieldRevenueList)

        exposure = 0
        maxExposure = 0

        # loop each well
        for monthlyVolList in monthlyEachWellProdVolTable:
            revenueList, capitalCostList, yearlyTangibleCapitalDepreciationListTemp, intangibleCapitalCostList, \
            royaltyList, bonusList, opertCostList, abandonCostList = wellVol2Cash(monthlyVolList)

            monthlyFieldRevenueList = [x+y for x,y in zip(monthlyFieldRevenueList, revenueList)]
            monthlyFieldVolList = [x+y for x,y in zip(monthlyFieldVolList, monthlyVolList)]
            monthlyFieldCapitalCostList = [x+y for x,y in zip(monthlyFieldCapitalCostList, capitalCostList)]
            monthlyFieldIntangibleCapitalCostList = [x+y for x,y in zip(monthlyFieldIntangibleCapitalCostList, intangibleCapitalCostList)]
            monthlyFieldRoyaltyList = [x+y for x,y in zip(monthlyFieldRoyaltyList, royaltyList)]
            monthlyFieldBonusList = [x+y for x,y in zip(monthlyFieldBonusList, bonusList)]
            monthlyFieldOpertCostList = [x+y for x,y in zip(monthlyFieldOpertCostList, opertCostList)]
            monthlyFieldAbandonCostList = [x+y for x,y in zip(monthlyFieldAbandonCostList, abandonCostList)]
            yearlyTangibleCapitalDepreciationList = [x+y for x,y in zip(yearlyTangibleCapitalDepreciationList, yearlyTangibleCapitalDepreciationListTemp)]
        
        # convert monthly data into yearly data
        for i in range(maxYear):
            yearlyFieldRevenueList[i] = sum(monthlyFieldRevenueList[i*12:i*12+12])
            yearlyFieldCapitalCostList[i] = sum(monthlyFieldCapitalCostList[i*12:i*12+12])
            yearlyFieldIntangibleCapitalCostList[i] = sum(monthlyFieldIntangibleCapitalCostList[i*12:i*12+12])
            yearlyFieldRoyaltyList[i] = sum(monthlyFieldRoyaltyList[i*12:i*12+12])
            yearlyFieldBonusList[i] = sum(monthlyFieldBonusList[i*12:i*12+12])
            yearlyFieldOpertCostList[i] = sum(monthlyFieldOpertCostList[i*12:i*12+12])
            yearlyFieldAbandonCostList[i] = sum(monthlyFieldAbandonCostList[i*12:i*12+12])
            yearlyFieldProdVolList[i] = sum(monthlyFieldProdVolList[i*12:i*12+12])

            yearlyTaxableProfit = yearlyFieldRevenueList[i] - yearlyFieldIntangibleCapitalCostList[i] - yearlyFieldRoyaltyList[i] - yearlyFieldBonusList[i]\
                                     - yearlyFieldOpertCostList[i] - yearlyFieldAbandonCostList[i] - yearlyTangibleCapitalDepreciationList[i]\
                                     - sum(accumulatedTaxLossList[max(0,i-taxLossCarryforwardYears) : i])

            if yearlyTaxableProfit > 0:
                yearlyTaxList[i] = taxRate * yearlyTaxableProfit
                accumulatedTaxLossList[:i] = [0]*len(accumulatedTaxLossList[:i])
            else:
                accumulatedTaxLossList[i] = yearlyTaxableProfit

            yearlyNetCashflowList[i] = yearlyFieldRevenueList[i] - yearlyFieldCapitalCostList[i] - yearlyFieldRoyaltyList[i] - yearlyFieldBonusList[i]\
                                  - yearlyFieldOpertCostList[i] - yearlyFieldAbandonCostList[i] - yearlyTaxList[i] - rental
            yearlyRealNetCashflowList[i] = yearlyNetCashflowList[i] / yearlyCPI**i
            # track maximum exposure
            exposure += yearlyRealNetCashflowList[i]
            if exposure < maxExposure:
                maxExposure = exposure
            yearlyCostList[i] = (yearlyFieldBonusList[i] + yearlyFieldOpertCostList[i] + rental) / (1-royaltyRate)

        # Prepare General Results
        # yearlyCostPerBarrelList = [a/b for a,b in zip(yearlyCostList, yearlyFieldProdVolList)]
        NPV = sum(yearlyRealNetCashflowList)
        for i in xrange(len(yearlyRealNetCashflowList)):
            if yearlyRealNetCashflowList[0] > 0:
                breakevenYear = 1
                break
            elif sum(yearlyRealNetCashflowList[:i]) < 0 and sum(yearlyRealNetCashflowList[:i+1]) >= 0:
                breakevenYear = i+1
                break
        else:
            breakevenYear = -1

        # Prepare AE&E Results
        minAEEShare = .1
        AEE_SharePercOfEachWell = [minAEEShare for _ in xrange(sum(np.asarray(wellData)[:,2]))]
        AEE_ShareIncreaseStep = .025
        dailyProdVolThreashold = 30
        dailyProdVolThreasholdIncreaseStep = 20

        currentWell = installPlanList_installmonth_nWell[0][1]
        for i in xrange(1, len(installPlanList_installmonth_nWell)):
            numFormerBatchWells = installPlanList_installmonth_nWell[i-1][1]
            numCurrentBatchWells = installPlanList_installmonth_nWell[i][1]
            currentMonth = installPlanList_installmonth_nWell[i][0]

            meanProdVolAtTheMoment = np.mean(np.asarray(monthlyEachWellProdVolTable)[(currentWell - numFormerBatchWells):currentWell, currentMonth])

            if meanProdVolAtTheMoment > (dailyProdVolThreashold*30) and AEE_SharePercOfEachWell[currentWell-1] <= .2:
                AEE_SharePercOfEachWell[currentWell:] = [(AEE_SharePercOfEachWell[currentWell-1] + AEE_ShareIncreaseStep) for _ in xrange(len(AEE_SharePercOfEachWell[currentWell:]))]
                dailyProdVolThreashold += dailyProdVolThreasholdIncreaseStep
            currentWell += numCurrentBatchWells

        # AEE_monthlyProdVolShareTable has total months Columns; nWells Rows (the same dimension as monthlyEachWellProdVolTable).
        AEE_monthlyProdVolShareTable = [[AEESharePerc*x for x in monthlyProdList] for AEESharePerc, monthlyProdList in zip(AEE_SharePercOfEachWell, monthlyEachWellProdVolTable)]
        # Calculate AEE share of Profit each year
        AEE_yearlyNetCashflowList = [0 for x in xrange(len(yearlyNetCashflowList))]
        for i, profit in enumerate(yearlyNetCashflowList):
            if profit > 0:
                try:
                    AEE_volume = sum(sum(np.asarray(AEE_monthlyProdVolShareTable)[:, i*12:i*12+12]))
                    total_volume = sum(sum(np.asarray(monthlyEachWellProdVolTable)[:, i*12:i*12+12]))
                    AEESharePerc = AEE_volume / total_volume
                except ZeroDivisionError:
                    AEESharePerc = 0
                AEE_yearlyNetCashflowList[i] = AEESharePerc * profit
        AEE_NPV = np.npv(yearlyCPI-1, AEE_yearlyNetCashflowList)

        return monthlyEachWellProdVolTable, monthlyFieldVolList, yearlyNetCashflowList, yearlyFieldRevenueList, yearlyFieldCapitalCostList,\
               yearlyFieldRoyaltyList, yearlyFieldBonusList, yearlyFieldOpertCostList, yearlyFieldAbandonCostList, yearlyTaxList,\
               monthlyFieldRevenueList, monthlyFieldCapitalCostList, monthlyFieldRoyaltyList, monthlyFieldBonusList,\
               monthlyFieldOpertCostList, monthlyFieldAbandonCostList, NPV, breakevenYear, maxExposure, AEE_yearlyNetCashflowList, AEE_NPV


class ParameterBox(QFrame):
    def __init__(self, parent=None):
        super(ParameterBox, self).__init__(parent)
        self.initParameterBox()
        self.setStyleSheet('''
            QFrame{
                background: #d5dbe3;
                border: None
            }

            QGroupBox{
                background:transparent;
                border: 2px solid gray;
                border-radius: 5px;
                margin-top: 23px;
            }

            QGroupBox::title{
                color: #4a4a4a;
                subcontrol-origin: margin;
                left: 12px;
                padding: 17px 0px 0px 0px;
            }
        ''')

    def initParameterBox(self):
        self.scrollArea = QScrollArea(self)
        # self.scrollArea.setStyleSheet('''background: transparent''')
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QFrame(self.scrollArea)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        subLayout = QVBoxLayout(self.scrollAreaWidgetContents)

        layout = QGridLayout(self)
        self.setLayout(layout)

        layout.addWidget(self.scrollArea)
        self.setLayout(layout)

        nc = 4  # column number
        # set default values
        currentPrice = 50
        taxRate = 40
        royaltyRate = 17.5
        rental = 1  # million $
        bonus = 0.1 # million $
        CPI = 3.0
        declineRate = 0.8
        capitalYear1 = 1.0
        capitalYear2 = 1.0
        capitalYear3 = 1.0
        tangibleCapitalPerc1 = 70
        tangibleCapitalPerc2 = 20
        tangibleCapitalPerc3 = 15
        fixedOpertExpense = 200000
        varOpertExpense = 3
        abandonCost = 100000

        recoveryRate = 15.0
        provenReserves = 20
        probableReserves = 20
        possibleReserves = 5
        maxNumOfProducingWells = 20

        # [year, # of wells, succ rate]
        self.wellData = [[0,0,3,75],[0,8,3,75],[1,4,3,75]]

        #####     Economic Parameters     #####
        groupBox_Eco = QGroupBox('Eonomic Parameters')
        box = QGridLayout()
        groupBox_Eco.setLayout(box)
        subLayout.addWidget(groupBox_Eco)

        # Oil Price
        self.OilPriceSpinbox = QDoubleSpinBox()
        self.OilPriceSpinbox.setMaximum(200)
        self.OilPriceSpinbox.setPrefix("$ ")
        self.OilPriceSpinbox.setValue(currentPrice)
        self.OilPriceSpinbox.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        box.addLayout(HBox(QLabel('Oil Price:'),self.OilPriceSpinbox),0,0)

        # CPI
        self.CPISpinbox = QDoubleSpinBox()
        self.CPISpinbox.setSuffix(" %")
        self.CPISpinbox.setValue(CPI)
        self.CPISpinbox.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        box.addLayout(HBox(QLabel('CPI Growth/yr:'),self.CPISpinbox),0,1)

        # Tax Rate
        self.taxRateSpinbox = QDoubleSpinBox()
        self.taxRateSpinbox.setSuffix(" %")
        self.taxRateSpinbox.setValue(taxRate)
        self.taxRateSpinbox.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        box.addLayout(HBox(QLabel('Tax Rate:'),self.taxRateSpinbox),1,0)



        #####     Contract Info     #####
        groupBox_Contract = QGroupBox('Contract Information')
        box = QGridLayout()
        groupBox_Contract.setLayout(box)
        subLayout.addWidget(groupBox_Contract)

        # Royalty Rate
        self.royaltyRateSpinbox = QDoubleSpinBox()
        self.royaltyRateSpinbox.setSingleStep(0.1)
        self.royaltyRateSpinbox.setSuffix(" %")
        self.royaltyRateSpinbox.setValue(royaltyRate)
        self.royaltyRateSpinbox.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        box.addLayout(HBox(QLabel("Royalty Rate:"), self.royaltyRateSpinbox),0,0)

        # rental
        self.rentalSpinbox = QDoubleSpinBox()
        self.rentalSpinbox.setSingleStep(0.1)
        self.rentalSpinbox.setPrefix("$mm ")
        self.rentalSpinbox.setValue(rental)
        self.rentalSpinbox.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        box.addLayout(HBox(QLabel("Rental/yr/well:"), self.rentalSpinbox),0,1)

        # bonus of first production
        self.bonusSpinbox = QDoubleSpinBox()
        self.bonusSpinbox.setSingleStep(0.02)
        self.bonusSpinbox.setPrefix("$mm ")
        self.bonusSpinbox.setValue(bonus)
        self.bonusSpinbox.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        box.addLayout(HBox(QLabel("Bonus of 1st oil:"), self.bonusSpinbox),1,0)



        #####     Expense Parameters     #####
        groupBox_Expense = QGroupBox('Expense Parameters of Each Well')
        box = QGridLayout()
        groupBox_Expense.setLayout(box)
        subLayout.addWidget(groupBox_Expense)

        # Capital Expense
        box.addWidget(QLabel('Year 1'),0,1)
        box.addWidget(QLabel('Year 2'),0,2)
        box.addWidget(QLabel('Year 3'),0,3)

        box.addWidget(QLabel("Capital Expense:"),1,0)
        # 1
        self.capitalCostSpinbox1 = QDoubleSpinBox()
        self.capitalCostSpinbox1.setSingleStep(0.1)
        self.capitalCostSpinbox1.setPrefix("$mm ")
        self.capitalCostSpinbox1.setValue(capitalYear1)
        self.capitalCostSpinbox1.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        # 2
        self.capitalCostSpinbox2 = QDoubleSpinBox()
        self.capitalCostSpinbox2.setSingleStep(0.1)
        self.capitalCostSpinbox2.setPrefix("$mm ")
        self.capitalCostSpinbox2.setValue(capitalYear2)
        self.capitalCostSpinbox2.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        # 3
        self.capitalCostSpinbox3 = QDoubleSpinBox()
        self.capitalCostSpinbox3.setSingleStep(0.1)
        self.capitalCostSpinbox3.setPrefix("$mm ")
        self.capitalCostSpinbox3.setValue(capitalYear3)
        self.capitalCostSpinbox3.setAlignment(Qt.AlignRight|Qt.AlignVCenter)

        box.addWidget(self.capitalCostSpinbox1,1,1)
        box.addWidget(self.capitalCostSpinbox2,1,2)
        box.addWidget(self.capitalCostSpinbox3,1,3)

        # percentage
        box.addWidget(QLabel("Tangible Exp:"),2,0)
        # 1
        self.tangibleCapitalPercSpinbox1 = QDoubleSpinBox()
        self.tangibleCapitalPercSpinbox1.setSuffix(" %")
        self.tangibleCapitalPercSpinbox1.setValue(tangibleCapitalPerc1)
        self.tangibleCapitalPercSpinbox1.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        # 2
        self.tangibleCapitalPercSpinbox2 = QDoubleSpinBox()
        self.tangibleCapitalPercSpinbox2.setSuffix(" %")
        self.tangibleCapitalPercSpinbox2.setValue(tangibleCapitalPerc2)
        self.tangibleCapitalPercSpinbox2.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        # 3
        self.tangibleCapitalPercSpinbox3 = QDoubleSpinBox()
        self.tangibleCapitalPercSpinbox3.setSuffix(" %")
        self.tangibleCapitalPercSpinbox3.setValue(tangibleCapitalPerc3)
        self.tangibleCapitalPercSpinbox3.setAlignment(Qt.AlignRight|Qt.AlignVCenter)

        box.addWidget(self.tangibleCapitalPercSpinbox1,2,1)
        box.addWidget(self.tangibleCapitalPercSpinbox2,2,2)
        box.addWidget(self.tangibleCapitalPercSpinbox3,2,3)

        # Line
        line = QFrame()
        line.setFrameStyle(QFrame.HLine)
        line.setStyleSheet('''border: 0px; background-color: #ededed''')
        box.addWidget(line,box.rowCount(),0,1,box.columnCount())

        # Fixed Operating Cost
        self.fixedOpertExpenseSpinbox = QDoubleSpinBox()
        self.fixedOpertExpenseSpinbox.setMaximum(10000000)
        self.fixedOpertExpenseSpinbox.setPrefix("$ ")
        self.fixedOpertExpenseSpinbox.setValue(fixedOpertExpense)
        self.fixedOpertExpenseSpinbox.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        box.addLayout(HBox(QLabel("Fixed Oprt Cost/yr:"), self.fixedOpertExpenseSpinbox), 4,0,1,2)

        # Variable Operating Cost
        self.varOpertExpenseSpinbox = QDoubleSpinBox()
        self.varOpertExpenseSpinbox.setPrefix("$ ")
        self.varOpertExpenseSpinbox.setValue(varOpertExpense)
        self.varOpertExpenseSpinbox.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        box.addLayout(HBox(QLabel("Var. Oprt Cost/bbl:"), self.varOpertExpenseSpinbox), 4,2,1,2)

        # Abandon Cost
        self.abandonCostSpinbox = QDoubleSpinBox()
        self.abandonCostSpinbox.setMaximum(10000000)
        self.abandonCostSpinbox.setPrefix("$ ")
        self.abandonCostSpinbox.setValue(abandonCost)
        self.abandonCostSpinbox.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        box.addLayout(HBox(QLabel("Abandon Cost/well:"), self.abandonCostSpinbox), 5,0,1,2)



        #####     Field Parameters     #####
        groupBox_Field = QGroupBox('Field Parameters')
        box = QGridLayout()
        groupBox_Field.setLayout(box)
        subLayout.addWidget(groupBox_Field)

        # Recovery Rate
        self.recoveryRateSpinbox = QDoubleSpinBox()
        self.recoveryRateSpinbox.setSuffix(" %")
        self.recoveryRateSpinbox.setValue(recoveryRate)
        self.recoveryRateSpinbox.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        box.addLayout(HBox(QLabel('Recovery Rate:'), self.recoveryRateSpinbox), 0,0)

        # maximum # of wells in this field
        self.maxNWellsSpinbox = QSpinBox()
        self.maxNWellsSpinbox.setValue(maxNumOfProducingWells)
        self.maxNWellsSpinbox.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        box.addLayout(HBox(QLabel('max # of wells:'), self.maxNWellsSpinbox), 0,1)

        # Proven Reserves 1P
        self.provenReservesSpinbox = QDoubleSpinBox()
        self.provenReservesSpinbox.setSuffix(" mm bbl")
        self.provenReservesSpinbox.setValue(provenReserves)
        self.provenReservesSpinbox.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        box.addLayout(HBox(QLabel('Proven Reserves:'), self.provenReservesSpinbox), 1,0)

        # Probable Reserves 2P
        self.probableReservesSpinbox = QDoubleSpinBox()
        self.probableReservesSpinbox.setSuffix(" mm bbl")
        self.probableReservesSpinbox.setValue(probableReserves)
        self.probableReservesSpinbox.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        box.addLayout(HBox(QLabel('Probable Reserves:'), self.probableReservesSpinbox), 1,1)

        # Possible Reserves 3P
        self.possibleReservesSpinbox = QDoubleSpinBox()
        self.possibleReservesSpinbox.setSuffix(" mm bbl")
        self.possibleReservesSpinbox.setValue(possibleReserves)
        self.possibleReservesSpinbox.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        box.addLayout(HBox(QLabel('Possible Reserves:'), self.possibleReservesSpinbox), 2,0)



        #####     Exploring Plan     #####
        groupBox_Plan = QGroupBox('Exploring Plan')
        box = QGridLayout()
        groupBox_Plan.setLayout(box)
        subLayout.addWidget(groupBox_Plan)

        # ButtonBox
        addwellButton = QPushButton("Add Wells...")
        # addwellButton.setStyleSheet(
        #     '''border: 20px solid black;
        #     border-radius: 10px;
        #     background-color: rgb(255, 255, 255);''')
        deletewellButton = QPushButton("Delte Wells...")
        addwellButton.clicked.connect(self.addWell)
        deletewellButton.clicked.connect(self.deleteWell)
        box.addLayout(HBox(addwellButton, deletewellButton),0,0)

        # Table
        headers = ["Year of Install.","Month of Install.","# of Wells","Succ. Rate %"]
        self.tableWidget = QTableWidget()
        self.tableWidget.horizontalHeader().setStretchLastSection(True)
        self.tableWidget.setRowCount(len(self.wellData))
        self.tableWidget.setColumnCount(len(headers))
        self.tableWidget.setHorizontalHeaderLabels(headers)
        box.addWidget(self.tableWidget)
        
        for irow, well in enumerate(self.wellData):
            for icol, wellValue in enumerate(well):
                item = QTableWidgetItem("%d" %self.wellData[irow][icol])
                item.setTextAlignment(Qt.AlignHCenter|Qt.AlignVCenter)
                self.tableWidget.setItem(irow, icol, item)

        self.connect(self.tableWidget, SIGNAL("itemChanged(QTableWidgetItem*)"), self.tableItemChanged)

    def addWell(self):
        irow = self.tableWidget.currentRow()
        if irow>=0:
            self.tableWidget.insertRow(irow+1)
            self.wellData.insert(irow+1,[0,0,0,0])
        else:
            self.tableWidget.insertRow(self.tableWidget.rowCount())
            self.wellData.insert(self.tableWidget.rowCount(),[0,0,0,0])

    def deleteWell(self):
        if not self.tableWidget.rowCount()<=1:
            irow = self.tableWidget.currentRow()
            if not irow < 0:
                self.tableWidget.removeRow(irow)
                self.wellData.pop(irow)

    def tableItemChanged(self, item):
        irow = self.tableWidget.currentRow()
        icol = self.tableWidget.currentColumn()
        self.wellData[irow][icol] = item.text().toInt()[0]
        item.setTextAlignment(Qt.AlignHCenter|Qt.AlignVCenter)

    def getParameters(self):
        highestPrice = 101
        lowestPrice = 30
        currentPrice = self.OilPriceSpinbox.value()
        priceList = [int(currentPrice*0.5), int(currentPrice*0.75), currentPrice, int(currentPrice*1.5), int(currentPrice*2)]
        taxRate = self.taxRateSpinbox.value()/100
        royaltyRate = self.royaltyRateSpinbox.value()/100
        rental = self.rentalSpinbox.value()*1000000
        bonus = self.bonusSpinbox.value()*1000000
        yearlyCPI = 1 + self.CPISpinbox.value()/100
        monthlyCPI = yearlyCPI**(1/12)
        fixedOpertExpense = self.fixedOpertExpenseSpinbox.value()
        varOpertExpense = self.varOpertExpenseSpinbox.value()
        abandonCost = self.abandonCostSpinbox.value()
        capitalCost = []
        tangibleCapitalPerc = []
        capitalCost[:] = [self.capitalCostSpinbox1.value()*1000000, self.capitalCostSpinbox2.value()*1000000,\
                        self.capitalCostSpinbox3.value()*1000000]
        tangibleCapitalPerc[:] = [self.tangibleCapitalPercSpinbox1.value()/100,\
                        self.tangibleCapitalPercSpinbox2.value()/100, self.tangibleCapitalPercSpinbox3.value()/100]
        recoverableVol = self.recoveryRateSpinbox.value()/100 * (self.provenReservesSpinbox.value()*0.9 + \
                         self.probableReservesSpinbox.value()*0.5 + self.possibleReservesSpinbox.value()*0.1) * 1000000
        maxNumOfProducingWells = self.maxNWellsSpinbox.value()

        return(currentPrice, priceList, taxRate, royaltyRate, rental, bonus, yearlyCPI, monthlyCPI, fixedOpertExpense, varOpertExpense,
               capitalCost, tangibleCapitalPerc, abandonCost, self.wellData, lowestPrice, highestPrice, recoverableVol, maxNumOfProducingWells)


class ResultTab1(QFrame):
    def __init__(self, parent=None):
        super(ResultTab1, self).__init__(parent)
        # self.setMinim2umWidth(700)
        # self.setMinimumHeight(700)
        self.initResultTab()

    def initResultTab(self):
        self.setStyleSheet(
            '''
                QFrame{
                background-color: white;
                }

                QWidget{
                background-color: white;
                border: 0px;
                }

                QLabel{
                color: #4a4a4a;
                font: bold 20px;
                }
            ''')


        layout = QGridLayout(self)

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget(self.scrollArea)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        subLayout0 = QVBoxLayout(self.scrollAreaWidgetContents)
        subLayout0.setAlignment(Qt.AlignHCenter)

        self.NPVBox = boxResult('Net Present Value')
        self.MaxExpoBox = boxResult('Max. Capital Exposure')
        self.BreakevenBox = boxResult('Average Breakeven Year')
        topResultBox = QHBoxLayout()
        topResultBox.addWidget(self.NPVBox)
        topResultBox.addSpacing(20)
        topResultBox.addWidget(self.MaxExpoBox)
        topResultBox.addSpacing(20)
        topResultBox.addWidget(self.BreakevenBox)

        self.oilProdChart = oilProdChart()
        self.NPVHist = NPVHist()
        self.AllNetCashflowChart = AllNetCashflowChart()
        self.IntervalChart = IntervalChart([0,0,0,0,0])
        self.outflowBarchart = stackedBarchart()
        self.heatmap = Heatmap()

        subLayout0.addSpacing(0)
        subLayout0.addWidget(QLabel('Summary:'))
        subLayout0.addSpacing(10)
        subLayout0.addLayout(topResultBox)
        subLayout0.addSpacing(30)
        subLayout0.addWidget(self.oilProdChart)
        subLayout0.addWidget(self.NPVHist)
        subLayout0.addWidget(self.AllNetCashflowChart)
        subLayout0.addWidget(self.outflowBarchart)


        subLayout0.addWidget(QLabel('Sensitivity Analysis:'))
        subLayout0.addWidget(self.IntervalChart)
        subLayout0.addWidget(self.heatmap)
        
        layout.addWidget(self.scrollArea)
        self.setLayout(layout)

    def updateGraphs(self, NPVTable_diffPrices, IntervalData, priceList, priceListCross, succRateList, NPVTableCross, averageBreakevenYear, yearlyRoyaltyCurrentPrice, yearlyCapitalCurrentPrice, yearlyOpertExpenseCurrentPrice,
                     yearlyBonusCurrentPrice, yearlyRentalCurrentPrice, yearlyIncomeTaxCurrentPrice, yearlyNetCashflowTableAllSamplesCurrentPrice, monthlyEachWellProdVolTable, maxExposureList):
        lowerLimit = np.percentile(np.array(NPVTable_diffPrices[2]), 5)/1000000
        upperLimit = np.percentile(np.array(NPVTable_diffPrices[2]), 95)/1000000
        maxExposure = -np.mean(maxExposureList)/1000000
        NPVdifferentPrices = [np.mean(x)/1000000 for x in NPVTable_diffPrices]

        self.NPVBox.updatePlot('$ %0.2f million' % (np.mean(NPVTable_diffPrices[2])/1000000))
        self.MaxExpoBox.updatePlot('$ %.2f million' % maxExposure)
        self.BreakevenBox.updatePlot('%0.1f years' % averageBreakevenYear)

        self.oilProdChart.updatePlot(monthlyEachWellProdVolTable = monthlyEachWellProdVolTable)
        self.NPVHist.updatePlot(NPVdata = NPVTable_diffPrices[2], lowerLimit=lowerLimit, upperLimit=upperLimit)
        self.AllNetCashflowChart.updatePlot(yearlyAverageNetCashflow=IntervalData[2], yearlyNetCashflowTableAllSamplesCurrentPrice=yearlyNetCashflowTableAllSamplesCurrentPrice)
        self.IntervalChart.updatePlot(priceList, IntervalData, NPVdifferentPrices)
        self.outflowBarchart.updatePlot(yearlyRoyaltyCurrentPrice, yearlyCapitalCurrentPrice, yearlyOpertExpenseCurrentPrice,
                                        yearlyBonusCurrentPrice, yearlyRentalCurrentPrice, yearlyIncomeTaxCurrentPrice)
        self.heatmap.updatePlot(priceListCross, succRateList, NPVTableCross, contour=1)


class ResultTab2(QFrame):
    def __init__(self, parent=None):
        super(ResultTab2, self).__init__(parent)
        # self.setMinim2umWidth(700)
        # self.setMinimumHeight(700)
        self.initResultTab()

    def initResultTab(self):
        self.setStyleSheet(
            '''
                QFrame{
                background-color: white;
                }

                QWidget{
                background-color: white;
                border: 0px;
                }

                QLabel{
                color: #4a4a4a;
                font: bold 20px;
                }
            ''')


        layout = QGridLayout(self)

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaWidgetContents = QWidget(self.scrollArea)
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        subLayout0 = QVBoxLayout(self.scrollAreaWidgetContents)
        subLayout0.setAlignment(Qt.AlignHCenter)

        self.NPVBox = boxResult('Net Present Value')
        self.AEEShareBox = boxResult('AE&E\'s Share of Profit\n\n(100% when field NPV\n is negative)')
        # self.BreakevenBox = boxResult('Average Breakeven Year')
        topResultBox = QHBoxLayout()
        topResultBox.addWidget(self.NPVBox)
        topResultBox.addSpacing(20)
        topResultBox.addWidget(self.AEEShareBox)
        topResultBox.addSpacing(20)
        # topResultBox.addWidget(self.BreakevenBox)

        self.NPVHist = NPVHist()
        self.AllNetCashflowChart = AllNetCashflowChart()
        self.IntervalChart = IntervalChart([0,0,0,0,0])
        self.heatmap = Heatmap()

        subLayout0.addSpacing(0)
        subLayout0.addWidget(QLabel('Summary:'))
        subLayout0.addSpacing(10)
        subLayout0.addLayout(topResultBox)
        subLayout0.addSpacing(30)
        subLayout0.addWidget(self.NPVHist)
        subLayout0.addWidget(self.AllNetCashflowChart)


        subLayout0.addWidget(QLabel('Sensitivity Analysis:'))
        subLayout0.addWidget(self.IntervalChart)
        subLayout0.addWidget(self.heatmap)
        
        layout.addWidget(self.scrollArea)
        self.setLayout(layout)

    def updateGraphs(self, NPVTable_diffPrices, priceList, priceListCross, succRateList, AEE_NPVTableResult_diffPrices, AEE_yearlyNetCashflowTableResult, AEE_TableCross, AEE_yearlyNetCashflowTableAllSamplesCurrentPrice):
        lowerLimit = np.percentile(np.array(AEE_NPVTableResult_diffPrices[2]), 5)/1000000
        upperLimit = np.percentile(np.array(AEE_NPVTableResult_diffPrices[2]), 95)/1000000
        totalNPV = np.mean(NPVTable_diffPrices[2])/1000000
        AEENPV = np.mean(AEE_NPVTableResult_diffPrices[2])/1000000
        if totalNPV > 0:
            AEEProfitSharePerc = AEENPV/totalNPV*100
        else:
            AEEProfitSharePerc = 100

        AEE_NPVdifferentPrices = [np.mean(x)/1000000 for x in AEE_NPVTableResult_diffPrices]

        self.NPVBox.updatePlot('$ %0.2f million' % AEENPV)
        self.AEEShareBox.updatePlot('%.2f %%' % AEEProfitSharePerc)
        # self.BreakevenBox.updatePlot('%0.1f years' % averageBreakevenYear)

        self.NPVHist.updatePlot(NPVdata = AEE_NPVTableResult_diffPrices[2], lowerLimit=lowerLimit, upperLimit=upperLimit)
        self.AllNetCashflowChart.updatePlot(yearlyAverageNetCashflow=AEE_yearlyNetCashflowTableResult[2], yearlyNetCashflowTableAllSamplesCurrentPrice=AEE_yearlyNetCashflowTableAllSamplesCurrentPrice)
        self.IntervalChart.updatePlot(priceList, AEE_yearlyNetCashflowTableResult, AEE_NPVdifferentPrices)
        self.heatmap.updatePlot(priceListCross, succRateList, AEE_TableCross, contour=0)



class HBox(QHBoxLayout):
    def __init__(self, *args):
        super(HBox, self).__init__(None)
        for w in args:
            self.addWidget(w)


if __name__ == "__main__":
    app = QApplication([])


    # # Create and display the splash screen
    # splash_pix = QPixmap('/Users/yunyunzhang/Desktop/PythonTest/quokka.jpg')
    # splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    # splash.setMask(splash_pix.mask())
    # splash.show()
    # # Simulate something that takes time
    # time.sleep(1)
    # win = FModel()
    # win.show()
    # splash.finish(win)


    win = FModel()
    win.show()


    sys.exit(app.exec_())




