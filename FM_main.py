# -*- coding:utf-8 -*-
from __future__ import division

import sys
import os
import random
import copy
import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import mlab as ml
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib as mpl

import seaborn as sns

from PyQt4.QtCore import *
from PyQt4.QtGui import *

mpl.rc("figure", facecolor="white")

DPI_default = 93

def irrF(values):
    res = np.roots(values[::-1])
    mask = (res.imag == 0) & (res.real > 0)
    if res.size == 0:
        return np.nan
    res = res[mask].real
    # NPV(rate) = 0 can have more than one solution so we return
    # only the solution closest to zero.
    rate = 1.0/res - 1
    # rate = rate.item(np.argmin(np.abs(rate)))
    # rate = rate.item(np.argmax(np.abs(rate)))
    return rate

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


class IntervalChart(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = plt.figure(figsize=(8,5), dpi=DPI_default)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Fixed,
                                   QSizePolicy.Fixed)
        FigureCanvas.updateGeometry(self)
        sns.set_style('whitegrid')
        sns.despine()
        self.alp = 0.4

        self.ax = self.fig.add_subplot(111)
        self.ax.hold(True)

        self.draw()

    def updatePlot(self, data, priceList):
        self.data = data
        self.data = [[item/1000000 for item in itemlist] for itemlist in self.data]
        self.priceList = range(5)
        tableLength = len(self.data)

        self.n=range(5)
        self.n[0] = int(0)
        self.n[4] = int(-1)
        self.n[2] = int(np.floor(tableLength/2))
        self.n[1] = int(np.floor(self.n[2]/2))
        self.n[3] = int(2*self.n[2] - self.n[1])

        # clear current figure
        self.ax.cla()

        ind = np.arange(len(self.data[0]))
        title = 'Field Net Cashflow with Oil Pirces at \$%d, %d, %d, %d and %d' \
                % (priceList[self.n[0]],priceList[self.n[1]],priceList[self.n[2]],priceList[self.n[3]],priceList[self.n[4]])
        
        self.ax.set_xlabel('Year', fontsize=12)
        self.ax.set_ylabel('Net Cashflow ($mm.)', fontsize=12)
        self.ax.set_title(title, fontsize=16, fontweight='bold')
        self.ax.set_xticks(ind)

        self.y = [[] for _ in range(5)]
        for i in range(5):
        	self.y[i] = self.data[self.n[i]]
        self.x = range(len(self.y[4]))

        self.ax.plot(self.x[:len(self.y[0])], self.y[2][:len(self.y[0])], linewidth=2.5)
        self.ax.fill_between(self.x[:len(self.y[0])], self.y[0][:len(self.y[0])], self.y[4][:len(self.y[0])], alpha = self.alp)
        self.ax.fill_between(self.x[:len(self.y[0])], self.y[1][:len(self.y[0])], self.y[3][:len(self.y[0])], alpha = self.alp)
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
        sns.set_style("whitegrid")
        self.colorList = sns.color_palette("cubehelix", 6)
        self.colorList = sns.cubehelix_palette(6, start=.5, rot=-.65, light=0.95)

        self.ax = self.fig.add_subplot(111)
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


class irrChart(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = plt.figure(figsize=(8,6), dpi=DPI_default)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Fixed,
                                   QSizePolicy.Fixed)
        FigureCanvas.updateGeometry(self)
        sns.set_style("whitegrid")
        self.colorList = sns.cubehelix_palette(6, start=.5, rot=-.65, light=0.95)

        self.ax1 = self.fig.add_subplot(211)
        self.ax1.hold(True)
        plt.subplots_adjust(hspace = 0.5)
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.hold(True)

        self.draw()

    def updatePlot(self, irrList, priceList):
        self.ax1.cla()
        self.ax2.cla()

        self.irrList = [a*100 for a in irrList]
        self.ax1.plot(priceList, self.irrList, linewidth=2)
        self.ax2.plot(priceList[:len(priceList)//2], self.irrList[:len(priceList)//2], linewidth=2)

        self.ax1.set_xlabel('Oil Price $', fontsize=12)
        self.ax1.set_ylabel('Internal Rate of Return %', fontsize=12)
        self.ax1.set_title('IRR for different Oil Prices', fontsize=16, fontweight='bold')
        # self.ax1.set_xticks(priceList)

        self.ax2.set_xlabel('Oil Price $', fontsize=12)
        self.ax2.set_ylabel('Internal Rate of Return %', fontsize=12)
        self.ax2.set_title('IRR for different Oil Prices', fontsize=16, fontweight='bold')
        # self.ax2.set_xticks(priceList[:len(priceList)//2])

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


class NPVChart(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = plt.figure(figsize=(8,3), dpi=DPI_default)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Fixed,
                                   QSizePolicy.Fixed)
        FigureCanvas.updateGeometry(self)
        sns.set_style("whitegrid")
        self.colorList = sns.cubehelix_palette(6, start=.5, rot=-.65, light=0.95)

        self.ax1 = self.fig.add_subplot(111)
        self.ax1.hold(True)

        self.draw()

    def updatePlot(self, NPVList, priceList):
        self.ax1.cla()

        self.NPVList = [a/1000000 for a in NPVList]
        self.ax1.plot(priceList, self.NPVList, linewidth=2)

        self.ax1.set_xlabel('Oil Price', fontsize=12)
        self.ax1.set_ylabel('Net Present Value $mm.', fontsize=12)
        self.ax1.set_title('NPV for different Oil Prices', fontsize=16, fontweight='bold')
        # self.ax1.set_xticks(priceList)

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
        # sns.set_style("whitegrid")

        self.ax = self.fig.add_subplot(111)
        self.ax.hold(True)

        self.draw()

    def updatePlot(self, x, y, z):
        self.x = x
        self.y = [(a*100) for a in y]
        self.z = [[a/1000000 for a in b] for b in z]
        self.gridsize = 25

        # clear current figure
        self.ax.cla()
        # self.fig.clf()

        indx = [int(np.floor(a)) for a in x]
        indy = [int(np.floor(a*100)) for a in y]

        self.ax.set_xlabel('Oil Price', fontsize=12)
        self.ax.set_ylabel('Success Rate %', fontsize=12)
        self.ax.set_title('Cross Analysis of Oil Price and Drilling Success Rate', fontsize=16, fontweight='bold')
        # self.ax.set_xticks(indx)
        # self.ax.set_yticks(indy)

        self.x, self.y = np.meshgrid(self.x, self.y)
        self.x = self.x.ravel()
        self.y = self.y.ravel()
        self.z = np.asarray(self.z).ravel()
        im = self.ax.hexbin(self.x, self.y, C=self.z, gridsize=self.gridsize, cmap = sns.cubehelix_palette(light=1,start=.5, rot=-.65, as_cmap=True), bins=None)
        self.ax.axis([self.x.min(), self.x.max(), self.y.min(), self.y.max()])

        if not hasattr(self, 'cb'):
            self.cb = self.fig.colorbar(im)
            self.cb.set_label('Net Present Value ($mm.)')

        # draw new figure
        self.draw()



####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################


__version__ = "1.0.0"

class FModel(QMainWindow):
    def __init__(self, parent=None):
        super(FModel, self).__init__(parent)
        self.initUI()
        

    def initUI(self):
        self.setStyleSheet(
            '''
                QMainWindow{
                background-color: white;
                }

                QSplitter{
                background-color: white;
                padding-top: 8px;
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
                min-width: 190px;
                min-height: 28px;
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
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setStretchFactor(0,0)
        self.splitter.setStretchFactor(1,1)
        self.parameterBox = ParameterBox()
        self.resultBox = QTabWidget()

        self.tab1 = ResultTab()
        self.tab2 = ResultTab()
        self.resultBox.addTab(self.tab1, 'Gross Field Report')
        self.resultBox.addTab(self.tab2, 'AE&&E Report')

        reportBtn = QPushButton('Generate Report')
        reportBtn.setFixedWidth(150)

        reportBtn.clicked.connect(self.generateReport)
        leftWidget = QWidget()
        leftWidget.setFixedWidth(550)
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

        self.calculateAllResults(currentPrice, priceList, taxRate, royaltyRate, rental, bonus, yearlyCPI, monthlyCPI, fixedOpertExpense, varOpertExpense,
                                 capitalCost, tangibleCapitalPerc, abandonCost, wellData, lowestPrice, highestPrice, recoverableVol, maxNumOfProducingWells)

        # (NPVList, irrList, netCashflowTable, breakevenPriceList, NPVTable, breakevenYearList, yearlyRoyaltyCurrentPrice, yearlyCapitalCurrentPrice, yearlyOpertExpenseCurrentPrice,
        # yearlyBonusCurrentPrice, yearlyRentalCurrentPrice, yearlyIncomeTaxCurrentPrice, succRateList) = \
        #     self.calculateAllResults(currentPrice, priceList, taxRate, royaltyRate, rental, bonus, CPI, fixedOpertExpense, varOpertExpense, capitalCost, tangibleCapitalPerc, self.wellData)
        # indexCurrentPrice = priceList.index(currentPrice)
        # irrCurrentPrice = irrList[indexCurrentPrice]
        # NPVCurrentPrice = NPVList[indexCurrentPrice]
        # breakevenYearCurrentPrice = breakevenYearList[indexCurrentPrice]

        # self.tab1.updateGraphs(netCashflowTable, yearlyRoyaltyCurrentPrice, yearlyCapitalCurrentPrice, yearlyOpertExpenseCurrentPrice,
        #                        yearlyBonusCurrentPrice, yearlyRentalCurrentPrice, yearlyIncomeTaxCurrentPrice, priceList, succRateList, NPVTable,
        #                        irrCurrentPrice, NPVCurrentPrice, breakevenYearCurrentPrice,
        #                        irrList, NPVList)
        # self.tab2.updateGraphs(netCashflowTable, yearlyRoyaltyCurrentPrice, yearlyCapitalCurrentPrice, yearlyOpertExpenseCurrentPrice,
        #                        yearlyBonusCurrentPrice, yearlyRentalCurrentPrice, yearlyIncomeTaxCurrentPrice, priceList, succRateList, NPVTable,
        #                        irrCurrentPrice, NPVCurrentPrice, breakevenYearCurrentPrice,
        #                        irrList, NPVList)



    def calculateAllResults(self, currentPrice, priceList, taxRate, royaltyRate, rental, bonus, yearlyCPI, monthlyCPI, fixedOpertExpense, varOpertExpense,
                            capitalCost, tangibleCapitalPerc, abandonCost, wellData, lowestPrice, highestPrice, recoverableVol, maxNumOfProducingWells):
        monthlyEachWellProdVolTable, monthlyFieldVolList, yearlyNetCashflow, yearlyFieldRevenueList, yearlyFieldCapitalCostList,\
        yearlyFieldRoyaltyList, yearlyFieldBonusList, yearlyFieldOpertCostList, yearlyFieldAbandonCostList, yearlyTaxList,\
        monthlyFieldRevenueList, monthlyFieldCapitalCostList, monthlyFieldRoyaltyList, monthlyFieldBonusList,\
        monthlyFieldOpertCostList, monthlyFieldAbandonCostList\
        = self.calculateResultCurrentPrice(currentPrice=currentPrice, taxRate=taxRate, royaltyRate=royaltyRate, rental=rental, bonus=bonus, yearlyCPI=yearlyCPI, monthlyCPI=monthlyCPI,
                                            fixedOpertExpense=fixedOpertExpense, varOpertExpense=varOpertExpense, capitalCost=capitalCost, tangibleCapitalPerc=tangibleCapitalPerc,
                                            abandonCost=abandonCost, wellData=wellData, recoverableVol=recoverableVol, maxNumOfProducingWells=maxNumOfProducingWells)
        # print yearlyFieldRevenueList
        x = self.calculateResultSensitivity()
        x = self.calculateResultAEE()



    def calculateResultCurrentPrice(self, **kwargs):
        # unpack parameters from **kwargs
        currentPrice = kwargs.pop('currentPrice'); taxRate = kwargs.pop('taxRate'); royaltyRate = kwargs.pop('royaltyRate'); rental = kwargs.pop('rental'); bonus = kwargs.pop('bonus');
        yearlyCPI = kwargs.pop('yearlyCPI'); monthlyCPI = kwargs.pop('monthlyCPI'); fixedOpertExpense = kwargs.pop('fixedOpertExpense'); varOpertExpense = kwargs.pop('varOpertExpense');
        capitalCost = kwargs.pop('capitalCost'); tangibleCapitalPerc = kwargs.pop('tangibleCapitalPerc'); abandonCost = kwargs.pop('abandonCost'); wellData = kwargs.pop('wellData');
        recoverableVol = kwargs.pop('recoverableVol'); maxNumOfProducingWells = kwargs.pop('maxNumOfProducingWells')
        if kwargs:
            raise TypeError('Unexpected **kwargs: %r' % kwargs)

        try:
            succRate = kwargs['succRate']
        except:
            succRate = 0

        depreciationLength = 25  # years
        taxLossCarryforwardYears = 7  # years

        developMonths = 6  # months
        averageWellVol = recoverableVol/maxNumOfProducingWells
        monthlyEachWellProdVolTable = []
        declineParameter = 1
        b = 1/2
        lastInstallMonth = 0

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
        for flag in yearlyELTFlagList: monthlyELTFlagList += [flag]*12

        # set monthly production volumn to 0 when oil field is OFF (ELT Flag = 0)
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
            revenueList, capitalCostList, yearlyTangibleCapitalDepreciationListTemp, intangibleCapitalCostList, royaltyList, bonusList, opertCostList, abandonCostList = wellVol2Cash(monthlyVolList)

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

        # Prepare results
        yearlyCostPerBarrelList = [a/b for a,b in zip(yearlyCostList, yearlyFieldProdVolList)]

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

        return monthlyEachWellProdVolTable, monthlyFieldVolList, yearlyNetCashflowList, yearlyFieldRevenueList, yearlyFieldCapitalCostList,\
               yearlyFieldRoyaltyList, yearlyFieldBonusList, yearlyFieldOpertCostList, yearlyFieldAbandonCostList, yearlyTaxList,\
               monthlyFieldRevenueList, monthlyFieldCapitalCostList, monthlyFieldRoyaltyList, monthlyFieldBonusList,\
               monthlyFieldOpertCostList, monthlyFieldAbandonCostList


    def calculateResultSensitivity(self):
        pass

    def calculateResultAEE(self ):
        pass


    def Original_calculateResult(self, currentPrice, priceList, taxRate, royaltyRate, rental, bonus, CPI, declineRate, fixedOpertExpense, varOpertExpense, capitalCost, tangibleCapitalPerc, wellData):
        depreciationPeriod = 25
        NPVList = copy.deepcopy(priceList)
        irrList = copy.deepcopy(priceList)
        breakevenYearList = copy.deepcopy(priceList)
        breakevenPriceList = []
        netCashflowTable = [[] for _ in priceList]
        for i, loopPrice in enumerate(priceList):
            if loopPrice == currentPrice:
                (NPVList[i], irrList[i], netCashflowTable[i], breakevenPriceList, breakevenYearList[i],
                 yearlyRoyaltyCurrentPrice, yearlyCapitalCurrentPrice, yearlyOpertExpenseCurrentPrice,
                 yearlyBonusCurrentPrice, yearlyRentalCurrentPrice, yearlyIncomeTaxCurrentPrice)\
                  = self.calculateSubResult(loopPrice, currentPrice, taxRate, royaltyRate, rental, bonus, CPI, declineRate, fixedOpertExpense, varOpertExpense, capitalCost, tangibleCapitalPerc, wellData, depreciationPeriod)
            else:
                (NPVList[i], irrList[i], netCashflowTable[i], breakevenPriceList, breakevenYearList[i]) \
                  = self.calculateSubResult(loopPrice, currentPrice, taxRate, royaltyRate, rental, bonus, CPI, declineRate, fixedOpertExpense, varOpertExpense, capitalCost, tangibleCapitalPerc, wellData, depreciationPeriod)

        succRateList = np.linspace(0.3, 0.8, 36)
        NPVTable = [[0 for _ in priceList] for _ in succRateList]
        for ii, succRate in enumerate(succRateList):
            for jj, loopPrice in enumerate(priceList):
                if loopPrice == currentPrice:
                    NPVTable[ii][jj],_,_,_,_,_,_,_,_,_,_ = self.calculateSubResult(loopPrice, currentPrice, taxRate, royaltyRate, rental, bonus, CPI, declineRate, fixedOpertExpense, varOpertExpense, capitalCost, tangibleCapitalPerc, wellData, depreciationPeriod, succRate)
                else:
                    NPVTable[ii][jj],_,_,_,_ = self.calculateSubResult(loopPrice, currentPrice, taxRate, royaltyRate, rental, bonus, CPI, declineRate, fixedOpertExpense, varOpertExpense, capitalCost, tangibleCapitalPerc, wellData, depreciationPeriod, succRate)
        return (NPVList, irrList, netCashflowTable, breakevenPriceList, NPVTable, breakevenYearList,
                yearlyRoyaltyCurrentPrice, yearlyCapitalCurrentPrice, yearlyOpertExpenseCurrentPrice,
                yearlyBonusCurrentPrice, yearlyRentalCurrentPrice, yearlyIncomeTaxCurrentPrice, succRateList)

    def Original_calculateSubResult(self, loopPrice, currentPrice, taxRate, royaltyRate, rental, bonus, CPI, declineRate, fixedOpertExpense, varOpertExpense, capitalCost, tangibleCapitalPerc, wellData, depreciationPeriod, succRate=None):
        # calculate oil production list (oilProdVolList)
        Price = copy.copy(loopPrice)
        oilProdVolList = [[0 for x in range(wellData[-1][0]+wellLifespan)] for x in range(len(wellData))]
        for iWell, ProdList in enumerate(oilProdVolList):
            initTime = wellData[iWell][0]
            numWell = wellData[iWell][1]
            initProd = wellData[iWell][2]
            if succRate == None:
                succRate = wellData[iWell][3]/100
            
            for year, prodVol in enumerate(ProdList):
                if year >= initTime:
                    p = succRate*365.25*numWell*initProd*(declineRate**(year-initTime))
                    oilProdVolList[iWell][year] = p
                    if year == initTime:
                        oilProdVolList[iWell][year] = oilProdVolList[iWell][year]/2




        oilProdVolList = []

        lastWellYear = 12 * (max(x[0] for x in wellData) + wellLifespan)

        for startYear, startMonth, nWells, succRate in iter(wellData):
            for _ in xrange(nWells):
                n = 12*startYear + startMonth
                if random.uniform(0,1) < succRate:
                    prodVol = [0 for _ in xrange(lastWellYear)]




        # Econmic Limit Test
        yearlyCPI = [CPI**i for i in range(len(oilProdVolList[0]))]
        yearlyOilPrice = [cpi*Price for cpi in yearlyCPI]
        grossRevenueListPreELT = [[Price*volumn for Price,volumn in zip(yearlyOilPrice, wellProdVol)] for wellProdVol in oilProdVolList]
        royaltyListPreELT = [[e * royaltyRate for e in well] for well in grossRevenueListPreELT]
        bonusList = [[0 for x in range(len(grossRevenueListPreELT[0]))] for x in range(len(grossRevenueListPreELT))]
        capitalCostList = copy.deepcopy(bonusList)
        tangibleCapitalCostList = copy.deepcopy(bonusList)
        intangibleCapitalCostList = copy.deepcopy(bonusList)
        tangibleCapitalDepreciationList = copy.deepcopy(bonusList)
        cashflowListPreRental = copy.deepcopy(bonusList)
        ELTFlagList = [[1 for x in range(len(grossRevenueListPreELT[0]))] for x in range(len(grossRevenueListPreELT))]
        

        for iWell, iWellRevenueList in enumerate(grossRevenueListPreELT):
            if succRate == None:
                succRate = wellData[iWell][3]/100
            for year, revenue in enumerate(iWellRevenueList):
                if revenue:
                    bonusList[iWell][year] = succRate*bonus*wellData[iWell][1]
                    c = [capital*(CPI**year)*wellData[iWell][1] for capital in capitalCost]
                    capitalCostList[iWell][year:year+2] = c
                    tangibleCapitalCostList[iWell][year:year+2] = [a*b for a,b in zip(c, tangibleCapitalPerc)]
                    intangibleCapitalCostList[iWell][year:year+2] = [a*(1-b) for a,b in zip(c, tangibleCapitalPerc)]
                    break

        # Calculate every year depreciation of tangible capital cost
        for year, data in enumerate(tangibleCapitalCostList):
            s1 = [0 for x in data]
            s2 = copy.deepcopy(s1)
            s3 = copy.deepcopy(s1)
            for i,d in enumerate(data):
                depLength = min(len(data)-i, depreciationPeriod)
                if d:
                    s1[i:] = [d/depreciationPeriod for x in range(depLength)]
                    s2[(i+1):] = [data[i+1]/depreciationPeriod for x in range(depLength-1)]
                    s3[(i+2):] = [data[i+2]/depreciationPeriod for x in range(depLength-2)]
                    l = [a+b+c for a,b,c in zip(s1,s2,s3)]
                    tangibleCapitalDepreciationList[year][0:len(l)] = l
                    break
            

        OpertExpenseListPreELT = [[fixedOpertExpense*wellData[iWell][1]*wellData[iWell][3]/100\
                                + vol*varOpertExpense for vol in iWellProd] for iWell,iWellProd in enumerate(oilProdVolList)]
        
        # ELT Flag = 0 if cashflow is negative, 1 if positive.
        for iWell in range(len(ELTFlagList)):
            for year in range(len(ELTFlagList[0])):
                cashflow = grossRevenueListPreELT[iWell][year] - royaltyListPreELT[iWell][year]\
                           - capitalCostList[iWell][year] - OpertExpenseListPreELT[iWell][year]
                if year > (wellData[iWell][0]+1) and cashflow <= 0:
                    ELTFlagList[iWell][year] = 0
                else:
                    cashflowListPreRental[iWell][year] = cashflow


        # consider Rental for all wells to determine field ELT:
        yearlyCashflowPreRental = [0 for x in range(len(ELTFlagList[0]))]
        for year in range(len(ELTFlagList[0])):
            for iWell in range(len(ELTFlagList)):
                yearlyCashflowPreRental[year] += cashflowListPreRental[iWell][year]

        # find the last year that all wells in the field is generating positive cashflow
        for year, data in enumerate(yearlyCashflowPreRental):
            yearCashflow = data - rental
            if yearCashflow < 0 and year > (wellData[-1][0]+1):
                lastFieldYear = year-1
                break
        else:
            lastFieldYear = len(yearlyCashflowPreRental)

        # Delete columns of years after last field economic year
        for iWell, ELTdata in enumerate(ELTFlagList):
            ELTFlagList[iWell][lastFieldYear+1:] = []

        # Calculate Post ELT lists
        grossRevenueListPostELT = copy.deepcopy(grossRevenueListPreELT[:][:lastFieldYear])
        royaltyListPostELT = copy.deepcopy(royaltyListPreELT[:][:lastFieldYear])
        OpertExpenseListPostELT = copy.deepcopy(OpertExpenseListPreELT[:][:lastFieldYear])
        oilProdVolListPostELT = copy.deepcopy(oilProdVolList[:][:lastFieldYear])
        
        for iWell, ELTList in enumerate(ELTFlagList):
            for year, ELT in enumerate(ELTList):
                grossRevenueListPostELT[iWell][year] = ELT*grossRevenueListPreELT[iWell][year]
                royaltyListPostELT[iWell][year] = ELT*royaltyListPreELT[iWell][year]
                OpertExpenseListPostELT[iWell][year] = ELT*OpertExpenseListPreELT[iWell][year]
                tangibleCapitalDepreciationList[iWell][year] = ELT*tangibleCapitalDepreciationList[iWell][year]
                oilProdVolListPostELT[iWell][year] = ELT*oilProdVolListPostELT[iWell][year]

        # Post ELT: Tax Calculations:
        yearlyRevenue = [0 for x in range(len(ELTFlagList[0]))]
        yearlyOilProdVol = copy.deepcopy(yearlyRevenue)
        yearlyRoyalty = copy.deepcopy(yearlyRevenue)
        yearlyCapital = copy.deepcopy(yearlyRevenue)
        yearlyTangibleCapitalDepreciation = copy.deepcopy(yearlyRevenue)
        yearlyIntangibleCapital = copy.deepcopy(yearlyRevenue)
        yearlyBonus = copy.deepcopy(yearlyRevenue)
        yearlyOpertExpense = copy.deepcopy(yearlyRevenue)
        yearlyRental = [rental for x in ELTFlagList[0]]

        taxLoss = copy.deepcopy(yearlyRevenue)
        taxAllowance = copy.deepcopy(yearlyRevenue)
        yearlyIncomeTax = copy.deepcopy(yearlyRevenue)
        yearlyCashOutflow = copy.deepcopy(yearlyRevenue)
        yearlyNetCashflow = copy.deepcopy(yearlyRevenue)
        yearlyCost = copy.deepcopy(yearlyRevenue)
        yearlyCostPerBarrel = copy.deepcopy(yearlyRevenue)
        yearlyBreakevenPrice = copy.deepcopy(yearlyRevenue)
        
        for year in range(len(ELTFlagList[0])):
            for iWell in range(len(ELTFlagList)):
                yearlyRevenue[year] += grossRevenueListPostELT[iWell][year]
                yearlyOilProdVol[year] += oilProdVolListPostELT[iWell][year]
                yearlyRoyalty[year] += royaltyListPostELT[iWell][year]
                yearlyCapital[year] += capitalCostList[iWell][year]
                yearlyTangibleCapitalDepreciation[year] += tangibleCapitalDepreciationList[iWell][year]
                yearlyIntangibleCapital[year] += intangibleCapitalCostList[iWell][year]
                yearlyBonus[year] += bonusList[iWell][year]
                yearlyOpertExpense[year] += OpertExpenseListPostELT[iWell][year]

        for year,item in enumerate(yearlyRevenue):
            taxLossCarryforwardYears = 7
            taxAllowance[year] = yearlyRoyalty[year] + yearlyTangibleCapitalDepreciation[year]\
             + yearlyIntangibleCapital[year] + yearlyOpertExpense[year] + yearlyBonus[year] + yearlyRental[year]
            taxLoss[year] = -(yearlyRevenue[year] - taxAllowance[year])
            if year>0:
                accumulatedTaxLoss = max(0, sum(taxLoss[max(0,(year-(taxLossCarryforwardYears-1))):year]))
                taxAllowance[year] = taxAllowance[year] + accumulatedTaxLoss
            taxableProfit = max(0, yearlyRevenue[year] - taxAllowance[year])
            yearlyIncomeTax[year] = taxableProfit * taxRate

        for year,item in enumerate(yearlyRevenue):
            yearlyCashOutflow[year] = yearlyRoyalty[year] + yearlyCapital[year] + yearlyOpertExpense[year]\
                                          + yearlyBonus[year] + yearlyRental[year] + yearlyIncomeTax[year]
            yearlyCost[year] = yearlyOpertExpense[year] + yearlyRental[year]
            yearlyBreakevenPrice[year] = yearlyCost[year]/yearlyOilProdVol[year]/(1-royaltyRate)
            yearlyNetCashflow[year] = yearlyRevenue[year] - yearlyCashOutflow[year]

        yearlyRealNetCashflow = [a/b for a,b in zip(yearlyNetCashflow, yearlyCPI)]
        yearlyCostPerBarrel = [a/b/c for a,b,c in zip(yearlyCost, yearlyCPI, yearlyOilProdVol)]


        NPV = (sum(yearlyRealNetCashflow))
        for i in xrange(len(yearlyRealNetCashflow)):
            if sum(yearlyRealNetCashflow[:i]) < 0 and sum(yearlyRealNetCashflow[:i+1]) >= 0:
                breakevenYear = i
                break
        else:
            breakevenYear = 0

        if all(item >= 0 for item in yearlyNetCashflow) or all(item < 0 for item in yearlyNetCashflow):
            irr = None
        else:
            irr = np.irr(yearlyNetCashflow)

        # print(loopPrice, irr)

        if loopPrice == currentPrice:
            result = (NPV, irr, yearlyNetCashflow, yearlyBreakevenPrice, breakevenYear, yearlyRoyalty,
                      yearlyCapital, yearlyOpertExpense, yearlyBonus, yearlyRental, yearlyIncomeTax)
        else:
            result = (NPV, irr, yearlyNetCashflow, yearlyBreakevenPrice, breakevenYear)
            
        return result


class ParameterBox(QFrame):
    def __init__(self, parent=None):
        super(ParameterBox, self).__init__(parent)
        self.initParameterBox()


    def initParameterBox(self):
        self.setStyleSheet(
            '''
                QFrame{
                    background-color: #d8dde3;
                }

                QGroupBox {
                    background-color: #d8dde3;
                    border: 2px solid gray;
                    border-radius: 5px;
                    margin-top: 30px;
                }

                QGroupBox::title {
                    font: bold 50px;
                    subcontrol-origin: margin;
                    left: 15px;
                    padding: 21px 3px 0 3px;
                }
            ''')
        layout = QGridLayout()
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
        layout.addWidget(groupBox_Eco)

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
        layout.addWidget(groupBox_Contract)

        # Royalty Rate
        self.royaltyRateSpinbox = QDoubleSpinBox()
        self.royaltyRateSpinbox.setSuffix(" %")
        self.royaltyRateSpinbox.setValue(royaltyRate)
        self.royaltyRateSpinbox.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        box.addLayout(HBox(QLabel("Royalty Rate:"), self.royaltyRateSpinbox),0,0)

        # rental
        self.rentalSpinbox = QDoubleSpinBox()
        self.rentalSpinbox.setPrefix("$mm ")
        self.rentalSpinbox.setValue(rental)
        self.rentalSpinbox.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        box.addLayout(HBox(QLabel("Rental/yr/well:"), self.rentalSpinbox),0,1)

        # bonus of first production
        self.bonusSpinbox = QDoubleSpinBox()
        self.bonusSpinbox.setPrefix("$mm ")
        self.bonusSpinbox.setValue(bonus)
        self.bonusSpinbox.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        box.addLayout(HBox(QLabel("Bonus of 1st oil:"), self.bonusSpinbox),1,0)

        


        
        #####     Expense Parameters     #####
        groupBox_Expense = QGroupBox('Expense Parameters of Each Well')
        box = QGridLayout()
        groupBox_Expense.setLayout(box)
        layout.addWidget(groupBox_Expense)

        # Capital Expense
        box.addWidget(QLabel('Year 1'),0,1)
        box.addWidget(QLabel('Year 2'),0,2)
        box.addWidget(QLabel('Year 3'),0,3)

        box.addWidget(QLabel("Capital Expense:"),1,0)
        # 1
        self.capitalCostSpinbox1 = QDoubleSpinBox()
        self.capitalCostSpinbox1.setPrefix("$mm ")
        self.capitalCostSpinbox1.setValue(capitalYear1)
        self.capitalCostSpinbox1.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        # 2
        self.capitalCostSpinbox2 = QDoubleSpinBox()
        self.capitalCostSpinbox2.setPrefix("$mm ")
        self.capitalCostSpinbox2.setValue(capitalYear2)
        self.capitalCostSpinbox2.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        # 3
        self.capitalCostSpinbox3 = QDoubleSpinBox()
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
        layout.addWidget(groupBox_Field)

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
        layout.addWidget(groupBox_Plan)

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
        priceList = range(lowestPrice, int(currentPrice), 2) + range(int(currentPrice), highestPrice, 2)
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


class ResultTab(QFrame):
    def __init__(self, parent=None):
        super(ResultTab, self).__init__(parent)
        self.setMinimumWidth(700)
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

        self.irrBox = boxResult('Internal Rate of Return')
        self.NPVBox = boxResult('Net Present Value')
        self.BreakevenBox = boxResult('Year of Breakeven')
        topResultBox = QHBoxLayout()
        topResultBox.addWidget(self.irrBox)
        topResultBox.addSpacing(20)
        topResultBox.addWidget(self.NPVBox)
        topResultBox.addSpacing(20)
        topResultBox.addWidget(self.BreakevenBox)

        self.IntervalChart = IntervalChart()
        self.outflowBarchart = stackedBarchart()
        self.heatmap = Heatmap()
        self.irrChart = irrChart()
        self.NPVChart = NPVChart()

        subLayout0.addSpacing(0)
        subLayout0.addWidget(QLabel('Summary:'))
        subLayout0.addSpacing(10)
        subLayout0.addLayout(topResultBox)
        subLayout0.addSpacing(30)

        subLayout0.addWidget(QLabel('Sensitivity Analysis:'))
        subLayout0.addWidget(self.IntervalChart)
        subLayout0.addWidget(self.outflowBarchart)
        subLayout0.addWidget(self.heatmap)
        subLayout0.addWidget(self.irrChart)
        subLayout0.addWidget(self.NPVChart)
        
        layout.addWidget(self.scrollArea)
        self.setLayout(layout)

    def updateGraphs(self, netCashflowTable, yearlyRoyaltyCurrentPrice, yearlyCapitalCurrentPrice, yearlyOpertExpenseCurrentPrice,
                     yearlyBonusCurrentPrice, yearlyRentalCurrentPrice, yearlyIncomeTaxCurrentPrice, priceList, succRateList, NPVTable,
                     irrCurrentPrice, NPVCurrentPrice, breakevenYearCurrentPrice,
                     irrList, NPVList):
        self.IntervalChart.updatePlot(netCashflowTable, priceList)
        self.outflowBarchart.updatePlot(yearlyRoyaltyCurrentPrice, yearlyCapitalCurrentPrice, yearlyOpertExpenseCurrentPrice,
                                        yearlyBonusCurrentPrice, yearlyRentalCurrentPrice, yearlyIncomeTaxCurrentPrice)
        self.heatmap.updatePlot(priceList, succRateList, NPVTable)

        self.irrBox.updatePlot('%0.2f%%' % (irrCurrentPrice*100))
        self.NPVBox.updatePlot('$mm. %0.2f' % (NPVCurrentPrice/1000000))
        self.BreakevenBox.updatePlot('%d' % breakevenYearCurrentPrice)
        self.irrChart.updatePlot(irrList, priceList)
        self.NPVChart.updatePlot(NPVList, priceList)


class HBox(QHBoxLayout):
    def __init__(self, *args):
        super(HBox, self).__init__(None)
        for w in args:
            self.addWidget(w)


if __name__ == "__main__":
    app = QApplication([])
    # Create and display the splash screen
    splash_pix = QPixmap('/Users/yunyunzhang/Desktop/PythonTest/quokka.jpg')
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())
    splash.show()
    # Simulate something that takes time
    time.sleep(1)
    win = FModel()
    win.show()
    splash.finish(win)

    sys.exit(app.exec_())




