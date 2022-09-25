from math import floor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os

class draw_hold_period:
    root = os.getcwd()
    fig = plt.figure(figsize=[21, 9], dpi=300, constrained_layout=True)
    allFontSize = 15
    companiesTradeInfo = dict()
    
    def __init__(self, expFolder, resultFolder, tech, bestHold, fundLV, isTrain, isTradition, delFile, setCompany):
        self.tech = tech
        self.drawBestHold = bestHold
        self.algoOrTrad = (lambda x: 'Tradition' if x == True else '')(isTradition)
        self.trainOrTest = (lambda x: 'train' if x == True else 'test')(isTrain)
        self.scatterClr = {'buy': 'black', 'sell date': 'lime', 'sell': 'yellow'}
        self.scatterMarker = {'buy': '.', 'sell date': '.', 'sell': '.'}
        
        os.chdir('../')
        parentFolder = os.getcwd()
        self.workRoot = [dir for dir in glob.glob(parentFolder + '/**/**') if expFolder in dir][0] + f'/{resultFolder}/result_{self.tech}/'
        # self.workRoot = self.workRoot.replace(parentFolder + '/', '')
        os.chdir(self.workRoot)
        
        if setCompany != 'all':
            setCompany = setCompany.split(',')
            self.allCompay = setCompany
        else:
            self.allCompay = [dir for dir in os.listdir() if os.path.isdir(dir)]
        
        if bestHold:
            self.holdPath = [i + f'/{self.trainOrTest + self.algoOrTrad}BestHold/' for i in self.allCompay]
        elif fundLV:
            self.holdPath = [i + f'/{self.trainOrTest + self.algoOrTrad}Hold/' for i in self.allCompay]
        
        self.fig = draw_hold_period.fig
        gridNum = 24
        self.gs = self.fig.add_gridspec(
            gridNum, 1, 
            # wspace=0, 
            # hspace=1, 
            # top=1, 
            # bottom=0, 
            # left=0.17, 
            # right=0.845
            )
        for holdDir in self.holdPath:
            os.chdir(holdDir)
            if delFile[0]:
                filesToDel = [i for i in glob.glob(f"*{delFile[1]}")]
                for fileToDel in filesToDel:
                    os.remove(fileToDel)
            else:
                holdFile = [i for i in glob.glob(f"*{'.csv'}") if 'hold' in i]
                for file in holdFile:
                    if fundLV == True:
                        self.draw_fundLv(file)
                    else :
                        if not isTrain or self.check_symmetric(file):  # 訓練期的持有區間只能畫對稱的滑動視窗，所以檢查滑動視窗是否對稱
                            self.process_file_and_draw(file)
            for i in range(2):
                os.chdir('../')
        self.output_tradeInfo()
    
    def draw_fundLv(self, file):
        df = pd.read_csv(file, index_col=0, usecols=[0, 7, 8])
        
        # =====可篩選想要畫出的資金水位的條件
        if df[df.columns[0]].iloc[-1] < df[df.columns[1]].iloc[-1] or df[df.columns[0]].iloc[-1] < 10000000:  # 如果測試期最後一天B&H高或是沒賺錢的就不畫
            return
        # if len([i for i, j in zip(df[df.columns[0]], df[df.columns[1]]) if i > j]) < len(df) * 0.5:  # 如果滑動視窗資金水位在測試期中沒有高過B&H的天數一半就不畫
        #     return
        # =====可增加畫出的資金水位的條件
        
        print(file)
        ax = self.fig.add_subplot(self.gs[:, :])
        ax.plot(df.index, df[df.columns[0]], label=df.columns[0], color='lightgray', linewidth=4)
        ax.plot(df.index, df[df.columns[1]], label=df.columns[1], color='red', linewidth=4)
        yearIndexes = []
        year = 0
        for i, date in enumerate(df.index):
            nowYear = date.split('-')[0]
            if year != nowYear:
                yearIndexes.append(i)
                year = nowYear
        yearIndexes.append(len(df) - 1)
        ax.xaxis.set_ticks(yearIndexes)
        ax.tick_params(axis='both', which='major', labelsize=draw_hold_period.allFontSize + 5)
        ax.legend(prop={'size': draw_hold_period.allFontSize + 5})
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(
        #     handles, labels, 
        #     loc='upper right', 
        #     bbox_to_anchor=(1, 1.3), 
        #     fancybox=True, shadow=False, 
        #     ncol=len(df.columns), 
        #     fontsize=draw_hold_period.allFontSize + 5)
        ax.grid()
        fileTitle = self.tech + '_' + '_'.join(file.split('_')[0:-1]) + '_fund_lv' 
        figTitle = fileTitle.replace('_', ' ')
        self.fig.suptitle(
            figTitle, 
            y=1.02,
            fontsize=draw_hold_period.allFontSize + 5)
        plt.savefig(fileTitle + '.png', dpi=draw_hold_period.fig.dpi, bbox_inches='tight')
        plt.clf()

    def check_symmetric(self, file):  # 訓練期的持有區間只能畫對稱的滑動視窗，所以檢查滑動視窗是否對稱
        window = file.split('_')[-2]
        if not window[0].isnumeric:
            if window[-1] == '#':
                return False
            else:
                return window.split('2')[0] == window.split('2')[1]
        windowType = [i for i in window if i == 'W' or i == 'D'][0]
        return window.split(windowType)[0] == window.split(windowType)[1]
    
    def process_file_and_draw(self, file):
        print(file)
        companyName = file.split('_')[2]
        df = pd.read_csv(file, index_col=0, usecols=[i for i in range(7)])
        yearIndexes = []
        year = 0
        for i, date in enumerate(df.index):
            nowYear = date.split('-')[0]
            if year != nowYear:
                yearIndexes.append(i)
                year = nowYear
        yearIndexes.append(len(df))
        if not self.drawBestHold:
            yearIndexes = [yearIndexes[0], yearIndexes[-1]]
        for yearIndex in range(len(yearIndexes)):
        # for yearIndex in range(len(yearIndexes) - 1, len(yearIndexes)):
        # for yearIndex in range(2, 3):
            if yearIndex == len(yearIndexes) - 1:
                newDf = df.iloc[yearIndexes[0]:yearIndexes[-1]]
            else:
                newDf = df.iloc[yearIndexes[yearIndex]:yearIndexes[yearIndex + 1]]
            
            tradeInfo = self.record_tradInfo(newDf)
            tableDf = self.make_tableDf(tradeInfo, yearIndexes, yearIndex, newDf, companyName)
            
            if self.drawBestHold:
                self.draw_table(tableDf)
                self.draw_hold(file, df, newDf, yearIndexes, yearIndex, tradeInfo)
    
    def record_tradInfo(self, newDf):  # 記錄交易資訊
        tradeInfo = dict()
        
        buyX = [i for i in newDf.index if not pd.isna(newDf.at[i, 'buy'])]
        buyY = [i for i in newDf['buy'] if not pd.isna(i)]
        # buyX = newDf[newDf['buy'].notnull()].index
        # buyY = newDf['buy'].values[newDf.index.isin(buyX)]
        tradeInfo.update(self.make_Series('buy', buyX, buyY))
        
        sellTechConditionX = [i for i in newDf.index if not pd.isna(newDf.at[i, 'sell'])]
        sellTechConditionY = [i for i in newDf['sell'] if not pd.isna(i)]
        tradeInfo.update(self.make_Series('sell', sellTechConditionX, sellTechConditionY))
        
        sellDateX = [i for i in newDf.index if not pd.isna(newDf.at[i, 'sell date'])]
        sellDateY = [i for i in newDf['sell date'] if not pd.isna(i)]
        tradeInfo.update(self.make_Series('sell date', sellDateX, sellDateY))
        
        sellX = [i for i in newDf.index if not pd.isna(newDf.at[i, 'sell date']) or not pd.isna(newDf.at[i, 'sell'])]
        sellY = list(newDf['Price'].values[newDf.index.isin(sellX)])
        tradeInfo.update(self.make_Series('sell', sellX, sellY))
        
        tradeInfo = pd.Series(tradeInfo)
        return tradeInfo
    
    def make_Series(self, title, x, y):
        return {title : pd.Series(y, index=x, dtype='float64')}
    
    def make_tableDf(self, tradeInfo, yearIndexes, yearIndex, newDf, companyName):
        cellData =  dict()
        
        cellData.update({'buy Num': len(tradeInfo['buy'].index)})
        cellData.update({'sell Num': len(tradeInfo['sell'].index)})
        cellData.update({'sell': len(tradeInfo['sell'].index)})
        cellData.update({'sell date': len(tradeInfo['sell date'].index)})
        
        buyY = tradeInfo['buy'].copy()
        sellY = tradeInfo['sell'].copy()
        
        if len(tradeInfo['buy']) and len(tradeInfo['sell']) and tradeInfo['buy'].index[0] > tradeInfo['sell'].index[0]: # 去年買今年賣，插入去年買buyY的尾巴
            buyY = pd.concat([self.lastBuyY, buyY])
        
        if len(tradeInfo['buy']) and len(tradeInfo['sell']) and tradeInfo['buy'].index[-1] > tradeInfo['sell'].index[-1] or len(tradeInfo['buy']) == 1 and len(tradeInfo['sell']) == 0: # 今年買明年賣，記錄今年buyY的尾巴
            self.lastBuyY = pd.Series({tradeInfo['buy'].index[-1]:tradeInfo['buy'].values[-1]})
        
        tradeNum = len(sellY)
        if tradeNum != 0:
            winRate = str(round(len([i for i, j in zip(buyY, sellY) if j - i > 0]) / tradeNum * 100, 2)) + '%'
        else:
            winRate = 0
        cellData.update({'win rate': winRate})
        
        profit = 10000000.0
        for i, j in zip(buyY, sellY):
            stockNum = floor(profit / float(i))
            profit = profit - stockNum * float(i)
            profit += stockNum * float(j)
        ARR = pow(profit / 10000000, 1 / len(newDf))
        ARR = round((pow(ARR, 251.7) - 1) * 100, 2)
        cellData.update({'IRR': str(ARR) + '%'})
        
        tableDf = pd.DataFrame([cellData])
        
        if yearIndex == len(yearIndexes) - 1:
            self.companiesTradeInfo.update({companyName: tableDf})
        
        return tableDf
    
    def draw_table(self, tableDf):
        tableAx = self.fig.add_subplot(self.gs[0, :])
        tableAx.axis('off')
        topTable = tableAx.table(
            colLabels = tableDf.columns, 
            cellText = tableDf.values, 
            cellLoc = 'center', 
            colColours = ['silver'] * (len(tableDf.columns)), 
            )
        topTable.auto_set_font_size(False)
        topTable.set_fontsize('large')  # xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller, None
        
    def draw_hold(self, file, df, newDf, yearIndexes, yearIndex, tradeInfo):
        ax = self.fig.add_subplot(self.gs[1:, :])
        ax.plot(newDf.index, newDf['Price'], label='Price', color='steelblue', linewidth=4)
        ax.plot(newDf.index, newDf['hold 1'], label='Hold', color='darkorange', linewidth=4)
        ax.plot(newDf.index, newDf['hold 2'], color='darkorange', linewidth=4)
        
        # 不知道為什麼會有warning(畫點)
        # ax.scatter(newDf.index, newDf['buy'], label='buy', color='black', s=40, zorder=10)
        # ax.scatter(newDf.index, newDf['sell date'], label='sell date', color='lime', s=40, zorder=10)
        # ax.scatter(newDf.index, newDf['sell'], label='sell', color='yellow', s=40, zorder=10)
        
        # 也是畫點
        # if yearIndex != len(yearIndexes) - 1:
        #     for index in tradeInfo.index[:-1]:
        #         ax.scatter(tradeInfo[index].index, tradeInfo[index].values, 
        #                    color=self.scatterClr[index], 
        #                    s=40, 
        #                    zorder=10, 
        #                    label=index, 
        #                    marker=self.scatterMarker[index])
        
        # 買賣點畫直線，應該用不到，需要用的話還要再修改
        # buy = [i for i in newDf.index if not pd.isna(newDf.at[i,'buy'])]
        # sell = [i for i in newDf.index if not pd.isna(newDf.at[i,'sell date'])]
        # plt.vlines(buy, color='darkorange', linestyle='-',alpha=0.5,label='buy',ymin=0,ymax=max(newDf['Price']))
        # plt.vlines(sell, color='purple', linestyle='-',alpha=0.5,label='sell date',ymin=0,ymax=max(newDf['Price']))
        
        mIndex = []
        if yearIndex == len(yearIndexes) - 1:
            mIndex = yearIndexes.copy()
            mIndex[-1] = mIndex[-1] - 1
        else:
            month = 0
            for i, date in enumerate(newDf.index):
                nowMonth = date.split('-')[1]
                if month != nowMonth:
                    mIndex.append(i)
                    month = nowMonth
            mIndex.append(len(newDf) - 1)
        
        if yearIndexes[1] - yearIndexes[0] < 20 and yearIndex == len(yearIndexes) - 1:
            ax.set_xticks(mIndex[1:], fontsize=draw_hold_period.allFontSize)
        else:
            ax.set_xticks(mIndex, fontsize=draw_hold_period.allFontSize)
        # plt.setp(ax.get_xticklabels(), ha="left", rotation=-45)
        ax.set_xlabel('Date', fontsize=draw_hold_period.allFontSize)
        ax.set_ylabel('Price', fontsize=draw_hold_period.allFontSize)
        ax.grid()
        handles, labels = ax.get_legend_handles_labels()
        self.fig.legend(
            handles, labels, 
            loc='upper center', 
            bbox_to_anchor=(0.5, 0), 
            fancybox=True, shadow=False, 
            ncol=len(newDf.columns), 
            fontsize=draw_hold_period.allFontSize)
        fileTitle = file.replace('.csv', '_')
        if yearIndex == len(yearIndexes) - 1:
            fileTitle += df.index[0].split('-')[0] + '-' + df.index[len(df.index) - 1].split('-')[0]
        else:
            fileTitle += newDf.index[0].split('-')[0]
        print(fileTitle)
        figTitle = fileTitle.replace('_', ' ')
        if len(self.tech.split('_')) > 1:
            figTitle = f'{self.tech} ' + ' '.join(figTitle.split(' ')[len(self.tech.split('_')):])
        self.fig.suptitle(
            figTitle, 
            y=1,
            fontsize=draw_hold_period.allFontSize)
        self.fig.savefig(f'{fileTitle}.png', dpi=draw_hold_period.fig.dpi, bbox_inches='tight')
        plt.clf()
    
    def output_tradeInfo(self):
        os.chdir('../')
        filename = f'{self.trainOrTest + self.algoOrTrad}' + f'_tradeInfo_{self.tech}.csv'
        for dfIndex, companyName in enumerate(self.companiesTradeInfo):
            # eachDf.insert(0, 'company', eachDfIndex)  # add new column to first position
            self.companiesTradeInfo[companyName].rename(index={0: companyName}, inplace=True)
            if dfIndex == 0:
                self.companiesTradeInfo[companyName].to_csv(filename)
            else:
                self.companiesTradeInfo[companyName].to_csv(filename, mode='a', header=None)
    
x = draw_hold_period(
    expFolder='exp_result', 
    resultFolder='result_2021', 
    tech='SMA_RSI_3', 
    bestHold=True, 
    fundLV=False, 
    isTrain=False, 
    isTradition=False, 
    delFile=[False, ''],  # 刪除畫的圖或csv用，False不執行，True且string內放副檔名代表要刪除該副檔名的檔案
    setCompany='all')  # all全部公司，或是特定幾間公司('AAPL,AXP,WBA')

# bestHold == True and fundLV == True: 畫最好滑動視窗的資金水位
# bestHold == True and fundLV == False: 畫最好滑動視窗的持有區間
# bestHold == False and fundLV == True: 畫所有滑動視窗的資金水位