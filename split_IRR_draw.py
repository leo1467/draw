from math import floor
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.ticker as mtick

file_extension = '.csv'
root = os.getcwd()

class split_IRR_draw:
    # 設定滑動視窗group
    slidingLableClrList = [
        [['YYY2YYY', 'YYY2YY', 'YYY2Y', 'YYY2YH', 'YY2YY', 'YY2YH', 'YY2Y', 'YH2YH', 'YH2Y', 'Y2Y'], 'gold'],
        [['YYY2H', 'YYY2Q', 'YYY2M', 'YY2H', 'YY2Q', 'YY2M', 'YH2H', 'YH2Q', 'YH2M', 'Y2H', 'Y2Q', 'Y2M'], 'limegreen'],
        [['H#', 'H2H', 'H2Q', 'H2M', 'Q#', 'Q2Q', 'Q2M', 'M#', 'M2M'], 'r'],
        [['20D20', '20D15', '20D10', '20D5', '15D15', '15D10', '15D5', '10D10', '10D5', '5D5'], 'grey'],
        [['4W4', '4W3', '4W2', '4W1', '3W3', '3W2', '3W1', '2W2', '2W1', '1W1'], 'darkgoldenrod'],
        [['5D4', '5D3', '5D2', '4D4', '4D3', '4D2', '3D3', '3D2', '2D2'], 'w']
        ]
    # slidingLableClrList = [ # 
    #     [['YYY2YYY', 'YYY2YY', 'YYY2YH', 'YYY2Y', 'YYY2H', 'YYY2Q', 'YYY2M', ], 'gold'],
    #     [['YY2YY', 'YY2YH', 'YY2Y', 'YY2H', 'YY2Q', 'YY2M', ], 'limegreen'], 
    #     [['YH2YH', 'YH2Y', 'YH2H', 'YH2Q', 'YH2M', ], 'purple'], 
    #     [['Y2Y', 'Y2H', 'Y2Q', 'Y2M', ], 'pink'], 
    #     [['H#', 'H2H', 'H2Q', 'H2M', 'Q#', 'Q2Q', 'Q2M', 'M#', 'M2M'], 'r'],
    #     [['20D20', '20D15', '20D10', '20D5', '15D15', '15D10', '15D5', '10D10', '10D5', '5D5'], 'grey'],
    #     [['4W4', '4W3', '4W2', '4W1', '3W3', '3W2', '3W1', '2W2', '2W1', '1W1'], 'darkgoldenrod'],
    #     [['5D4', '5D3', '5D2', '4D4', '4D3', '4D2', '3D3', '3D2', '2D2'], 'w']
    #     ]
    reorderList = [
        'B&H',
        'YYY2YYY', 'YYY2YY', 'YYY2YH', 'YYY2Y', 'YYY2H', 'YYY2Q', 'YYY2M', 
        'YY2YY', 'YY2YH', 'YY2Y', 'YY2H', 'YY2Q', 'YY2M', 
        'YH2YH', 'YH2Y', 'YH2H', 'YH2Q', 'YH2M', 
        'Y2Y', 'Y2H', 'Y2Q', 'Y2M', 
        'H#', 'H2H', 'H2Q', 'H2M', 'Q#', 'Q2Q', 'Q2M', 'M#', 'M2M', 
        '20D20', '20D15', '20D10', '20D5', '4W4', '4W3', '4W2', '4W1', 
        '15D15', '15D10', '15D5', '3W3', '3W2', '3W1', 
        '10D10', '10D5', '2W2', '2W1', 
        '5D5', '1W1', '5D4', '5D3', '5D2', 
        '4D4', '4D3', '4D2', 
        '3D3', '3D2', 
        '2D2']
    # reorderList = [
    #     'B&H',
    #     'YYY2YYY', 'YYY2YY', 'YYY2Y', 'YYY2YH', 'YY2YY', 'YY2YH', 'YY2Y', 'YH2YH', 'YH2Y', 'Y2Y', 
    #     'YYY2H', 'YYY2Q', 'YYY2M', 'YY2H', 'YY2Q', 'YY2M', 'YH2H', 'YH2Q', 'YH2M', 'Y2H', 'Y2Q', 'Y2M', 
    #     'H#', 'H2H', 'H2Q', 'H2M', 'Q#', 'Q2Q', 'Q2M', 'M#', 'M2M', 
    #     '20D20', '20D15', '20D10', '20D5', '15D15', '15D10', '15D5', '10D10', '10D5', '5D5', 
    #     '4W4', '4W3', '4W2', '4W1', '3W3', '3W2', '3W1', '2W2', '2W1', '1W1', 
    #     '5D4', '5D3', '5D2', '4D4', '4D3', '4D2', '3D3', '3D2', '2D2'
    #     ]
    # reorderList = [
    #     'B&H',
    #     'YYY2YYY', 'YYY2YY' ,'YYY2YH', 'YY2YY', 'YYY2Y', 'YY2YH', 'YYY2H', 'YYY2Q', 'YYY2M', 'YH2YH', 'YY2Y', 'YH2Y', 'YY2H', 'YY2Q', 'YY2M', 'Y2Y', 'YH2H', 'YH2Q', 'YH2M', 'Y2H', 'Y2Q', 'Y2M', 'H#', 'H2H', 'H2Q', 'H2M', 'Q#', 'Q2Q', 'Q2M', 'M#', 'M2M', 
    #     '20D20', '4W4', '20D15', '4W3', '20D10', '4W2', '20D5', '4W1', '15D15', '3W3', '15D10', '3W2', '15D5', '3W1', '10D10', '2W2', '10D5', '2W1', '5D5', '1W1', 
    #     '5D4', '5D3', '5D2', '4D4', '4D3', '4D2', '3D3', '3D2', '2D2'
    #     ]
    # 設定bar屬性
    barColorSet = ['steelblue', 'darkorange', 'paleturquoise', 'wheat', 'lightcyan', 'lightyellow', 'red']
    BHColor = 'r'
    totalBarWidth = 0.80
    # 儲存全部table資料
    tables = {}
    # 設定figure大小
    # plt.rcParams['figure.figsize'] = (21, 9)
    fig = plt.figure(
        figsize=[21, 9], 
        dpi=300, 
        # figCnt+1, 
        # sharey = True, 
        # gridspec_kw={'height_ratios': grid}, 
        constrained_layout=True
        )
    allFontSize = 14
    
    def __init__(self, IRRFileName, splitIRRFile, drawTable, drawBar, seperateTable, reorder, setCompany):
        if type(IRRFileName) == list:
            # if not os.path.isdir('windowRank'):
            #     os.mkdir('windowRank')
            os.chdir('windowRank')
            for file in IRRFileName:
                self.draw_rank(file)
        else:    
            split_IRR_draw.seperateTable = seperateTable or drawBar
            split_IRR_draw.reorder = reorder
            if split_IRR_draw.reorder:
                split_IRR_draw.reorderList.reverse()
            self.IRRFileName = IRRFileName + '.csv'
            self.dirName = 'split_' + self.IRRFileName.split('.')[0]
            
            print(self.IRRFileName)
            self.process_fileName_dir()
            
            if splitIRRFile:
                self.split_csv()
            else:
                os.chdir(self.dirName)
            
            if setCompany == 'all':
                self.allIRRFile = [i for i in glob.glob(f"*{file_extension}")]
            else:
                self.allIRRFile = [i for i in glob.glob(f"*{file_extension}") if setCompany in i]
            
            for FileIndex, file in enumerate(self.allIRRFile):
                processACompany = self.ProcessACompany(FileIndex, file)
                fig = split_IRR_draw.fig
                gs, gridNum = processACompany.set_grid(fig)
                if drawTable:
                    processACompany.start_draw_tables(fig, gs)
                if drawBar:
                    processACompany.draw_bar(fig, gs, gridNum)
                # break
            plt.close(split_IRR_draw.fig)
            
            # exit(0)
            os.chdir(root)
            OutputTableFileName = IRRFileName.split('.')[0] + '_tables.csv'
            for dfIndex, companyName in enumerate(self.tables):
                self.tables[companyName].rename(index={0: companyName}, inplace=True)
                if dfIndex == 0:
                    self.tables[companyName].to_csv(OutputTableFileName)
                else:
                    self.tables[companyName].to_csv(OutputTableFileName, mode='a', header=None)
    
    def split_csv(self):
        self.IRRdf = pd.read_csv(self.IRRFileName, index_col=False)
        if True in self.IRRdf.columns.str.contains('^Unnamed'):
            self.IRRdf = self.IRRdf.loc[:, ~self.IRRdf.columns.str.contains('^Unnamed')]
            self.IRRdf.to_csv(self.IRRFileName, index=None)
        self.IRRdf = self.IRRdf.T.reset_index().T.reset_index(drop=True)
        os.chdir(self.dirName)
        index = [i for i in self.IRRdf.index if self.IRRdf.at[i, 0][0] == '=']
        index.append(len(self.IRRdf))
        for cellIndex in range(len(index) - 1):
            companyName = self.IRRdf.at[index[cellIndex], self.IRRdf.columns[0]].replace('=', '')
            self.IRRdf.at[index[cellIndex], self.IRRdf.columns[0]] = 'window'
            self.IRRdf[index[cellIndex]:index[cellIndex + 1]].to_csv(companyName + '_IRR.csv', header=None, index=None)
    
    def process_fileName_dir(self):
        IRRFileNameList = self.IRRFileName.split('.')[0].split('_')
        split_IRR_draw.allTitle = '_'.join(IRRFileNameList[IRRFileNameList.index('sorted') + 1:])
        split_IRR_draw.trainOrTest = IRRFileNameList[0]
        if not os.path.isdir(self.dirName):
            os.mkdir(self.dirName)
    
    def draw_rank(self, IRRFileName):
        rankDf = pd.read_csv(IRRFileName + '.csv', index_col='window')
        rankDf = rankDf.drop('B&H')
        ax1, ax2, ax3 = self.fig.subplots(nrows=3, sharey=True)
        rankPlot = []
        for index, ax in enumerate([ax1, ax2, ax3]):
            if index == 0:
                subDf = rankDf.iloc[:20]
            elif index == 1:
                subDf = rankDf.iloc[20:40]
            elif index == 2:
                subDf = rankDf.iloc[40:]
            
            plotBar = subDf.plot.bar(
                ax=ax, 
                # width=split_IRR_draw.totalBarWidth, 
                rot=0, 
                # color=self.colorDict, 
                edgecolor='black', 
                linewidth=0.2, 
                legend=None
                )
            rankPlot.append(plotBar)
            
            ax.set_xticklabels(subDf.index, rotation=45)
            ax.set(xlabel='', ylabel='')
            ax.grid(axis='y')
            ax.tick_params(axis='x', labelsize=split_IRR_draw.allFontSize + 6)  #設定xlabel ylabel字形大小
            ax.tick_params(axis='y', labelsize=10)
            for cellText in ax.get_xticklabels():
                    txt = cellText.get_text()
                    for slideGroup in self.slidingLableClrList:
                        if txt in slideGroup[0]:
                            plt.setp(cellText, bbox=dict(boxstyle='round, pad=0.15', edgecolor='none', alpha=1, facecolor=slideGroup[1]))
                            break
        
        handles, labels = rankPlot[0].get_legend_handles_labels()
        
        self.fig.legend(
            # handles, ['accumulated rank'], 
            handles, labels, 
            loc='upper right', 
            bbox_to_anchor=(1, 1.05), 
            fancybox=True, shadow=False, 
            ncol=1, 
            fontsize=split_IRR_draw.allFontSize)
        'windowRank_train_SMA_Tradition'
        fileNameSplit = IRRFileName.split('_')
        self.fig.suptitle(fileNameSplit[2] + ' ' + fileNameSplit[1]+ ' ' + fileNameSplit[3] + ' ' + 'window ranking', 
                    ha='left', 
                    x=0.025, 
                    y=1.03, 
                    fontsize=split_IRR_draw.allFontSize)
        if 'B&H' not in rankDf.columns:
            IRRFileName += '_noBH'
        self.fig.savefig(IRRFileName + '.png', dpi=self.fig.dpi, bbox_inches='tight')
        plt.clf()

    class ProcessACompany:
        def __init__(self, FileIndex, file):
            self.IRRData = dict()
            self.cellData = dict()
            self.tableNum = 0
            self.tablePlotObjs = list()
            print(file)
            if file.split('.')[1] == 'csv':
                self.process_df(file)
                self.find_techNames()
                self.process_IRRFile(FileIndex, file)

        def process_df(self, file):
            # df前處理
            self.company = file.split('_')[0]
            self.df = pd.read_csv(file, index_col=0)
            # df.rename(columns = {df.columns[0]: 'window'}, inplace = True)
            for colIndex in self.df.columns:
                for rowIndex in self.df.index:
                    self.df.at[rowIndex, colIndex] *= 100
        
        def find_techNames(self):
            self.techNames = [name.split(' ')[0] for name in self.df.columns if 'B&H' not in name]
            if "window num" in self.df.columns:
                self.techNames = [tech for tech in self.df.columns[:-1]]
            else:
                self.techNames = [y for x, y in enumerate(self.techNames) if x % 2 == 0]
            self.techNum = len(self.techNames)
            self.titleTechNames = [i + '"' for i in ['"' + j for j in self.techNames]]
            self.mixedTech = (lambda x: True if len(x.split('_')) > 1 else False)(self.techNames[0])
        
        def process_IRRFile(self, FileIndex, file):
            # table資料
            for colIndex, col in enumerate(self.df.columns):
                self.df.sort_values(by=col, ascending=False, inplace=True)
                self.IRRData.update({col: pd.Series({self.df.index[i]: x for i, x in enumerate(self.df[col]) if self.df.index[i] != 'B&H'})})
            
            if split_IRR_draw.reorder:
                self.df = self.df.reindex(split_IRR_draw.reorderList)
            else:
                self.df.sort_values(by=self.df.columns[0], ascending=False, inplace=True)
            
            if "window num" in self.df.columns:  # 如果是計算不同指數出現次數，下面就不用做
                return
            
            for i in range(self.techNum):
                self.add_info(self.techNames[i], '', i * 2, i * 2 + 1, False)
            
            if self.techNum > 1:
                for techIndex, row in zip(range(1, self.techNum), range(2, self.techNum * 2, 2)):
                    self.add_info(self.techNames[0], self.techNames[techIndex], 0, row, True)
            
            if self.mixedTech and 'choose' in self.df.columns[-1]:
                windowChoose = list()
                for techIndex, col in zip(range(1, self.techNum), range(self.techNum * 2, len(self.df.columns), 2)):
                    windowChoose.append(self.df[self.df.columns[col]] / self.df['window num'] * 100)
                for i in range(len(windowChoose)):
                    windowChoose[i] = ['%.2f' % elem + '%' for elem in windowChoose[i]]
                self.windowChooseDf = pd.DataFrame(windowChoose, columns=[self.df.index], index=[self.techNames[1:]])
                self.df = self.df.drop(columns = [col for col in self.df.columns[self.techNum * 2: ]])
            
            for cellIndex in self.cellData:
                try:
                    if 'ranking' not in cellIndex and 'positive' not in cellIndex:
                        cellData = str(round(float(self.cellData[cellIndex]), 2)) + '%'
                    else:
                        continue
                except ValueError:
                    continue
                self.cellData[cellIndex] = cellData
            self.tableDf = pd.DataFrame([self.cellData])
            split_IRR_draw.tables.update({self.company: self.tableDf})
        
        def add_info(self, comp1, comp2, col1, col2, techCompare):
            IRRDataAlgoCol1 = self.IRRData[self.df.columns[col1]]
            IRRDataAlgoCol2 = self.IRRData[self.df.columns[col2]]
            IRRDataTradCol1 = self.IRRData[self.df.columns[col1 + 1]]
            dfCol1 = self.df[self.df.columns[col1]].sort_values(ascending=False)
            dfCol2 = self.df[self.df.columns[col1 + 1]].sort_values(ascending=False)
            if 'B&H' in self.IRRData:
                BHCol = self.IRRData['B&H'].iloc[:-1]
            else:
                BHCol = self.df[self.df.columns[0]].iloc[self.df.index.get_loc('B&H'):self.df.index.get_loc('B&H') + 1]
            if not techCompare:
                data = {
                    # 'GNQTS best window': IRRDataAlgoCol1.index[0],
                    # 'GNQTS worst window': IRRDataAlgoCol1.index[-1], 
                    # 'Traditional best window': IRRDataAlgoCol2.index[0],
                    # 'Traditional worst window': IRRDataAlgoCol2.index[-1], 
                    # 'B&H IRR': self.df.at['B&H', self.df.columns[col1]], 
                    # 'GNQTS highest IRR': IRRDataAlgoCol1[0], 
                    # 'GNQTS average IRR': np.average(IRRDataAlgoCol1), 
                    # 'Traditional highest IRR': IRRDataTradCol1[0], 
                    # 'Traditional average IRR': np.average(IRRDataAlgoCol2), 
                    f'{comp1} GNQTS b/w w': IRRDataAlgoCol1.index[0] + '/' + IRRDataAlgoCol1.index[-1],
                    f'{comp1} Traditional b/w w': IRRDataAlgoCol2.index[0] + '/' + IRRDataAlgoCol2.index[-1],
                    f'{comp1} GNQTS best IRR': IRRDataAlgoCol1[0], 
                    f'{comp1} GNQTS average IRR': np.average(IRRDataAlgoCol1), 
                    f'{comp1} Traditional best IRR': IRRDataTradCol1[0], 
                    f'{comp1} Traditional average IRR': np.average(IRRDataAlgoCol2), 
                    f'{comp1} best B&H': BHCol.values.max(), 
                    f'{comp1} B&H average': np.average(BHCol.values),
                    # f'{comp1} yellow w': np.average([values for index, values in IRRDataAlgoCol1.items() if index in split_IRR_draw.slidingLableClrList[0][0]]), 
                    # f'{comp1} green w': np.average([values for index, values in IRRDataAlgoCol1.items() if index in split_IRR_draw.slidingLableClrList[1][0]]), 
                    # f'{comp1} red w': np.average([values for index, values in IRRDataAlgoCol1.items() if index in split_IRR_draw.slidingLableClrList[2][0]]), 
                    # f'{comp1} grey w': np.average([values for index, values in IRRDataAlgoCol1.items() if index in split_IRR_draw.slidingLableClrList[3][0]]), 
                    # f'{comp1} brown w': np.average([values for index, values in IRRDataAlgoCol1.items() if index in split_IRR_draw.slidingLableClrList[4][0]]), 
                    # f'{comp1} white w': np.average([values for index, values in IRRDataAlgoCol1.items() if index in split_IRR_draw.slidingLableClrList[5][0]]), 
                    # [f'{comp1} highest IRR diff GNQTS/Traditional', IRRDataCol1[0] - IRRDataCol2[0]], 
                    
                    # f'{comp1} GNQTS win rate': len([i for i, j in zip(IRRDataAlgoCol1.sort_index(), IRRDataAlgoCol2.sort_index()) if i >= j]) / len(IRRDataAlgoCol1) * 100, 
                    # f'{comp1} GNQTS B&H ranking': str(dfCol1.index.get_loc('B&H') + 1),
                    # f'{comp1} Traditional B&H ranking': str(dfCol2.index.get_loc('B&H') + 1), 
                    # f'{comp1} GNQTS positive IRR window': len([i for i in IRRDataAlgoCol1 if i >= 0]), 
                    # f'{comp1} Traditional positive IRR window': len([i for i in IRRDataAlgoCol2 if i >= 0]), 
                    
                    # [f'{comp1} highest IRR diff GNQTS/B&H', IRRDataCol1[0] - self.df.at['B&H', self.df.columns[col1]]], 
                    # [f'{comp1} highest IRR diff Traditional/B&H', IRRDataCol2[0] - self.df.at['B&H', self.df.columns[col2]]], 
                    # f'{comp1} GNQTS "Y" avg IRR': np.average([IRR for w, IRR in zip(self.df.index, dfCol1) if w[0] == 'Y']), 
                    # f'{comp1} GNQTS "HQM" avg IRR': np.average([IRR for w, IRR in zip(self.df.index, dfCol1) if w[0] == 'H' or w[0] == 'Q' or w[0] == 'M']), 
                    # f'{comp1} GNQTS "DW" avg IRR': np.average([IRR for w, IRR in zip(self.df.index, dfCol1) if 'W' in w or 'D' in w]), 
                }
                self.dataColLen = len(data)
            else:
                IRRDataTradCol2 = self.IRRData[self.df.columns[col2 + 1]]
                data = {
                    f'GNQTS win {comp1}/{comp2}': len([i for i, j in zip(IRRDataAlgoCol1.sort_index(), IRRDataAlgoCol2.sort_index()) if i >= j]) / len(IRRDataAlgoCol1) * 100, 
                    f'Traditional win {comp1}/{comp2}': len([i for i, j in zip(IRRDataTradCol1.sort_index(), IRRDataTradCol2.sort_index()) if i >= j]) / len(IRRDataTradCol1) * 100, 
                    # f'highest IRR diff of GNQTS {comp1}/{comp2}': IRRDataCol1[0] - IRRDataCol2[0],
                    # f'highest IRR diff of Traditional {comp1}/{comp2}': max(self.IRRData[self.df.columns[col1 + 1]]) - max(IRRDataTradCol2),
                    # f'GNQTS highest IRR gain {comp1}/{comp2} ': (IRRDataAlgoCol1[0] - IRRDataAlgoCol2[0]) / IRRDataAlgoCol2[0] * 100, 
                    # f'GNQTS avg IRR gain {comp1}/{comp2}': (np.average(IRRDataAlgoCol1) - np.average(IRRDataAlgoCol2)) / np.average(IRRDataAlgoCol2) * 100,
                    # f'Traditional highest IRR gain {comp1}/{comp2} ': (IRRDataTradCol1[0] - IRRDataTradCol2[0]) / IRRDataTradCol2[0] * 100, 
                    # f'Traditional avg IRR gain {comp1}/{comp2}': (np.average(IRRDataTradCol1) - np.average(IRRDataTradCol2)) / np.average(IRRDataTradCol2) * 100,
                }
                # if len(self.IRRData) > 3:
                #     for techName in self.techNames:
                #         data.update({techName: 1})
                self.frontLen = len(data)
                backData = {
                    # 訓練期才用
                    # f'GNQTS "Y" win {comp1}/{comp2}': str(len([w for w in self.df.index if self.compare(w, col1, col2) and w[0] == 'Y'])) + '/' + str(len([w for w in self.df.index if w[0] == 'Y'])),
                    # f'GNQTS "H" "Q" "M" win {comp1}/{comp2}': str(len([w for w in self.df.index if self.compare(w, col1, col2) and (w[0] == 'H' or w[0] == 'Q' or w[0] == 'M')])) + '/' + str(len([w for w in self.df.index if w[0] == 'H' or w[0] == 'Q' or w[0] == 'M'])),
                    # f'GNQTS "D" "W" win {comp1}/{comp2}': str(len([w for w in self.df.index if self.compare(w, col1, col2) and ('W' in w or 'D' in w)])) + '/' + str(len([w for w in self.df.index if ('W' in w or 'D' in w)])),
                    
                    # f'Traditional "Y" win {comp1}/{comp2}': str(len([w for w in self.df.index if self.compare(w, col1 + 1, col2 + 1) and w[0] == 'Y'])) + '/' + str(len([w for w in self.df.index if w[0] == 'Y'])),
                    # f'Traditional "H" "Q" "M" win {comp1}/{comp2}': str(len([w for w in self.df.index if self.compare(w, col1 + 1, col2 + 1) and (w[0] == 'H' or w[0] == 'Q' or w[0] == 'M')])) + '/' + str(len([w for w in self.df.index if w[0] == 'H' or w[0] == 'Q' or w[0] == 'M'])),
                    # f'Traditional "D" "W" win {comp1}/{comp2}': str(len([w for w in self.df.index if self.compare(w, col1 + 1, col2 + 1) and ('W' in w or 'D' in w)])) + '/' + str(len([w for w in self.df.index if ('W' in w or 'D' in w)])),
                    
                    # f'GNQTS "Y" win {comp1}/{comp2}': len([w for w in self.df.index if self.compare(w, col1, col2) and w[0] == 'Y']) / len([w for w in self.df.index if w[0] == 'Y']) * 100,
                    # f'GNQTS "H" "Q" "M" win {comp1}/{comp2}': len([w for w in self.df.index if self.compare(w, col1, col2) and (w[0] == 'H' or w[0] == 'Q' or w[0] == 'M')]) / len([w for w in self.df.index if w[0] == 'H' or w[0] == 'Q' or w[0] == 'M']) * 100,
                    # f'GNQTS "D" "W" win {comp1}/{comp2}': len([w for w in self.df.index if self.compare(w, col1, col2) and ('W' in w or 'D' in w)]) / len([w for w in self.df.index if ('W' in w or 'D' in w)]) * 100,
                    
                    # f'Traditional "Y" win {comp1}/{comp2}': len([w for w in self.df.index if self.compare(w, col1 + 1, col2 + 1) and w[0] == 'Y']) / len([w for w in self.df.index if w[0] == 'Y']) * 100,
                    # f'Traditional "H" "Q" "M" win {comp1}/{comp2}': len([w for w in self.df.index if self.compare(w, col1 + 1, col2 + 1) and (w[0] == 'H' or w[0] == 'Q' or w[0] == 'M')]) / len([w for w in self.df.index if w[0] == 'H' or w[0] == 'Q' or w[0] == 'M']) * 100,
                    # f'Traditional "D" "W" win {comp1}/{comp2}': len([w for w in self.df.index if self.compare(w, col1 + 1, col2 + 1) and ('W' in w or 'D' in w)]) / len([w for w in self.df.index if ('W' in w or 'D' in w)]) * 100,
                    # 訓練期才用
                }
                self.backLen = len(backData)
                for elem in backData.items():
                    data.update({elem})
                self.finalCompareLen = len(data)
            for elem in data.items():
                self.cellData.update({elem})
            if len(data) > 0:
                self.tableNum += 1
        
        def compare(self, window, col1, col2):
            return self.df.at[window, self.df.columns[col1]] >= self.df.at[window, self.df.columns[col2]]
        
        def set_grid(self, fig):
            gridNum = (lambda x: 24 if x == 1 else 25)(split_IRR_draw.seperateTable)
            if self.techNum == 1:
                hspace = -0.9
            elif self.techNum == 2:
                hspace = -0.57
            elif self.techNum == 3:
                hspace = -0.25
            
            if split_IRR_draw.seperateTable:
                gs = fig.add_gridspec(gridNum, 1)
            else:
                gs = fig.add_gridspec(
                    gridNum, 1, 
                    wspace=0, 
                    hspace=hspace, 
                    top=1, 
                    bottom=0, 
                    # left=0.17, 
                    # right=0.845
                    )
            return gs, gridNum
        
        def start_draw_tables(self, fig, gs):
            # 設定top table
            self.draw_tables(fig, gs)
            if split_IRR_draw.seperateTable:
                # fig.suptitle(self.company + " " + split_IRR_draw.trainOrTest + ' ' + ' '.join(self.titleTechNames) + ' compare table', 
                    # fontsize=split_IRR_draw.allFontSize + 8)
                fig.suptitle(' ')
                fig.savefig(self.company + '_table'  + '.png', dpi=fig.dpi, bbox_inches='tight')
                plt.clf()
                self.tablePlotObjs.clear()
        
        def draw_tables(self, fig, gs):
            for i, j in zip(range(0, self.dataColLen * self.techNum, self.dataColLen), range(self.techNum)):
                tableAx = fig.add_subplot(gs[j, :])
                tableAx.axis('off')
                self.draw_table(tableAx, i, i + self.dataColLen)
            
            # 比較table中不同指標誰比較大，並著色
            if len(self.techNames) > 1:
                for colNum in range(self.dataColLen):
                    self.find_big_cell(colNum)

            # 設定比較table
            if len(self.techNames) > 1:
                for i in range(self.tableNum - self.techNum):
                    tableAx = fig.add_subplot(gs[self.techNum + i, :])
                    tableAx.axis('off')
                    startCol = self.dataColLen * self.techNum + i * self.finalCompareLen
                    endCol = startCol + self.finalCompareLen
                    self.draw_table(tableAx, startCol, endCol)
        
        def draw_table(self, tableAx, startCol, endCol):
            tmpTableDf = self.tableDf.iloc[:, startCol:endCol]
            topTable = tableAx.table(
                colLabels=tmpTableDf.columns, 
                cellText=tmpTableDf.values, 
                # loc='best', 
                cellLoc='center', 
                # colColours=['silver'] * (len(tmpTableDf.columns)), 
                colColours=[(lambda x: 'silver' if len(self.tablePlotObjs) < self.techNum else 'darkorange')(0)] * (len(tmpTableDf.columns)), 
                bbox=[0, 1, 1, 2]
                )
            self.tablePlotObjs.append(topTable)
            # for colIndex in range(len(tableDf.columns)):  #設定cell text顏色
            #     topTable[0, colIndex].get_text().set_color('white')
            # topTable.auto_set_column_width(col=list(range(len(self.tableDf.columns))))
            topTable.auto_set_font_size(False)
            topTable.set_fontsize('large')  # Valid font size are xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller, None
            
            # 看每個視窗勝率時可以分顏色
            if len(self.tablePlotObjs) > self.techNum and True in [True for i in tmpTableDf if 'win' in i]:
                # topTable.auto_set_column_width(col=list(range(len(self.tableDf.columns))))
                for i in range(self.backLen):
                    if i < 3:
                        topTable[0, self.frontLen + i].set_color('lightsteelblue')
                    else:
                        topTable[0, self.frontLen + i].set_color('bisque')
                    topTable[0, self.frontLen + i].set_edgecolor('black')
        
        def find_big_cell(self, colNum):
            cellCompare = list()
            for eachTable in self.tablePlotObjs:
                cellCompare.append(eachTable[1, colNum])
            compare = -1000
            for cell in cellCompare:
                try:
                    cellData = float(cell.get_text()._text.strip('%'))
                except ValueError:
                    continue
                if cellData > compare:
                    compare = cellData
            bigCell = list()
            for cell in cellCompare:
                try:
                    cellData = float(cell.get_text()._text.strip('%'))
                except ValueError:
                    continue
                if cellData == compare:
                    bigCell.append(cell)
            for cell in bigCell:
                cell.set_color('lime')
                cell.set_edgecolor('black')
        
        def draw_bar(self, fig, gs, gridNum):
            # 設定每個bar的顏色及bar的最終寬度
            colorDict = dict(zip(self.df.columns, split_IRR_draw.barColorSet))
            if len(self.df.columns) < 3:
                split_IRR_draw.totalBarWidth = 0.5
            
            # 設定要畫幾個子圖
            if split_IRR_draw.reorder:
                figCnt = 1
            else:
                # figCnt = (lambda x: 2 if x < 5 else 3)(len(self.df.columns))
                figCnt = 3
            
            # 將過長的df切開
            dfCuttedIndex = list()
            for i in range(figCnt):
                dfCuttedIndex.append(floor(len(self.df) / figCnt) * i)
            dfCuttedIndex.append(len(self.df))
            
            # 找出plot bar要佔用哪些grid
            startGrid = (lambda tableObjSize: 0 if tableObjSize == 0 else (tableObjSize - 1) * 2 - 1)(len(self.tablePlotObjs))
            figJump = ceil((gridNum - startGrid) / figCnt)
            figGrid = [startGrid]
            for i in range(figCnt):
                figGrid.append(figGrid[i] + figJump)
            
            # plot bar
            barAxes = list()
            for splitIndex in range(figCnt):
                # 設定每個子圖佔多少grid
                if splitIndex == 0:
                    barAx = fig.add_subplot(gs[figGrid[splitIndex]:figGrid[splitIndex + 1], :])
                else:
                    barAx = fig.add_subplot(gs[figGrid[splitIndex]:figGrid[splitIndex + 1], :], sharey=barAxes[0])
                barAxes.append(barAx)
                
                # 分割需要畫的df出來
                if split_IRR_draw.reorder and "window num" not in self.df.columns:
                    subDf = self.df.iloc[dfCuttedIndex[splitIndex]:dfCuttedIndex[splitIndex + 1] - 1, [0]]
                elif split_IRR_draw.reorder and "window num" in self.df.columns:
                    subDf = self.df.iloc[dfCuttedIndex[splitIndex]:dfCuttedIndex[splitIndex + 1] - 1, :-1]
                else:
                    subDf = self.df.iloc[dfCuttedIndex[splitIndex]:dfCuttedIndex[splitIndex + 1]]
                
                plot = subDf.plot.bar(
                    ax=barAx, 
                    width=split_IRR_draw.totalBarWidth, 
                    rot=0, 
                    color=colorDict, 
                    edgecolor='black', 
                    linewidth=0.2, 
                    legend=None
                    )
                
                # 找出B&H位置，將B&H的bar變成紅色
                BHIndex = [i for i, x in enumerate(subDf.index) if x == 'B&H']
                if not len(BHIndex) and not split_IRR_draw.reorder:
                    axIndexForLegned = splitIndex
                elif split_IRR_draw.reorder:
                    axIndexForLegned = 0
                if len(BHIndex):
                    BHIndex = BHIndex[0]
                    for barIndex, barContainer in enumerate(plot.containers):
                        if barIndex == 0:
                            singleBarWidth = barContainer[BHIndex].get_width()
                            barX = ((barContainer[BHIndex].get_x() + (barContainer[BHIndex].get_x() + (len(subDf.columns) * singleBarWidth))) - singleBarWidth) / 2
                        barContainer[BHIndex].set_color(split_IRR_draw.BHColor)
                        barContainer[BHIndex].set_x(barX)
                        barContainer[BHIndex].set_edgecolor('black')
                
                # 如果是train，變更每個滑動視窗的B&H顏色
                if split_IRR_draw.trainOrTest == 'train' and not split_IRR_draw.reorder:
                    for singleBar in plot.containers[-1]:
                        singleBar.set_color(split_IRR_draw.BHColor)
                        singleBar.set_edgecolor('black')
                
                # 設定其他屬性
                barAx.grid(axis='y')
                barAx.yaxis.set_major_formatter(mtick.PercentFormatter())  #把座標變成%
                # barAx.locator_params(axis='y', nbins=10)
                barAx.set_xticklabels(subDf.index, rotation=(lambda x: 90 if x else 45)(split_IRR_draw.reorder))
                barAx.set(xlabel='', ylabel='')
                barAx.tick_params(axis='x', labelsize=split_IRR_draw.allFontSize + 6)  #設定xlabel ylabel字形大小
                barAx.tick_params(axis='y', labelsize=10)
                
                # 設定lable顏色
                for cellText in barAx.get_xticklabels():
                    txt = cellText.get_text()
                    for slideGroup in split_IRR_draw.slidingLableClrList:
                        if txt in slideGroup[0]:
                            plt.setp(cellText, bbox=dict(boxstyle='round, pad=0.15', edgecolor='none', alpha=1, facecolor=slideGroup[1]))
                            break
                # if len(self.df.columns) > 4:
                #     continue
                #     chooseDf = windowChooseDf[windowChooseDf.columns[dfCuttedIndex[splitIndex]:dfCuttedIndex[splitIndex + 1]]]
                #     table = axs[splitIndex].table(cellText=chooseDf.values, loc='bottom', cellLoc='center', rowLabels = [chooseDf.index[0], chooseDf.index[1]], bbox=[0, 0, 1, 0.2])
            handles, labels = barAxes[axIndexForLegned].get_legend_handles_labels()
            fig.legend(
                handles, labels, 
                loc='upper right', 
                bbox_to_anchor=(1, 1.05), 
                fancybox=True, shadow=False, 
                ncol=len(self.df.columns), 
                fontsize=split_IRR_draw.allFontSize)
            
            figTitle = self.company + (lambda x: ' reorder ' if x else ' ')(split_IRR_draw.reorder) + split_IRR_draw.trainOrTest + ' ' +  ' '.join(self.titleTechNames)
            if "window num" in self.df.columns:
                figTitle += ' highest IRR appearance'
            else:
                figTitle += ' IRR rank'
            
            fig.suptitle(figTitle, 
                         ha='left', 
                         x=0.025, 
                         y=(lambda tableObjSize: 1.07 if tableObjSize > 1 else 1.03)(len(self.tablePlotObjs)), 
                         fontsize=split_IRR_draw.allFontSize)
            # fig.subplots_adjust(hspace=1)
            
            figName = self.company + (lambda x: '_reorder' if x else '')(split_IRR_draw.reorder)
            if "window num" in self.df.columns:
                    figName += '_highest_IRR_appearence.png'
            elif split_IRR_draw.seperateTable:
                figName += '_all_IRR_no_table.png'
            else:
                figName += '_all_IRR.png'
            fig.savefig(figName, dpi=fig.dpi, bbox_inches='tight')
            plt.clf()
        
csv=[
        "windowRank_test_HI-all_GNQTS",
        "windowRank_test_HI-all_Tradition", 
        "windowRank_test_HI-RS_GNQTS", 
        "windowRank_test_HI-SR_Tradition", 
        "windowRank_test_RSI_GNQTS", 
        "windowRank_test_RSI_Tradition", 
        "windowRank_test_SMA_GNQTS", 
        "windowRank_test_SMA_Tradition", 
        "windowRank_train_all-average_GNQTS", 
        "windowRank_train_HI-all_GNQTS", 
        "windowRank_train_HI-all_Tradition", 
        "windowRank_train_RSI_GNQTS", 
        "windowRank_train_RSI_Tradition", 
        "windowRank_train_SMA_GNQTS", 
        "windowRank_train_SMA_Tradition", 
    ]

x = split_IRR_draw(IRRFileName='train_IRR_name_sorted_tech_highest_IRR_apearance',  # 若是compare類型，reorder要true，且csv內最多三個指標，超過放不下
                   splitIRRFile=True, 
                   drawBar=True, 
                   drawTable=False, 
                   seperateTable=True, 
                   reorder=False, 
                   setCompany='all')

# x = split_IRR_draw(IRRFileName=csv, 
#                 splitIRRFile=True, 
#                 drawBar=False, 
#                 drawTable=False, 
#                 seperateTable=True, 
#                 reorder=False, 
#                 setCompany='all')