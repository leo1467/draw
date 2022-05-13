from math import floor
from math import ceil
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.ticker as mtick

file_extension = '.csv'
root = os.getcwd()

class split_IRR_draw:
    #設定滑動視窗group
    slidingLableClrList = [
        [['YYY2YYY', 'YYY2YY', 'YYY2Y', 'YYY2YH', 'YY2YY', 'YY2YH', 'YY2Y', 'YH2YH', 'YH2Y', 'Y2Y'], 'gold'],
        [['YYY2H', 'YYY2Q', 'YYY2M', 'YY2H', 'YY2Q', 'YY2M', 'YH2H', 'YH2Q', 'YH2M', 'Y2H', 'Y2Q', 'Y2M'], 'limegreen'],
        [['H#', 'H2H', 'H2Q', 'H2M', 'Q#', 'Q2Q', 'Q2M', 'M#', 'M2M'], 'r'],
        [['20D20', '20D15', '20D10', '20D5', '15D15', '15D10', '15D5', '10D10', '10D5', '5D5'], 'grey'],
        [['4W4', '4W3', '4W2', '4W1', '3W3', '3W2', '3W1', '2W2', '2W1', '1W1'], 'darkgoldenrod'],
        [['5D4', '5D3', '5D2', '4D4', '4D3', '4D2', '3D3', '3D2', '2D2'], 'w']
        ]
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
        '5D5', '5D4', '5D3', '5D2', '1W1', 
        '4D4', '4D3', '4D2', 
        '3D3', '3D2', 
        '2D2']
    #設定bar屬性
    barColorSet = ['steelblue', 'darkorange', 'paleturquoise', 'wheat', 'lightcyan', 'lightyellow']
    BHColor = 'r'
    totalBarWidth = 0.80
    #儲存全部table資料
    tables = []
    #設定figure大小
    # plt.rcParams['figure.figsize'] = (21, 9)
    fig = plt.figure(
        figsize = [21, 9], 
        dpi = 300, 
        # figCnt+1, 
        # sharey = True, 
        # gridspec_kw={'height_ratios': grid}, 
        constrained_layout = True
        )
    allFontSize = 10
    
    def __init__(self, fileName, splitCSV, drawTable, drawBar, seperateTable, reorder):
        split_IRR_draw.seperateTable = seperateTable
        split_IRR_draw.reorder = reorder
        if split_IRR_draw.reorder:
            split_IRR_draw.reorderList.reverse()
        self.fileName = fileName + '.csv'
        self.dirName = 'split_' + self.fileName.split('.')[0]
        
        print(self.fileName)
        self.process_fileName_dir()
        
        if splitCSV:
            self.split_csv()
        else:
            os.chdir(self.dirName)
        
        self.allIRRFile = [i for i in glob.glob(f"*{file_extension}")]
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
        
        os.chdir(root)
        for dfIndex, eachDf in enumerate(self.tables):
            if dfIndex == 0:
                eachDf.to_csv(fileName.split('.')[0] + '_tables.csv')
            else:
                eachDf.to_csv(fileName.split('.')[0] + '_tables.csv', mode = 'a', header = None)
        
    def split_csv(self):
        self.IRRdf = pd.read_csv(self.fileName, index_col = False)
        if True in self.IRRdf.columns.str.contains('^Unnamed'):
            self.IRRdf = self.IRRdf.loc[: , ~self.IRRdf.columns.str.contains('^Unnamed')]
            self.IRRdf.to_csv(self.fileName, index = None)
        self.IRRdf = self.IRRdf.T.reset_index().T.reset_index(drop = True)
        os.chdir(self.dirName)
        index = [i for i in self.IRRdf.index if self.IRRdf.at[i, 0][0] == '=']
        index.append(len(self.IRRdf))
        for cellIndex in range(len(index) - 1):
            companyName = self.IRRdf.at[index[cellIndex], self.IRRdf.columns[0]].replace('=', '')
            self.IRRdf.at[index[cellIndex], self.IRRdf.columns[0]] = 'window'
            self.IRRdf[index[cellIndex]: index[cellIndex + 1]].to_csv(companyName + '_IRR.csv', header = None, index = None)
                
    def process_fileName_dir(self):
        fileNameList = self.fileName.split('.')[0].split('_')
        split_IRR_draw.allTitle = '_'.join(fileNameList[fileNameList.index('sorted') + 1: ])
        split_IRR_draw.trainOrTest = fileNameList[0]
        if not os.path.isdir(self.dirName):
            os.mkdir(self.dirName)
    
    class ProcessACompany:
        def __init__(self, FileIndex, file):
            self.table = []
            self.tableColumns = list()
            self.cellData = dict()
            self.IRRData = dict()
            self.tableObjs = list()
            print(file)
            if file.split('.')[1] == 'csv':
                self.process_df(file)
                self.find_techNames()
                self.process_IRRFile(FileIndex, file)
                
        def process_df(self, file):
            #df前處理
            self.company = file.split('_')[0]
            self.df = pd.read_csv(file, index_col = 0)
            # df.rename(columns = {df.columns[0]: 'window'}, inplace = True)
            for colIndex in self.df.columns:
                for rowIndex in self.df.index:
                    self.df.at[rowIndex, colIndex] *= 100
        
        def find_techNames(self):
            self.techNames = [name.split(' ')[0] for name in self.df.columns if 'B&H' not in name]
            if len(self.techNames[0].split('_')) > 1:
                self.techNames = [y for x, y in enumerate(self.techNames[:-5]) if x % 2 == 0]
            else:
                self.techNames = [y for x, y in enumerate(self.techNames) if x % 2 == 0]
            self.techNum = len(self.techNames)
            self.titleTechNames = [i + '"' for i in ['"' + j for j in self.techNames]]
            self.mixedTech = (lambda x:True if len(x.split('_')) > 1 else False)(self.techNames[0])
            
        def process_IRRFile(self, FileIndex, file):
            # table資料
            for colIndex, col in enumerate(self.df.columns):
                self.df.sort_values(by = col, ascending = False, inplace = True)
                self.IRRData.update({col: pd.Series({self.df.index[i] : x for i, x in enumerate(self.df[col]) if self.df.index[i] != 'B&H'})})
            
            if split_IRR_draw.reorder:
                self.df = self.df.reindex(split_IRR_draw.reorderList)
            else:
                self.df.sort_values(by = self.df.columns[0], ascending = False, inplace = True)
            
            for i in range(self.techNum):
                self.add_info(self.techNames[i], '', i * 2, i * 2 + 1, False)
            
            if self.techNum > 1:
                for techIndex, row in zip(range(1, self.techNum), range(2, self.techNum * 2, 2)):
                    self.add_info(self.techNames[0], self.techNames[techIndex], 0, row, True)
            
            if self.mixedTech and len(self.df.columns) > self.techNum * 2:
                windowChoose = list()
                for techIndex, col in zip(range(1, self.techNum), range(self.techNum * 2, len(self.df.columns), 2)):
                    windowChoose.append(self.df[self.df.columns[col]] / self.df['window num'] * 100)
                for i in range(len(windowChoose)):
                    windowChoose[i] = ['%.2f' % elem + '%' for elem in windowChoose[i]]
                self.windowChooseDf = pd.DataFrame(windowChoose, columns = [self.df.index], index = [self.techNames[1 : ]])
                self.df = self.df.drop(columns = [col for col in self.df.columns[self.techNum * 2: ]])
            
            for cellIndex in self.cellData:
                try:
                    cellData = str(round(float(self.cellData[cellIndex]), 2)) + '%'
                except ValueError:
                    continue
                self.cellData[cellIndex] = cellData
            self.tableDf = pd.DataFrame([self.cellData])
            split_IRR_draw.tables.append(self.tableDf)
                
        def add_info(self, comp1, comp2, col1, col2, techCompare):
            if not techCompare:
                data = {
                    f'{comp1} algo best/worst w': self.IRRData[self.df.columns[col1]].index[0] + '/' + self.IRRData[self.df.columns[col1]].index[-1],
                    f'{comp1} trad best/worst w': self.IRRData[self.df.columns[col2]].index[0] + '/' + self.IRRData[self.df.columns[col2]].index[-1],
                    f'{comp1} highest algo IRR': max(self.IRRData[self.df.columns[col1]]),
                    # [f'{comp1} highest IRR diff algo/trad', max(self.IRRData[self.df.columns[col1]]) - max(self.IRRData[self.df.columns[col2]])], 
                    # [f'{comp1} algo win rate', len([i for i, j in zip(self.IRRData[self.df.columns[col1]], self.IRRData[self.df.columns[col2]]) if i > j]) / len(self.IRRData[self.df.columns[col1]]) * 100],
                    f'{comp1} algo avg IRR': np.average(self.IRRData[self.df.columns[col1]]), 
                    f'{comp1} trad avg IRR': np.average(self.IRRData[self.df.columns[col2]]), 
                    # [f'{comp1} highest IRR diff algo/B&H', max(self.IRRData[self.df.columns[col1]]) - self.df.at['B&H', self.df.columns[col1]]], 
                    # [f'{comp1} highest IRR diff trad/B&H', max(self.IRRData[self.df.columns[col2]]) - self.df.at['B&H', self.df.columns[col2]]], 
                    f'{comp1} algo "Y" avg IRR': np.average([IRR for w, IRR in zip(self.df.index, self.df[self.df.columns[col1]]) if w[0] == 'Y']), 
                    f'{comp1} algo "H" "Q" "M" avg IRR': np.average([IRR for w, IRR in zip(self.df.index, self.df[self.df.columns[col1]]) if w[0] == 'H' or w[0] == 'Q' or w[0] == 'M']), 
                    f'{comp1} algo "D" "W" avg IRR': np.average([IRR for w, IRR in zip(self.df.index, self.df[self.df.columns[col1]]) if 'W' in w or 'D' in w]), 
                }
                self.dataColLen = len(data)
            else:
                data = {
                    # [f'highest IRR diff of algo {comp1}/{comp2}', max(self.IRRData[self.df.columns[col1]]) - max(self.IRRData[self.df.columns[col2]])],
                    # [f'highest IRR diff of trad {comp1}/{comp2}', max(self.IRRData[self.df.columns[col1+1]]) - max(self.IRRData[self.df.columns[col2 + 1]])],
                    f'algo win rate {comp1}/{comp2}': len([i for i, j in zip(self.IRRData[self.df.columns[col1]], self.IRRData[self.df.columns[col2]]) if i > j]) / len(self.IRRData[self.df.columns[col1]]) * 100, 
                    f'trad win rate {comp1}/{comp2}': len([i for i, j in zip(self.IRRData[self.df.columns[col1 + 1]], self.IRRData[self.df.columns[col2 + 1]]) if i > j]) / len(self.IRRData[self.df.columns[col1]]) * 100,
                }
                self.frontLen = len(data)
                backData = {
                    f'{comp1} algo "Y" win': str(len([w for w in self.df.index if self.compare(w, col1, col2) and w[0] == 'Y'])) + '/' + str(len([w for w in self.df.index if w[0] == 'Y'])),
                    f'{comp1} algo "H" "Q" "M" win': str(len([w for w in self.df.index if self.compare(w, col1, col2) and (w[0] == 'H' or w[0] == 'Q' or w[0] == 'M')])) + '/' + str(len([w for w in self.df.index if w[0] == 'H' or w[0] == 'Q' or w[0] == 'M'])),
                    f'{comp1} algo "D" "W" win': str(len([w for w in self.df.index if self.compare(w, col1, col2) and ('W' in w or 'D' in w)])) + '/' + str(len([w for w in self.df.index if ('W' in w or 'D' in w)])),
                    
                    f'{comp1} trad "Y" win': str(len([w for w in self.df.index if self.compare(w, col1+1, col2+1) and w[0] == 'Y'])) + '/' + str(len([w for w in self.df.index if w[0] == 'Y'])),
                    f'{comp1} trad "H" "Q" "M" win': str(len([w for w in self.df.index if self.compare(w, col1+1, col2+1) and (w[0] == 'H' or w[0] == 'Q' or w[0] == 'M')])) + '/' + str(len([w for w in self.df.index if w[0] == 'H' or w[0] == 'Q' or w[0] == 'M'])),
                    f'{comp1} trad "D" "W" win': str(len([w for w in self.df.index if self.compare(w, col1+1, col2+1) and ('W' in w or 'D' in w)])) + '/' + str(len([w for w in self.df.index if ('W' in w or 'D' in w)])),
                }
                self.backLen = len(backData)
                for elem in backData.items():
                    data.update({elem})
                self.finalCompareLen = len(data)
            for elem in data.items():
                self.cellData.update({elem})
        
        def compare(self, window, col1, col2):
            return self.df.at[window, self.df.columns[col1]] > self.df.at[window, self.df.columns[col2]]
        
        def start_draw_tables(self, fig, gs):
            #設定top table
            self.draw_tables(fig, gs)
            if split_IRR_draw.seperateTable:
                fig.suptitle(self.company + " " + split_IRR_draw.trainOrTest + ' ' +  ' '.join(self.titleTechNames) + ' compare table', 
                    fontsize = split_IRR_draw.allFontSize + 5)
                fig.savefig(self.company + '_table'  + '.png', dpi = fig.dpi, bbox_inches = 'tight')
                plt.clf()
                self.tableObjs.clear()
        
        def draw_bar(self, fig, gs, gridNum):
            #設定每個bar的顏色及bar的最終寬度
            colorDict = dict(zip(self.df.columns, split_IRR_draw.barColorSet))
            if len(self.df.columns) < 3:
                split_IRR_draw.totalBarWidth = 0.5
            
            #將過長的df切開
            figCnt = (lambda x : 2 if x < 5 else 3)(len(self.df.columns))
            dfCuttedIndex = list()
            for i in range(figCnt):
                dfCuttedIndex.append(floor(len(self.df) / figCnt) * i)
            dfCuttedIndex.append(len(self.df))
            
            #找出plot bar要佔用哪些grid
            startGrid = (lambda tableObjSize : 0 if tableObjSize == 0 else (tableObjSize - 1) * 2 - 1)(len(self.tableObjs))
            figJump = ceil((gridNum - startGrid) / figCnt)
            figGrid = [startGrid]
            for i in range(figCnt):
                figGrid.append(figGrid[i] + figJump)
            
            #plot bar
            barAxes = list()
            for splitIndex in range(figCnt):
                if splitIndex == 0:
                    barAx = fig.add_subplot(gs[figGrid[splitIndex]: figGrid[splitIndex + 1], :])
                else:
                    barAx = fig.add_subplot(gs[figGrid[splitIndex]: figGrid[splitIndex + 1], :], sharey = barAxes[0])
                barAxes.append(barAx)
                subDf = self.df.iloc[dfCuttedIndex[splitIndex]: dfCuttedIndex[splitIndex + 1]]
                plot = subDf.plot.bar(
                    ax = barAx, 
                    width = split_IRR_draw.totalBarWidth, 
                    rot = 0, 
                    color = colorDict, 
                    edgecolor = 'black', 
                    linewidth = 0.2, 
                    legend = None
                    )
                
                #找出B&H位置，將B&H的bar變成紅色
                BHIndex = [i for i, x in enumerate(subDf.index) if x == 'B&H']
                if not len(BHIndex):
                    axIndexForLegned = splitIndex
                if len(BHIndex):
                    BHIndex = BHIndex[0]
                    for barIndex, barContainer in enumerate(plot.containers):
                        if barIndex == 0:
                            singleBarWidth = barContainer[BHIndex].get_width()
                            barX = ((barContainer[BHIndex].get_x() + (barContainer[BHIndex].get_x() + (len(subDf.columns) * singleBarWidth))) - singleBarWidth) / 2
                        barContainer[BHIndex].set_color(split_IRR_draw.BHColor)
                        barContainer[BHIndex].set_x(barX)
                        barContainer[BHIndex].set_edgecolor('black')
                
                #如果是train，變更每個滑動視窗的B&H顏色
                if split_IRR_draw.trainOrTest == 'train':
                    for singleBar in plot.containers[-1]:
                        singleBar.set_color(split_IRR_draw.BHColor)
                        singleBar.set_edgecolor('black')
                
                #設定其他屬性
                barAx.grid(axis = 'y')
                barAx.yaxis.set_major_formatter(mtick.PercentFormatter())  #把座標變成%
                barAx.locator_params(axis = 'y', nbins = 10)
                barAx.set_xticklabels(subDf.index, rotation = 45)
                barAx.set(xlabel = '', ylabel = '')
                barAx.tick_params(axis = 'both', labelsize = split_IRR_draw.allFontSize)  #設定xlabel ylabel字形大小
                
                #設定lable顏色
                for cellIndex in barAx.get_xticklabels():
                    txt = cellIndex.get_text()
                    for slideGroup in split_IRR_draw.slidingLableClrList:
                        if txt in slideGroup[0]:
                            plt.setp(cellIndex, bbox = dict(boxstyle = 'round', edgecolor = 'none', alpha = 1, facecolor = slideGroup[1]))
                            break
                # if len(self.df.columns) > 4:
                #     continue
                #     chooseDf = windowChooseDf[windowChooseDf.columns[dfCuttedIndex[splitIndex]: dfCuttedIndex[splitIndex + 1]]]
                #     table = axs[splitIndex].table(cellText = chooseDf.values, loc = 'bottom', cellLoc = 'center', rowLabels = [chooseDf.index[0], chooseDf.index[1]], bbox = [0, 0, 1, 0.2])
            handles, labels = barAxes[axIndexForLegned].get_legend_handles_labels()
            fig.legend(
                handles, labels, 
                loc = 'upper center', 
                bbox_to_anchor = (0.5, 0), 
                fancybox = True, shadow = False, 
                ncol = len(self.df.columns), 
                fontsize = split_IRR_draw.allFontSize)
            figTitle = self.company + (lambda x : ' reorder ' if x else ' ')(split_IRR_draw.reorder) + split_IRR_draw.trainOrTest + ' ' +  ' '.join(self.titleTechNames) + ' IRR rank'
            fig.suptitle(figTitle, 
                        y = (lambda tableObjSize:1.07 if tableObjSize > 1 else 1.03)(len(self.tableObjs)), 
                        fontsize = split_IRR_draw.allFontSize + 5)
            # fig.subplots_adjust(hspace=1)
            figName = self.company + (lambda x : '_reorder' if x else '')(split_IRR_draw.reorder)
            if split_IRR_draw.seperateTable:
                figName += '_all_IRR_no_table'  + '.png'
            else:
                figName += '_all_IRR'  + '.png'
            fig.savefig(figName, dpi = fig.dpi, bbox_inches = 'tight')
            plt.clf()
        
        def set_grid(self, fig):
            gridNum = (lambda x : 24 if x == 1 else 25)(split_IRR_draw.seperateTable)
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
                    wspace = 0, 
                    hspace = hspace, 
                    top = 1, 
                    bottom = 0, 
                    # left = 0.17, 
                    # right = 0.845
                    )
            return gs, gridNum
                
        def draw_tables(self, fig, gs):
            for i, j in zip(range(0, self.dataColLen * self.techNum, self.dataColLen), range(self.techNum)):
                tableAx = fig.add_subplot(gs[j, : ])
                tableAx.axis('off')
                self.draw_table(tableAx, i, i + self.dataColLen)
            
            #比較table中不同指標誰比較大，並著色
            if len(self.techNames) > 1:
                for colNum in range(self.dataColLen):
                    self.find_big_cell(colNum)

            #設定比較table
            if len(self.techNames) > 1:
                for i in range(self.techNum - 1):
                    tableAx = fig.add_subplot(gs[self.techNum + i, : ])
                    tableAx.axis('off')
                    startCol = self.dataColLen * self.techNum + i * self.finalCompareLen
                    endCol = startCol + self.finalCompareLen
                    self.draw_table(tableAx, startCol, endCol)
        
        def draw_table(self, tableAx, startCol, endCol):
            tmpTableDf = self.tableDf.iloc[ : , startCol : endCol]
            topTable = tableAx.table(
                colLabels = tmpTableDf.columns, 
                cellText = tmpTableDf.values, 
                # loc = 'best', 
                cellLoc = 'center', 
                colColours = ['silver'] * (len(tmpTableDf.columns)), 
                bbox = [0, 1, 1, 2]
                )
            self.tableObjs.append(topTable)
            # for colIndex in range(len(tableDf.columns)):  #設定cell text顏色
            #     topTable[0, colIndex].get_text().set_color('white')
            # topTable.auto_set_column_width(col = list(range(len(self.tableDf.columns))))
            topTable.auto_set_font_size(False)
            topTable.set_fontsize('large')  # Valid font size are xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller, None
            if len(self.tableObjs) > self.techNum:
                # topTable.auto_set_column_width(col = list(range(len(self.tableDf.columns))))
                for i in range(6):
                    if i < 3:
                        topTable[0, self.frontLen + i].set_color('lightsteelblue')
                    else:
                        topTable[0, self.frontLen + i].set_color('bisque')
                    topTable[0, self.frontLen + i].set_edgecolor('black')
        
        def find_big_cell(self, colNum):
            cellCompare = list()
            for eachTable in self.tableObjs:
                cellCompare.append(eachTable[1, colNum])
            compare = -1
            for cell in cellCompare:
                try:
                    cellData = float(cell.get_text()._text[:-1])
                except ValueError:
                    continue
                if cellData > compare:
                    compare = cellData
            bigCell = list()
            for cell in cellCompare:
                try:
                    cellData = float(cell.get_text()._text[:-1])
                except ValueError:
                    continue
                if cellData == compare:
                    bigCell.append(cell)
            for cell in bigCell:
                cell.set_color('lime')
                cell.set_edgecolor('black')

x = split_IRR_draw('train_IRR_IRR_sorted_RSI_2', splitCSV = True, drawBar = True, drawTable = True, seperateTable = True, reorder = False)