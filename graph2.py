from math import floor
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.ticker as mtick

root = os.getcwd()
file_extension = '.csv'
splitTargetFolder = ['/testBestHold']

class split_testIRR_draw:
    #設定滑動視窗group
    slidingLableClrList = [[['YYY2YYY', 'YYY2YY', 'YYY2Y', 'YYY2YH', 'YY2YY', 'YY2YH', 'YY2Y', 'YH2YH', 'YH2Y', 'Y2Y'], 'gold'],
                           [['YYY2H', 'YYY2Q', 'YYY2M', 'YY2H', 'YY2Q', 'YY2M', 'YH2H', 'YH2Q', 'YH2M', 'Y2H', 'Y2Q', 'Y2M'], 'limegreen'],
                           [['H2H', 'H#', 'H2Q', 'Q2Q', 'Q#', 'H2M', 'Q2M', 'M2M', 'M#'], 'r'],
                           [['20D20', '20D15', '20D10', '20D5', '15D15', '15D10', '15D5', '10D10', '10D5', '5D5'], 'grey'],
                           [['4W4', '4W3', '4W2', '4W1', '3W3', '3W2', '3W1', '2W2', '2W1', '1W1'], 'darkgoldenrod']]
    #設定bar屬性
    barColorSet = ['steelblue', 'darkorange', 'paleturquoise', 'wheat', 'lightcyan', 'lightyellow']
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
    
    def __init__(self, fileName, split, draw):
        self.process_fileName_dir(fileName)
        if split:
            self.split()
        else:
            os.chdir(self.dirName)
        self.allIRRFile = [i for i in glob.glob(f"*{file_extension}")]
        for FileIndex, file in enumerate(self.allIRRFile):
            processACompany = self.ProcessACompany(FileIndex, file)
            if draw:
                processACompany.draw()
            # break
        plt.close(split_testIRR_draw.fig)
        os.chdir(root)
        for dfIndex, eachDf in enumerate(self.tables):
            if dfIndex == 0:
                eachDf.to_csv(fileName.split('.')[0] + '_tables.csv')
            else:
                eachDf.to_csv(fileName.split('.')[0] + '_tables.csv', mode = 'a', header = None)
            
    def process_fileName_dir(self, fileName):
        self.fileName = fileName
        print(self.fileName)
        fileNameList = fileName.split('.')[0].split('_')
        split_testIRR_draw.allTitle = '_'.join(fileNameList[fileNameList.index('sorted') + 1: ])
        split_testIRR_draw.trainOrTest = fileNameList[0]
        self.dirName = 'split_' + fileName.split('.')[0]
        if not os.path.isdir(self.dirName):
            os.mkdir(self.dirName)
        
    def split(self):
        self.IRRdf = pd.read_csv(self.fileName)
        if True in self.IRRdf.columns.str.contains('^Unnamed'):
            self.IRRdf = self.IRRdf.loc[: , ~self.IRRdf.columns.str.contains('^Unnamed')]
            self.IRRdf.to_csv(self.fileName, index = None)
        self.IRRdf = self.IRRdf.T.reset_index().T.reset_index(drop = True)
        os.chdir(self.dirName)
        index = []
        for row, content in enumerate(self.IRRdf[0]):
            if content[0] == '=':
                index.append(row)
        index.append(len(self.IRRdf))
        for cellIndex in range(len(index) - 1):
            companyName = self.IRRdf.at[index[cellIndex], self.IRRdf.columns[0]].replace('=', '')
            self.IRRdf.at[index[cellIndex], self.IRRdf.columns[0]] = 'window'
            self.IRRdf[index[cellIndex]: index[cellIndex + 1]].to_csv(companyName + '_IRR.csv', header = None, index = None)
    
    class ProcessACompany:
        def __init__(self, FileIndex, file):
            self.table = []
            self.tableColumns = list()
            self.cellData = list()
            self.IRRData = dict()
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
            self.techNames = list()
            for colIndex in range(len(self.df.columns)):
                self.techNames.append(self.df.columns[colIndex].split(' ')[0])
            if len(self.techNames[0].split('_')) > 1:
                self.techNames = [y for x, y in enumerate(self.techNames[:-5]) if x % 2 == 0]
            else:
                self.techNames = [y for x, y in enumerate(self.techNames) if x % 2 == 0]
            self.techNum = len(self.techNames)
            
        def process_IRRFile(self, FileIndex, file):
            # table資料
            for colIndex, col in enumerate(self.df.columns):
                self.IRRData.update({col: np.array([x for i, x in enumerate(self.df[col]) if self.df.index[i] != 'B&H'])})
            self.add_col(self.techNames[0], '', 6)
            self.add_info(0, 1, 6)
            if len(self.df.columns) > 2:
                self.add_col(self.techNames[1], '', 6)
                self.add_info(2, 3, 6)
                if len(self.df.columns) > 4:
                    self.add_col(self.techNames[2], '', 6)
                    self.add_info(4, 5, 6)
                    windowChoose = list()
                    windowChoose.append(self.df[self.df.columns[6]] / self.df['window num'] * 100)
                    windowChoose.append(self.df[self.df.columns[8]] / self.df['window num'] * 100)
                    for i in range(len(windowChoose)):
                        windowChoose[i] = ['%.2f' % elem + '%' for elem in windowChoose[i]]
                    self.windowChooseDf = pd.DataFrame(windowChoose, columns = [self.df.index], index = [self.df.columns[6], self.df.columns[8]])
                    self.df = self.df.drop(columns = [col for col in self.df.columns[-5: ]])
            if len(self.df.columns) > 2:
                self.add_col(self.techNames[0], self.techNames[1], 4)
                self.add_info(0, 2, 4)
                if len(self.df.columns) > 4:
                    self.add_col(self.techNames[0], self.techNames[2], 4)
                    self.add_info(0, 4, 4)
            self.cellData = ['%.2f' % elem for elem in self.cellData]
            self.cellData = np.array([[elem + '%'] for elem in self.cellData])
            self.tableDf = pd.DataFrame(self.cellData.reshape(1, len(self.tableColumns)), columns = self.tableColumns)
            self.tableDf.rename(index = { 0: self.company }, inplace = True)
            split_testIRR_draw.tables.append(self.tableDf)
                
        def add_col(self, comp1, comp2, colNum):
            if colNum == 6:
                newColumns = [
                    f'{comp1} highest algo IRR',
                    f'{comp1} highest IRR diff algo/trad', 
                    f'{comp1} algo win rate', 
                    f'{comp1} algo avg IRR', 
                    f'{comp1} trad avg IRR', 
                    f'{comp1} highest IRR diff algo/B&H', 
                    f'{comp1} highest IRR diff trad/B&H'
                    ]
            else:
                newColumns = [
                    f'highest IRR diff of algo {comp1}/{comp2}', 
                    f'highest IRR diff of trad {comp1}/{comp2}', 
                    f'algo win rate {comp1}/{comp2}', 
                    f'trad win rate {comp1}/{comp2}', 
                    ]
            for t in newColumns:
                self.tableColumns.append(t)
            
        def add_info(self, col1, col2, dataNum):
            if dataNum == 6:
                data = [
                    max(self.IRRData[self.df.columns[col1]]),
                    max(self.IRRData[self.df.columns[col1]]) - max(self.IRRData[self.df.columns[col2]]),
                    len([i for i, j in zip(self.IRRData[self.df.columns[col1]], self.IRRData[self.df.columns[col2]]) if i > j]) / len(self.IRRData[self.df.columns[col1]]) * 100,
                    np.average(self.IRRData[self.df.columns[col1]]),
                    np.average(self.IRRData[self.df.columns[col2]]),
                    max(self.IRRData[self.df.columns[col1]]) - self.df.at['B&H', self.df.columns[col1]],
                    max(self.IRRData[self.df.columns[col2]]) - self.df.at['B&H', self.df.columns[col2]]
                    ]
            else:
                data = [
                    max(self.IRRData[self.df.columns[col1]]) - max(self.IRRData[self.df.columns[col2]]),
                    max(self.IRRData[self.df.columns[col1+1]]) - max(self.IRRData[self.df.columns[col2 + 1]]),
                    len([i for i, j in zip(self.IRRData[self.df.columns[col1]], self.IRRData[self.df.columns[col2]]) if i > j]) / len(self.IRRData[self.df.columns[col1]]) * 100,
                    len([i for i, j in zip(self.IRRData[self.df.columns[col1 + 1]], self.IRRData[self.df.columns[col2 + 1]]) if i > j]) / len(self.IRRData[self.df.columns[col1]]) * 100,
                    ]
            for d in data:
                self.cellData.append(d)
        
        def draw(self):
            #設定每個bar的顏色及bar的最終寬度
            colorDict = dict(zip(self.df.columns, split_testIRR_draw.barColorSet))
            if len(self.df.columns) < 3:
                split_testIRR_draw.totalBarWidth = 0.5
            
            #將過長的df切開
            figCnt = (lambda x : 2 if x < 5 else 3)(len(self.df.columns))
            dfCuttedIndex = list()
            for i in range(figCnt):
                dfCuttedIndex.append(floor(len(self.df) / figCnt) * i)
            dfCuttedIndex.append(len(self.df))
            
            #宣告fig
            fig = split_testIRR_draw.fig
            
            #設定grid資訊
            gridNum = 25
            if self.techNum == 1:
                hspace = -0.9
            elif self.techNum == 2:
                hspace = -0.6
            elif self.techNum == 3:
                hspace = -0.25
            gs = fig.add_gridspec(
                gridNum, 1, 
                wspace = 0, 
                hspace = hspace, 
                top = 1, 
                bottom = 0, 
                # left = 0.17, 
                # right = 0.845
                )
            
            #設定top table跟bottom table
            dataColLen = 6
            for i, j in zip(range(0, dataColLen * self.techNum, dataColLen), range(self.techNum)):
                tableAx = fig.add_subplot(gs[j, : ])
                tableAx.axis('off')
                self.draw_table(tableAx, i, i + dataColLen)
            
            if len(self.techNames) > 1:
                finalCompareLen = 4
                for i in range(self.techNum - 1):
                    tableAx = fig.add_subplot(gs[self.techNum + i, : ])
                    tableAx.axis('off')
                    startCol = dataColLen * self.techNum + i * finalCompareLen
                    endCol = startCol + finalCompareLen
                    self.draw_table(tableAx, startCol, endCol)
                
            #找出plot bar要佔用哪些grid
            startGrid = self.techNum * 2 - 1
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
                    width = split_testIRR_draw.totalBarWidth, 
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
                        barContainer[BHIndex].set_color('r')
                        barContainer[BHIndex].set_x(barX)
                        barContainer[BHIndex].set_edgecolor('black')
                
                #設定其他屬性
                barAx.grid(axis = 'y')
                barAx.yaxis.set_major_formatter(mtick.PercentFormatter())  #把座標變成%
                barAx.locator_params(axis = 'y', nbins = 10)
                barAx.set_xticklabels(subDf.index, rotation = 45)
                barAx.set(xlabel = '', ylabel = '')
                barAx.tick_params(axis = 'both', labelsize = split_testIRR_draw.allFontSize)  #設定xlabel ylabel字形大小
                
                #設定lable顏色
                for cellIndex in barAx.get_xticklabels():
                    txt = cellIndex.get_text()
                    for slideGroup in split_testIRR_draw.slidingLableClrList:
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
                fontsize = split_testIRR_draw.allFontSize)
            titleTechNames = [i + '"' for i in ['"' + j for j in self.techNames]]
            fig.suptitle(self.company + " " + split_testIRR_draw.trainOrTest + ' ' +  ' '.join(titleTechNames) + ' IRR rank', 
                         y = 1.07, 
                         fontsize = split_testIRR_draw.allFontSize + 5)
            # fig.subplots_adjust(hspace=1)
            plt.savefig(self.company + '_all_IRR'  + '.png', dpi = fig.dpi, bbox_inches = 'tight')
            plt.clf()
            
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
            # for colIndex in range(len(tableDf.columns)):  #設定cell text顏色
            #     topTable[0, colIndex].get_text().set_color('white')
            # topTable.auto_set_column_width(col = list(range(len(self.tableDf.columns))))
            topTable.auto_set_font_size(False)
            topTable.set_fontsize('large')  # Valid font size are xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller, None


if __name__ == '__main__':
    # draw_hold()
    x = split_testIRR_draw('train_IRR_IRR_sorted_SMA_2' + '.csv', 1, 0)

# def split_testIRR_draw(fileName, split, draw):
#     print(fileName)
#     fileNameList = fileName.split('.')[0].split('_')
#     techName = '_'.join(fileNameList[fileNameList.index('sorted') + 1: ])
#     dirName = 'split_' + fileName.split('.')[0]
#     if not os.path.isdir(dirName):
#             os.mkdir(dirName)
#     if split:
#         oriIRRFile = pd.read_csv(fileName)
#         if True in oriIRRFile.columns.str.contains('^Unnamed'):
#             oriIRRFile = oriIRRFile.loc[: , ~oriIRRFile.columns.str.contains('^Unnamed')]
#             oriIRRFile.to_csv(fileName, index = None)
#         oriIRRFile = oriIRRFile.T.reset_index().T.reset_index(drop = True)
#         os.chdir(dirName)
#         index = []
#         for row, content in enumerate(oriIRRFile[0]):
#             if content[0] == '=':
#                 index.append(row)
#         index.append(len(oriIRRFile))
#         for cellIndex in range(len(index) - 1):
#             companyName = oriIRRFile.at[index[cellIndex], oriIRRFile.columns[0]].replace('=', '')
#             oriIRRFile.at[index[cellIndex], oriIRRFile.columns[0]] = 'window'
#             oriIRRFile[index[cellIndex]: index[cellIndex + 1]].to_csv(companyName + '_IRR.csv', header = None, index = None)
#     else:
#         os.chdir(dirName)
#     allIRRFile = [i for i in glob.glob(f"*{file_extension}")]
#     #設定滑動視窗group
#     slidingLableClrList = [[['YYY2YYY', 'YYY2YY', 'YYY2Y', 'YYY2YH', 'YY2YY', 'YY2YH', 'YY2Y', 'YH2YH', 'YH2Y', 'Y2Y'], 'gold'],
#                            [['YYY2H', 'YYY2Q', 'YYY2M', 'YY2H', 'YY2Q', 'YY2M', 'YH2H', 'YH2Q', 'YH2M', 'Y2H', 'Y2Q', 'Y2M'], 'limegreen'],
#                            [['H2H', 'H#', 'H2Q', 'Q2Q', 'Q#', 'H2M', 'Q2M', 'M2M', 'M#'], 'r'],
#                            [['20D20', '20D15', '20D10', '20D5', '15D15', '15D10', '15D5', '10D10', '10D5', '5D5'], 'grey'],
#                            [['4W4', '4W3', '4W2', '4W1', '3W3', '3W2', '3W1', '2W2', '2W1', '1W1'], 'darkgoldenrod']]
#     #設定bar屬性
#     barColorSet = ['steelblue', 'darkorange', 'paleturquoise', 'wheat', 'lightcyan', 'lightyellow']
#     totalBarWidth = 0.80
#     #儲存全部table資料
#     tables = []
#     #設定figure大小
#     plt.rcParams['figure.figsize'] = (21, 9)
#     allFontSize = 11
#     for Fileindex, file in enumerate(allIRRFile):
#         print(file)
#         if file.split('.')[1] == 'csv':
#             #df前處理
#             company = file.split('_')[0]
#             df = pd.read_csv(file, index_col = 0)
#             # df.rename(columns = {df.columns[0]: 'window'}, inplace = True)
#             for colIndex in df.columns:
#                 for rowIndex in df.index:
#                     df.at[rowIndex, colIndex] *= 100
#             # table資料
#             tableColumns = ['highest IRR diff\n algo/trad', 'algo win rate', 'algo avg IRR', 'trad avg IRR', 'highest IRR diff\n algo/B&H', 'highest IRR diff\n trad/B&H']
#             cellData = list()
#             IRRData = dict()
#             for colIndex, col in enumerate(df.columns):
#                 IRRData.update({col: np.array([x for i, x in enumerate(df[col]) if df.index[i] != 'B&H'])})
#             cellData.append(max(IRRData[df.columns[0]]) - max(IRRData[df.columns[1]]))
#             cellData.append(len([i for i, j in zip(IRRData[df.columns[0]], IRRData[df.columns[1]]) if i > j]) / len(IRRData[df.columns[0]]) * 100)
#             cellData.append(np.average(IRRData[df.columns[0]]))
#             cellData.append(np.average(IRRData[df.columns[1]]))
#             cellData.append(max(IRRData[df.columns[0]]) - df.at['B&H', df.columns[0]])
#             cellData.append(max(IRRData[df.columns[1]]) - df.at['B&H', df.columns[0]])
#             if len(df.columns) > 2:
#                 secndTechName = df.columns[2].split(" ")[0]
#                 tableColumns.append(f'highest IRR diff of algo\n{techName}/{secndTechName}')
#                 cellData.append(max(IRRData[df.columns[0]]) - max(IRRData[df.columns[2]]))
#                 tableColumns.append(f'highest IRR diff of trad\n{techName}/{secndTechName}')
#                 cellData.append(max(IRRData[df.columns[1]]) - max(IRRData[df.columns[3]]))
#                 tableColumns.append(f'algo win rate\n{techName}/{secndTechName}')
#                 cellData.append(len([i for i, j in zip(IRRData[df.columns[0]], IRRData[df.columns[2]]) if i > j]) / len(IRRData[df.columns[0]]) * 100)
#                 tableColumns.append(f'trad win rate\n{techName}/{secndTechName}')
#                 cellData.append(len([i for i, j in zip(IRRData[df.columns[1]], IRRData[df.columns[3]]) if i > j]) / len(IRRData[df.columns[0]]) * 100)
                
#                 tableColumns.append(f'{secndTechName} highest IRR diff\n algo/trad')
#                 cellData.append(max(IRRData[df.columns[2]]) - max(IRRData[df.columns[3]]))
#                 tableColumns.append(f'{secndTechName} algo win rate')
#                 cellData.append(len([i for i, j in zip(IRRData[df.columns[2]], IRRData[df.columns[3]]) if i > j]) / len(IRRData[df.columns[0]]) * 100)
#                 tableColumns.append(f'{secndTechName} highest IRR diff\n algo/B&H')
#                 cellData.append(max(IRRData[df.columns[2]]) - df.at['B&H', df.columns[0]])
#             if len(df.columns) > 4:
#                 thirdTechName = df.columns[4].split(" ")[0]
#                 tableColumns.append(f'highest IRR diff of algo\n{techName}/{thirdTechName}')
#                 cellData.append(max(IRRData[df.columns[0]]) - max(IRRData[df.columns[4]]))
#                 tableColumns.append(f'highest IRR diff of trad\n{techName}/{thirdTechName}')
#                 cellData.append(max(IRRData[df.columns[1]]) - max(IRRData[df.columns[5]]))
#                 tableColumns.append(f'algo win rate\n{techName}/{thirdTechName}')
#                 cellData.append(len([i for i, j in zip(IRRData[df.columns[0]], IRRData[df.columns[4]]) if i > j]) / len(IRRData[df.columns[0]]) * 100)
#                 tableColumns.append(f'trad win rate\n{techName}/{thirdTechName}')
#                 cellData.append(len([i for i, j in zip(IRRData[df.columns[1]], IRRData[df.columns[5]]) if i > j]) / len(IRRData[df.columns[0]]) * 100)
#                 windowChoose = list()
#                 windowChoose.append(df[df.columns[6]] / df['window num'] * 100)
#                 windowChoose.append(df[df.columns[8]] / df['window num'] * 100)
#                 for i in range(len(windowChoose)):
#                     windowChoose[i] = ['%.2f' % elem + '%' for elem in windowChoose[i]]
#                 windowChooseDf = pd.DataFrame(windowChoose, columns = [df.index], index = [df.columns[6], df.columns[8]])
#                 df = df.drop(columns = [col for col in df.columns[-5: ]])
#             cellData = ['%.2f' % elem for elem in cellData]
#             cellData = np.array([[elem + '%'] for elem in cellData])
#             tableDf = pd.DataFrame(cellData.reshape(1, len(tableColumns)), columns = tableColumns)
#             tableDf.rename(index = { 0: company }, inplace = True)
#             tables.append(tableDf)
#             if draw:
#                 #設定每個bar的顏色及bar的最終寬度
#                 colorDict = dict(zip(df.columns, barColorSet))
#                 if len(df.columns) < 3:
#                     totalBarWidth = 0.5
#                 #將過長的df切開
#                 if len(df.columns) < 5:
#                     figCnt = 2
#                 else:
#                     figCnt = 3
#                 dfCuttedIndex = list()
#                 for i in range(figCnt):
#                     dfCuttedIndex.append(floor(len(df) / figCnt) * i)
#                 dfCuttedIndex.append(len(df))
#                 #開始畫圖
#                 fig, axs = plt.subplots(figCnt, sharey = True)
#                 axIndexForLegned = 0
#                 for splitIndex in range(len(dfCuttedIndex) - 1):
#                     subDf = df.iloc[dfCuttedIndex[splitIndex]: dfCuttedIndex[splitIndex + 1]]
#                     plot = subDf.plot.bar(ax = axs[splitIndex], width = totalBarWidth, rot = 0, color = colorDict, edgecolor = 'black', linewidth = 0.2, legend = None)
#                     #找出B&H位置，將B&H的bar變成紅色
#                     BHIndex = [i for i, x in enumerate(subDf.index) if x == 'B&H']
#                     if not len(BHIndex):
#                         axIndexForLegned = splitIndex
#                     if len(BHIndex):
#                         for barIndex, barContainer in enumerate(plot.containers):
#                             if barIndex == 0:
#                                 singleBarWidth = barContainer[BHIndex[0]].get_width()
#                                 barX = ((barContainer[BHIndex[0]].get_x() + (barContainer[BHIndex[0]].get_x() + (len(subDf.columns) * singleBarWidth))) - singleBarWidth) / 2
#                             barContainer[BHIndex[0]].set_color('r')
#                             barContainer[BHIndex[0]].set_x(barX)
#                             barContainer[BHIndex[0]].set_edgecolor('black')
#                     #設定其他屬性
#                     axs[splitIndex].grid(axis = 'y')
#                     axs[splitIndex].yaxis.set_major_formatter(mtick.PercentFormatter())  #把座標變成%
#                     axs[splitIndex].set_xticklabels(subDf.index, rotation = 45)
#                     axs[splitIndex].set(xlabel = "", ylabel = "")
#                     axs[splitIndex].tick_params(axis = 'both', labelsize = allFontSize)  #設定xlabel ylabel字形大小
#                     #設定lable顏色
#                     for cellIndex in axs[splitIndex].get_xticklabels():
#                         txt = cellIndex.get_text()
#                         for slideGroup in slidingLableClrList:
#                             if txt in slideGroup[0]:
#                                 plt.setp(cellIndex, bbox = dict(boxstyle = 'round', edgecolor = 'none', alpha = 1, facecolor = slideGroup[1]))
#                                 break
#                     #設定top table跟bottom table
#                     if splitIndex == 0:
#                         if len(subDf.columns) < 5:
#                             myBox = [0, 1.05, 1, 0.22]
#                         else:
#                             myBox = [0, 1.05, 1, 0.4]
#                         celBGC = [['gainsboro'] * 6, ['silver'] * 4, ['darkgray'] * 4]
#                         colClrs = list()
#                         for i in range(figCnt):
#                             for clr in celBGC[i]:
#                                 colClrs.append(clr)
#                         topTable = axs[splitIndex].table(colLabels = tableDf.columns, cellText = tableDf.values, loc = 'top', cellLoc = 'center', colColours = colClrs, bbox = myBox)
#                         # for colIndex in range(len(tableDf.columns)):  #設定cell text顏色
#                         #     topTable[0, colIndex].get_text().set_color('white')
#                         topTable.auto_set_column_width(col = list(range(len(tableDf.columns))))
#                         topTable.auto_set_font_size(False)
#                         topTable.set_fontsize('medium')  # Valid font size are xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller, None
#                     if len(df.columns) > 4:
#                         continue
#                         chooseDf = windowChooseDf[windowChooseDf.columns[dfCuttedIndex[splitIndex]: dfCuttedIndex[splitIndex + 1]]]
#                         table = axs[splitIndex].table(cellText = chooseDf.values, loc = 'bottom', cellLoc = 'center', rowLabels = [chooseDf.index[0], chooseDf.index[1]], bbox = [0, 0, 1, 0.2])
#                 handles, labels = axs[axIndexForLegned].get_legend_handles_labels()
#                 fig.legend(handles, labels, loc = 'upper center', bbox_to_anchor = (0.5, 0), fancybox = True, shadow = False, ncol = len(df.columns), fontsize = allFontSize)
#                 # plt.legend(loc = 'upper center', bbox_to_anchor = (0.5, -0.3), fancybox = True, shadow = False, ncol = len(df.columns), fontsize = allFontSize)
#                 fig.tight_layout()
#                 fig.suptitle(company + '_' + techName + '_IRR_rank', y = 1.02, fontsize = allFontSize + 5)
#                 plt.savefig(company + '_all_IRR'  + '.png', dpi = 300, bbox_inches = 'tight')
#                 plt.cla()
#                 plt.close(fig)
#             # exit(0)
#     os.chdir(root)
#     for dfIndex, eachDf in enumerate(tables):
#         if dfIndex == 0:
#             eachDf.to_csv(fileName.split('.')[0] + '_tables.csv')
#         else:
#             eachDf.to_csv(fileName.split('.')[0] + '_tables.csv', mode = 'a', header = None)

# def split_hold_period():
#     for targetFolder in splitTargetFolder:
#         for comName in all_company:
#             if comName != '.DS_Store':
#                 os.chdir(comName+targetFolder)
#                 all_filename = [i for i in glob.glob(f"*{file_extension}")]
#                 for file in all_filename:
#                     if (targetFolder == '/testBestHold' and len(file.split('_')) == 2) or (targetFolder == '/specify' and file.split('_')[0] == 'hold' and len(file.split('_')) == 6):
#                         print('spliting ' + file)
#                         year = [str(i) for i in range(2013, 2021)]
#                         yIndex = []
#                         yIN = -1
#                         yIndex.append(int(1))
#                         # with open(file, 'rt') as f:
#                         #     reader = csv.reader(f, delimiter=',')
#                         #     for y in year:
#                         #         print(y)
#                         #         for row in reader:  # 預想是每break一次都會從第一個row開始，但是好像會接續從break時的row接續下去？？不知道為什麼
#                         #             print(row)
#                         #             yIN += 1
#                         #             if y == row[0].split('-')[0]:
#                         #                 yIndex.append(int(yIN))
#                         #                 yIndex.append(int(yIN))
#                         #                 break
#                         csvfile = open(''.join(file), 'r').readlines()
#                         for y in year:
#                             for row in range(len(csvfile)):
#                                 yIN += 1
#                                 if y == csvfile[yIN].split('-')[0]:
#                                     yIndex.append(int(yIN))
#                                     yIndex.append(int(yIN))
#                                     break
#                         yIndex.append(len(csvfile))
#                         for i in range(len(yIndex)):
#                             if i % 2 == 0:
#                                 if targetFolder == '/testBestHold':
#                                     f = open(
#                                         file.split('.')[0] + '_' + str(csvfile[yIndex[i]]).split('-')[0] + '_hold.csv', 'w+')
#                                 elif targetFolder == '/specify':
#                                     f = open(
#                                         file.split('.')[0] + '_' + str(csvfile[yIndex[i]]).split('-')[0] + '.csv', 'w+')
#                                 f.write('Date,Price,Hold\n')
#                                 f.writelines(csvfile[yIndex[i]:yIndex[i+1]])
#                                 f.close()
#             os.chdir(oowd)


# def draw_hold_period():
#     fig = plt.figure(figsize=[16, 4.5], dpi=300)
#     os.chdir(oowd)
#     now = 0
#     month = ['01', '02', '03', '04', '05', '06',
#              '07', '08', '09', '10', '11', '12']
#     year = [str(i) for i in range(2012, 2021)]
#     for targetFolder in splitTargetFolder:
#         for comName in all_company:
#             if now >= 0 and comName != '.DS_Store':
#                 print(comName)
#                 os.chdir(comName+targetFolder)
#                 NewAll_filename = [i for i in glob.glob(f"*{file_extension}")]
#                 print(NewAll_filename)
#                 for files in NewAll_filename:
#                     if files.split('_')[0] != 'RoR':
#                         df = pd.read_csv(files)
#                         xIndex = []
#                         dfIndex = -1
#                         if len(files.split('_')) == 4 or len(files.split('_')) == 7:
#                             for m in month:
#                                 for row in df.Date:
#                                     dfIndex += 1
#                                     if m == df.Date[dfIndex].split('-')[1]:
#                                         xIndex.append(int(dfIndex))
#                                         break
#                         else:
#                             for y in year:
#                                 for row in df.Date:
#                                     dfIndex += 1
#                                     if y == df.Date[dfIndex].split('-')[0]:
#                                         xIndex.append(int(dfIndex))
#                                         break
#                         plt.title(files.replace('.csv', ''))
#                         plt.plot(df.Date, df.Price, label='Price')
#                         plt.plot(df.Date, df.Hold, label='Hold')
#                         if len(files.split('_')) != 2 and files.split('_') != 6:
#                             plt.scatter(df.Date, df.Hold,
#                                         c='darkorange', s=5, zorder=10)
#                         plt.xlabel('Date', fontsize=12, c='black')
#                         plt.ylabel('Price', fontsize=12, c='black')

#                         plt.xticks(xIndex, fontsize=9)
#                         plt.yticks(fontsize=9)
#                         plt.legend()

#                         plt.grid()
#                         plt.savefig(files.split('.')[0]+'.png',
#                                     dpi=fig.dpi, bbox_inches='tight')
#                         plt.cla()
#                         gc.collect()
#                         print('line chart for ' + files + ' created')
#             now += 1
#             os.chdir(oowd)
#         plt.close(fig)

# def draw_hold():
#     fig = plt.figure(figsize=[16, 4.5], dpi=300)
#     for company in all_company:
#         os.chdir(company+'/testBestHold')
#         all_filename = [i for i in glob.glob(f"*{file_extension}")]
#         print(all_filename)
#         for file in all_filename:
#             df = pd.read_csv(file)
            
#             yearIndexes = []
#             year = 0
#             for i, date in enumerate(df['Date']):
#                 nowYear = int(date.split('-')[0])
#                 if year != nowYear:
#                     yearIndexes.append(i)
#                     year = nowYear
#             yearIndexes.append(df.index[-1])
            
#             for yearIndex in range(len(yearIndexes)-1):
#                 newDf = df.iloc[yearIndexes[yearIndex]:yearIndexes[yearIndex+1]]
#                 newDf.reset_index(inplace = True, drop = True)
                
#                 ax = plt.gca()
#                 ax.plot(newDf['Date'],newDf['Price'],label='Price',color='g')
#                 ax.plot(newDf['Date'],newDf['Hold'],label='Hold',color='r')
#                 ax.scatter(newDf['Date'],newDf['buy'],c='darkorange', s=8, zorder=10,label='buy')
#                 ax.scatter(newDf['Date'],newDf['sell'],c='purple', s=8, zorder=10,label='sell')
                
#                 # buy = [i for i in newDf.index if not np.isnan(newDf.at[i,'buy'])]
#                 # sell = [i for i in newDf.index if not np.isnan(newDf.at[i,'sell'])]
#                 # ax.vlines(buy, color='darkorange', linestyle='-',alpha=0.5,label='buy',ymin=0,ymax=max(newDf['Price']))
#                 # ax.vlines(sell, color='purple', linestyle='-',alpha=0.5,label='sell',ymin=0,ymax=max(newDf['Price']))
#                 mIndex = []
#                 month = 0
#                 for i, date in enumerate(newDf['Date']):
#                     nowMonth = int(date.split('-')[1])
#                     if month != nowMonth:
#                         mIndex.append(i)
#                         month = nowMonth
#                 mIndex.append(newDf.index[-1])
#                 # ax.set_xticks(mIndex)
#                 plt.xticks(mIndex,fontsize=9)
#                 plt.yticks(fontsize=9)
#                 ax.legend()
#                 ax.grid()
#                 ax.set_xlabel('Date', fontsize=12, c='black')
#                 ax.set_ylabel('Price', fontsize=12, c='black')
#                 title = file.replace('.csv','_') + newDf.at[0,'Date'].split('-')[0] + '_Hold'
#                 print(title)
#                 ax.set_title(title)
#                 plt.savefig(title +'.png',dpi=fig.dpi, bbox_inches='tight')
#                 plt.clf()
#         os.chdir(oowd)
        