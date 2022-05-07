from matplotlib import table
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os

root = os.getcwd()
file_extension = '.csv'

class draw_hold_period:
    fig = plt.figure(figsize = [21, 9], dpi = 300, constrained_layout = True)
    allFontSize = 15
    scatterClr = ['green', 'red', 'purple']
    tradeInfo = list()
    
    def __init__(self, year, tech, isTrain, isTradition, setCompany):
        self.algoOrTrad = (lambda x : 'Tradition' if x == 1 else '')(isTradition)
        self.trainOrTest = (lambda x : 'train' if x == 1 else 'test')(isTrain)
        os.chdir('../')
        parentFolder = os.getcwd()
        self.workRoot = [dir for dir in glob.glob(parentFolder + '/**/**/**') if 'exp_result' in dir and year in dir][0] + f'/result_{tech}/'
        self.workRoot = self.workRoot.replace(parentFolder + '/', '')
        os.chdir(self.workRoot)
        if setCompany != 'all':
            self.allCompay = setCompany
        else:
            self.allCompay = [dir for dir in os.listdir() if os.path.isdir(dir)]
        self.allBestHoldPath = [i + f'/{self.trainOrTest + self.algoOrTrad}BestHold/' for i in self.allCompay]
        self.fig = draw_hold_period.fig
        gridNum = 24        
        self.gs = self.fig.add_gridspec(
            gridNum, 1, 
            # wspace = 0, 
            # hspace = 1, 
            # top = 1, 
            # bottom = 0, 
            # left = 0.17, 
            # right = 0.845
            )
        for bestHoldDir in self.allBestHoldPath:
            os.chdir(bestHoldDir)
            holdFile = [i for i in glob.glob(f"*{file_extension}") if 'hold' in i]
            print(holdFile)
            for file in holdFile:
                self.process_file_and_draw(file)
            for i in range(2):
                os.chdir('../')
    
    def process_file_and_draw(self, file):
        df = pd.read_csv(file, index_col = 0)
        yearIndexes = []
        year = 0
        for i, date in enumerate(df.index):
            nowYear = date.split('-')[0]
            if year != nowYear:
                yearIndexes.append(i)
                year = nowYear
        yearIndexes.append(len(df))
        for yearIndex in range(len(yearIndexes)):
            if yearIndex == len(yearIndexes) - 1:
                newDf = df.iloc[yearIndexes[0] : yearIndexes[-1]]
            else:
                newDf = df.iloc[yearIndexes[yearIndex] : yearIndexes[yearIndex+1]]
                
            tradeInfo = self.record_tradInfo(newDf)
            tableDf = self.make_tableDf(tradeInfo, yearIndexes, yearIndex)
            
            self.draw_table(tableDf)
            self.plot_hold(file, df, newDf, yearIndexes, yearIndex, tradeInfo)
    
    def record_tradInfo(self, newDf):
        tradeInfo = list()
        
        buyX = [i for i in newDf.index if not np.isnan(newDf.at[i, 'buy'])]
        buyY = [i for i in newDf['buy'] if not np.isnan(i)]
        tradeInfo.append(['buy', buyX, buyY])
        
        sellDateX = [i for i in newDf.index if not np.isnan(newDf.at[i, 'sell date'])]
        sellDateY = [i for i in newDf['sell date'] if not np.isnan(i)]
        tradeInfo.append(['sell date', sellDateX, sellDateY])
        
        sellTechConditionX = [i for i in newDf.index if not np.isnan(newDf.at[i, newDf.columns[4]])]
        sellTechConditionY = [i for i in newDf[newDf.columns[4]] if not np.isnan(i)]
        tradeInfo.append([newDf.columns[4], sellTechConditionX, sellTechConditionY])
        
        return tradeInfo
    
    def make_tableDf(self, tradeInfo, yearIndexes, yearIndex):
        cellData =  list()
        
        cellData.append(['buy Num', len(tradeInfo[0][1])])
        cellData.append(['sell Num', len(tradeInfo[1][1]) + len(tradeInfo[2][1])])
        cellData.append([tradeInfo[1][0], len(tradeInfo[1][1])])
        cellData.append([tradeInfo[2][0], len(tradeInfo[2][1])])
        
        tableCol = [elem[0] for elem in cellData]
        cellData = np.array([elem[1] for elem in cellData]).reshape(1, len(tableCol))
        tableDf = pd.DataFrame(cellData,columns = tableCol)
        
        if yearIndex == len(yearIndexes) - 1:
            self.tradeInfo.append(tableDf)
        
        return tableDf
    
    def draw_table(self, tableDf):
        tableAx = self.fig.add_subplot(self.gs[0, : ])
        tableAx.axis('off')
        topTable = tableAx.table(
            colLabels = tableDf.columns,
            cellText = tableDf.values,
            cellLoc = 'center', 
            colColours = ['silver'] * (len(tableDf.columns)), 
            )
        topTable.auto_set_font_size(False)
        topTable.set_fontsize('large')  # xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller, None
        
    def plot_hold(self, file, df, newDf, yearIndexes, yearIndex, tradeInfo):
        ax = self.fig.add_subplot(self.gs[1: , :])
        ax.plot(newDf.index, newDf['Price'], label = 'Price', color = 'steelblue', linewidth = 4)
        ax.plot(newDf.index, newDf['Hold'], label = 'Hold', color = 'darkorange', linewidth = 4)
        if yearIndex != len(yearIndexes) - 1:
            for scaterInfo, scaterClr in zip(tradeInfo, draw_hold_period.scatterClr):
                ax.scatter(scaterInfo[1], scaterInfo[2], color = scaterClr, s = 40, zorder = 10, label = scaterInfo[0])
        
        # buy = [i for i in newDf.index if not np.isnan(newDf.at[i,'buy'])]
        # sell = [i for i in newDf.index if not np.isnan(newDf.at[i,'sell date'])]
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
            mIndex.append(len(newDf)-1)
            
        ax.set_xticks(mIndex, fontsize=draw_hold_period.allFontSize)
        ax.set_xlabel('Date', fontsize=draw_hold_period.allFontSize)
        ax.set_ylabel('Price', fontsize=draw_hold_period.allFontSize)
        ax.grid()
        handles, labels = ax.get_legend_handles_labels()
        self.fig.legend(
            handles, labels, 
            loc = 'upper center', 
            bbox_to_anchor = (0.5, 0), 
            fancybox = True, shadow = False, 
            ncol = len(newDf.columns), 
            fontsize = draw_hold_period.allFontSize)
        if yearIndex == len(yearIndexes) - 1:
            title = file.replace('.csv', '_') + df.index[0].split('-')[0] + '-' + df.index[len(df.index) - 1].split('-')[0]
        else:
            title = file.replace('.csv', '_') + newDf.index[0].split('-')[0]
        print(title)
        self.fig.suptitle(title, 
            y = 1,
            fontsize = draw_hold_period.allFontSize)
        self.fig.savefig(title +'.png', dpi = draw_hold_period.fig.dpi, bbox_inches = 'tight')
        plt.clf()
    

x = draw_hold_period('2021', 'RSI', 0, 0, 'all')
    

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
        