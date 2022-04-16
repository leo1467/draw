from email import header
from itertools import count
from math import floor
from operator import le
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import os
import gc
import matplotlib.ticker as mtick
from sympy import false, rotations

owd = os.getcwd()
# all_company = os.listdir(os.chdir('result'))
# all_company.sort()
oowd = os.getcwd()
file_extension = '.csv'
splitTargetFolder = ['/testBestHold']


def split_hold_period():
    for targetFolder in splitTargetFolder:
        for comName in all_company:
            if comName != '.DS_Store':
                os.chdir(comName+targetFolder)
                all_filename = [i for i in glob.glob(f"*{file_extension}")]
                for file in all_filename:
                    if (targetFolder == '/testBestHold' and len(file.split('_')) == 2) or (targetFolder == '/specify' and file.split('_')[0] == 'hold' and len(file.split('_')) == 6):
                        print('spliting ' + file)
                        year = [str(i) for i in range(2013, 2021)]
                        yIndex = []
                        yIN = -1
                        yIndex.append(int(1))
                        # with open(file, 'rt') as f:
                        #     reader = csv.reader(f, delimiter=',')
                        #     for y in year:
                        #         print(y)
                        #         for row in reader:  # 預想是每break一次都會從第一個row開始，但是好像會接續從break時的row接續下去？？不知道為什麼
                        #             print(row)
                        #             yIN += 1
                        #             if y == row[0].split('-')[0]:
                        #                 yIndex.append(int(yIN))
                        #                 yIndex.append(int(yIN))
                        #                 break
                        csvfile = open(''.join(file), 'r').readlines()
                        for y in year:
                            for row in range(len(csvfile)):
                                yIN += 1
                                if y == csvfile[yIN].split('-')[0]:
                                    yIndex.append(int(yIN))
                                    yIndex.append(int(yIN))
                                    break
                        yIndex.append(len(csvfile))
                        for i in range(len(yIndex)):
                            if i % 2 == 0:
                                if targetFolder == '/testBestHold':
                                    f = open(
                                        file.split('.')[0] + '_' + str(csvfile[yIndex[i]]).split('-')[0] + '_hold.csv', 'w+')
                                elif targetFolder == '/specify':
                                    f = open(
                                        file.split('.')[0] + '_' + str(csvfile[yIndex[i]]).split('-')[0] + '.csv', 'w+')
                                f.write('Date,Price,Hold\n')
                                f.writelines(csvfile[yIndex[i]:yIndex[i+1]])
                                f.close()
            os.chdir(oowd)


def draw_hold_period():
    fig = plt.figure(figsize=[16, 4.5], dpi=300)
    os.chdir(oowd)
    now = 0
    month = ['01', '02', '03', '04', '05', '06',
             '07', '08', '09', '10', '11', '12']
    year = [str(i) for i in range(2012, 2021)]
    for targetFolder in splitTargetFolder:
        for comName in all_company:
            if now >= 0 and comName != '.DS_Store':
                print(comName)
                os.chdir(comName+targetFolder)
                NewAll_filename = [i for i in glob.glob(f"*{file_extension}")]
                print(NewAll_filename)
                for files in NewAll_filename:
                    if files.split('_')[0] != 'RoR':
                        df = pd.read_csv(files)
                        xIndex = []
                        dfIndex = -1
                        if len(files.split('_')) == 4 or len(files.split('_')) == 7:
                            for m in month:
                                for row in df.Date:
                                    dfIndex += 1
                                    if m == df.Date[dfIndex].split('-')[1]:
                                        xIndex.append(int(dfIndex))
                                        break
                        else:
                            for y in year:
                                for row in df.Date:
                                    dfIndex += 1
                                    if y == df.Date[dfIndex].split('-')[0]:
                                        xIndex.append(int(dfIndex))
                                        break
                        plt.title(files.replace('.csv', ''))
                        plt.plot(df.Date, df.Price, label='Price')
                        plt.plot(df.Date, df.Hold, label='Hold')
                        if len(files.split('_')) != 2 and files.split('_') != 6:
                            plt.scatter(df.Date, df.Hold,
                                        c='darkorange', s=5, zorder=10)
                        plt.xlabel('Date', fontsize=12, c='black')
                        plt.ylabel('Price', fontsize=12, c='black')

                        plt.xticks(xIndex, fontsize=9)
                        plt.yticks(fontsize=9)
                        plt.legend()

                        plt.grid()
                        plt.savefig(files.split('.')[0]+'.png',
                                    dpi=fig.dpi, bbox_inches='tight')
                        plt.cla()
                        gc.collect()
                        print('line chart for ' + files + ' created')
            now += 1
            os.chdir(oowd)
        plt.close(fig)


def split_testIRR_draw(split,fileName):
    fileNameList = fileName.split('.')[0].split('_')
    techName = '_'.join(fileNameList[fileNameList.index('sorted') + 1: ])
    os.chdir('result')
    dirName = 'split_' + fileName.split('.')[0]
    if not os.path.isdir(dirName):
            os.mkdir(dirName)
    if split:
        oriIRRFile = pd.read_csv(fileName)
        oriIRRFile = oriIRRFile.loc[: , ~oriIRRFile.columns.str.contains('^Unnamed')]
        oriIRRFile.to_csv(fileName, index = None)
        oriIRRFile = pd.read_csv(fileName, header = None)
        os.chdir(dirName)
        index = []
        for row, content in enumerate(oriIRRFile[0]):
            if content[0] == '=':
                index.append(row)
        index.append(len(oriIRRFile))
        for cellIndex in range(len(index) - 1):
            oriIRRFile[index[cellIndex]: index[cellIndex + 1]].to_csv(oriIRRFile[0][index[cellIndex]].replace('=', '') + '_IRR.csv', header = None, index = None)
    else:
        os.chdir(dirName)
    allIRRFile = [i for i in glob.glob(f"*{file_extension}")]
    #設定滑動視窗group
    slidingLableClrList = [[['YYY2YYY', 'YYY2YY', 'YYY2Y', 'YYY2YH', 'YY2YY', 'YY2YH', 'YY2Y', 'YH2YH', 'YH2Y', 'Y2Y'], 'gold'],
                           [['YYY2H', 'YYY2Q', 'YYY2M', 'YY2H', 'YY2Q', 'YY2M', 'YH2H', 'YH2Q', 'YH2M', 'Y2H', 'Y2Q', 'Y2M'], 'limegreen'],
                           [['H2H', 'H#', 'H2Q', 'Q2Q', 'Q#', 'H2M', 'Q2M', 'M2M', 'M#'], 'r'],
                           [['20D20', '20D15', '20D10', '20D5', '15D15', '15D10', '15D5', '10D10', '10D5', '5D5'], 'grey'],
                           [['4W4', '4W3', '4W2', '4W1', '3W3', '3W2', '3W1', '2W2', '2W1', '1W1'], 'darkgoldenrod']]
    #設定bar屬性
    barColorSet = ['steelblue', 'darkorange', 'lightcyan', 'wheat', 'lightcyan', 'lightyellow']
    totalBarWidth = 0.85
    singleBarWidth = 0.20
    #儲存全部table資料
    tables = []
    #設定figure大小
    plt.rcParams['figure.figsize'] = (21, 9)
    allFontSize = 12
    for Fileindex, file in enumerate(allIRRFile):
        print(file)
        if file.split('.')[1] == 'csv':
            #df前處理
            df = pd.read_csv(file)
            company = df.columns[0].replace('=', '')
            df.rename(columns = {df.columns[0]: 'window'}, inplace = True)
            for col in range(1, len(df.columns)):
                for row in range(len(df)):
                    df.at[row, df.columns[col]] *= 100
            # table資料
            tableColumns = ['highest IRR diff b/t algo and trad', 'algo win rate', 'algo avg IRR', 'trad avg IRR']
            cellData = list()
            IRRData = dict()
            for colIndex, col in enumerate(df.columns):
                if colIndex != 0:
                    IRRData.update({col: np.array([x for i, x in enumerate(df[col]) if df.window[i] != 'B&H'])})
            cellData.append(max(IRRData[df.columns[1]]) - max(IRRData[df.columns[2]]))
            cellData.append(len([i for i, j in zip(IRRData[df.columns[1]], IRRData[df.columns[2]]) if i > j]) / len(IRRData[df.columns[1]]) * 100)
            cellData.append(np.average(IRRData[df.columns[1]]))
            cellData.append(np.average(IRRData[df.columns[2]]))
            if len(df.columns) > 3:
                tableColumns.append('highest IRR diff of algo of tech')
                cellData.append(max(IRRData[df.columns[1]]) - max(IRRData[df.columns[3]]))
                tableColumns.append('highest IRR diff of trad of tech')
                cellData.append(max(IRRData[df.columns[2]]) - max(IRRData[df.columns[4]]))
                tableColumns.append('algo win rate of tech')
                cellData.append(len([i for i, j in zip(IRRData[df.columns[1]], IRRData[df.columns[3]]) if i > j]) / len(IRRData[df.columns[1]]) * 100)
                tableColumns.append('trad win rate of tech')
                cellData.append(len([i for i, j in zip(IRRData[df.columns[2]], IRRData[df.columns[4]]) if i > j]) / len(IRRData[df.columns[1]]) * 100)
            if len(df.columns) > 5:
                windowChoose = list()
                windowChoose.append(df[df.columns[7]] / df['window num'] * 100)
                windowChoose.append(df[df.columns[9]] / df['window num'] * 100)
                for i in range(len(windowChoose)):
                    windowChoose[i] = ['%.2f' % elem + '%' for elem in windowChoose[i]]
                windowChooseDf = pd.DataFrame(windowChoose, columns = [df.window], index = [df.columns[7], df.columns[9]])
                df = df.drop(columns = [col for col in df.columns[-5: ]])
            cellData = ['%.2f' % elem for elem in cellData]
            cellData = np.array([[elem + '%'] for elem in cellData])
            tableDf = pd.DataFrame(cellData.reshape(1, len(tableColumns)), columns = tableColumns)
            tableDf.rename(index = { 0: company }, inplace = True)
            tables.append(tableDf)
            #設定bar寬度
            colSet = df.columns[1: ]
            colorDict = dict(zip(colSet, barColorSet))
            if len(colSet) * singleBarWidth < totalBarWidth:
                barWidth = len(colSet) * singleBarWidth
            else:
                barWidth = totalBarWidth
            #將過長的df切一半
            dfCuttedIndex = [0, floor(len(df) / 2), len(df)]
            #開始畫圖
            fig, axs = plt.subplots(2, sharey = True)
            for splitIndex in range(len(dfCuttedIndex) - 1):
                #將過長的df切一半
                subDf = df.iloc[dfCuttedIndex[splitIndex]: dfCuttedIndex[splitIndex + 1]]
                #plot bar
                subDf.plot(ax = axs[splitIndex], kind = 'bar', width = barWidth, rot = 0, color = colorDict, edgecolor = 'black', linewidth = 0.2, legend = None)
                axs[splitIndex].grid(axis = 'y')
                axs[splitIndex].yaxis.set_major_formatter(mtick.PercentFormatter())  #把座標變成%
                axs[splitIndex].set_xticklabels(subDf.window, rotation = 45)
                axs[splitIndex].tick_params(axis = 'both', labelsize = allFontSize)  #設定xlabel ylabel字形大小
                #設定lable顏色
                for cellIndex in axs[splitIndex].get_xticklabels():
                    txt = cellIndex.get_text()
                    for slideGroup in slidingLableClrList:
                        if txt in slideGroup[0]:
                            plt.setp(cellIndex, bbox = dict(boxstyle = 'round', edgecolor = 'none', alpha = 1, facecolor = slideGroup[1]))
                            break
                #設定top table跟bottom table
                if splitIndex == 0:
                    topTable = axs[splitIndex].table(colLabels = tableDf.columns,
                                                     cellText = tableDf.values,
                                                     loc = 'top',
                                                     cellLoc = 'center')
                    topTable.auto_set_font_size(False)
                    topTable.set_fontsize('medium')
                    # Valid font size are xx-small, x-small, small, medium, large, x-large, xx-large, larger, smaller, None
                if len(df.columns) > 5:
                    if splitIndex == 0:
                        chooseDf = windowChooseDf[windowChooseDf.columns[0: floor(len(windowChooseDf.columns) / 2)]]
                    elif splitIndex == 1:
                        chooseDf = windowChooseDf[windowChooseDf.columns[floor(len(windowChooseDf.columns) / 2): len(windowChooseDf.columns)]]
                    table = axs[splitIndex](cellText = chooseDf.values, 
                                loc = 'bottom', 
                                cellLoc = 'center', 
                                rowLabels = [chooseDf.index[0], chooseDf.index[1]],
                                bbox = [0.0, -0.6, 1, 0.28])
            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc = 'upper center', bbox_to_anchor = (0.5, 0), fancybox = True, shadow = False, ncol = 6, fontsize = allFontSize)
            fig.tight_layout()
            fig.suptitle(company + '_' + techName + 'IRR_rank', y = 1.02, fontsize = allFontSize + 5)
            plt.savefig(company + '_all_IRR'  + '.png', dpi = 300, bbox_inches = 'tight')
            plt.cla()
            plt.close()
        exit(0)
    os.chdir(owd + '/result/')
    for dfIndex, eachDf in enumerate(tables):
        if dfIndex == 0:
            eachDf.to_csv(fileName.split('.')[0] + '_tables.csv')
        else:
            eachDf.to_csv(fileName.split('.')[0] + '_tables.csv', mode = 'a', header = None)
        

def draw_hold():
    fig = plt.figure(figsize=[16, 4.5], dpi=300)
    for company in all_company:
        os.chdir(company+'/testBestHold')
        all_filename = [i for i in glob.glob(f"*{file_extension}")]
        print(all_filename)
        for file in all_filename:
            df = pd.read_csv(file)
            
            yearIndexes = []
            year = 0
            for i, date in enumerate(df['Date']):
                nowYear = int(date.split('-')[0])
                if year != nowYear:
                    yearIndexes.append(i)
                    year = nowYear
            yearIndexes.append(df.index[-1])
            
            for yearIndex in range(len(yearIndexes)-1):
                newDf = df.iloc[yearIndexes[yearIndex]:yearIndexes[yearIndex+1]]
                newDf.reset_index(inplace = True, drop = True)
                
                ax = plt.gca()
                ax.plot(newDf['Date'],newDf['Price'],label='Price',color='g')
                ax.plot(newDf['Date'],newDf['Hold'],label='Hold',color='r')
                ax.scatter(newDf['Date'],newDf['buy'],c='darkorange', s=8, zorder=10,label='buy')
                ax.scatter(newDf['Date'],newDf['sell'],c='purple', s=8, zorder=10,label='sell')
                
                # buy = [i for i in newDf.index if not np.isnan(newDf.at[i,'buy'])]
                # sell = [i for i in newDf.index if not np.isnan(newDf.at[i,'sell'])]
                # ax.vlines(buy, color='darkorange', linestyle='-',alpha=0.5,label='buy',ymin=0,ymax=max(newDf['Price']))
                # ax.vlines(sell, color='purple', linestyle='-',alpha=0.5,label='sell',ymin=0,ymax=max(newDf['Price']))
                mIndex = []
                month = 0
                for i, date in enumerate(newDf['Date']):
                    nowMonth = int(date.split('-')[1])
                    if month != nowMonth:
                        mIndex.append(i)
                        month = nowMonth
                mIndex.append(newDf.index[-1])
                # ax.set_xticks(mIndex)
                plt.xticks(mIndex,fontsize=9)
                plt.yticks(fontsize=9)
                ax.legend()
                ax.grid()
                ax.set_xlabel('Date', fontsize=12, c='black')
                ax.set_ylabel('Price', fontsize=12, c='black')
                title = file.replace('.csv','_') + newDf.at[0,'Date'].split('-')[0] + '_Hold'
                print(title)
                ax.set_title(title)
                plt.savefig(title +'.png',dpi=fig.dpi, bbox_inches='tight')
                plt.clf()
        os.chdir(oowd)
        
def draw_test_IRR():
    tagetFile = ''
    

if __name__ == '__main__':
    # draw_hold()
    split_testIRR_draw(1, 'test_IRR_IRR_sorted_SMA_' + '.csv')
    