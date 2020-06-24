import sklearn.preprocessing
import pandas as pd
import numpy as np

class DataSourceProcessor():
    """ Data Source Processor """

    #Contructor
    def __init__(self) -> None:
        self.__m_scaler = None
        return
    # End def        

    @classmethod
    def NormalizeDataSource(self, OriginalDataSource: np) -> np:
        
        temp = []
        ColCount = OriginalDataSource.shape[1]
        for row in range(ColCount):
            temp.append( self.__m_scaler.fit_transform(OriginalDataSource[:,row].reshape(-1, 1)).flatten())
        DS_Norm = np.array(temp).T
        return DS_Norm
    # End def

    @classmethod
    def DenormalizeDataSource(self, NormalizedDataSource: np ) -> np:
        DS_deNorm = self.__m_scaler.inverse_transform(NormalizedDataSource)
        return DS_deNorm
    # End def

    @classmethod
    def CreateDataProcessor(self, feature_range=(-1, 1)) -> None:
        """ Create DataProcessor (Data Scaler), default feature range=(-1,1) """

        self.__m_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=feature_range)

    # End def

    @classmethod
    def GenerateIODataFrame(self, DataSource: np, InputCol, ReferDays) -> [np, np] :
        
        InputDataTable = []
        OutputDataTable = []
        TotalTestDataLen = DataSource.__len__() - ReferDays
        print('TestData Count =', TotalTestDataLen, "Rows")

        for i in range(TotalTestDataLen):
            InputDataRow =  np.array(DataSource[i:i+ReferDays, 0:InputCol]).flatten()
            OutputDataRow = np.array(DataSource[i+ReferDays, (InputCol-1)]).flatten()
            InputDataTable.append(InputDataRow)
            OutputDataTable.append(OutputDataRow)
        # End for

        InputData = np.array(InputDataTable)
        OutputData = np.array(OutputDataTable)
        return [InputData, OutputData]
    # End def


    @classmethod
    def SplitDataFrame(self, DataSource: np, FrameSize) -> np:

        DataFrameList = []
        DataSourceSize = DataSource.__len__()
        offset = DataSourceSize%FrameSize

        DataFrameListSize = int(DataSourceSize/FrameSize)

        for i in range(DataFrameListSize):
            StartIndex = i*FrameSize + offset
            EndIndex = StartIndex + FrameSize
            SubDataFrame = DataSource[StartIndex:EndIndex]
            DataFrameList.append(SubDataFrame)
        # End for

        np_DataFrameList = np.array(DataFrameList)

        return np_DataFrameList

    # End def   
# End class    