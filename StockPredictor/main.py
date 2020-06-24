import numpy as np
from time import time
from DataSourceManager import DataSourceManager
from DataSourceProcessor import DataSourceProcessor
from NetworkStructureManager import NetworkStructureManager

# -------------------- Define Param -------------------------------------------
StockName = '2330'
InputFileDirection = './Data/'
DataFileName = '_Data-Full.csv'

InputCol = 4
ReferDays = 20
InputDim = InputCol*ReferDays
HiddenDim = InputDim
OutputDim = 1
BatchSize = 80

ModelDirection = './Model/'
# -----------------------------------------------------------------------------

def main():
    DSM = DataSourceManager()
    DSP = DataSourceProcessor()
    NSM = NetworkStructureManager()   
    
    # Get Stock Data as data frame
    s_DataPath = InputFileDirection + StockName + DataFileName
    df_DataSource = DSM.GetDataSourceFromCSV(s_DataPath)
    print('Total Date =', df_DataSource.__len__(), "days")

    # Create DataSourceProcessor    
    DSP.CreateDataProcessor(feature_range=(-1,1))
    np_DataSourceNormal = DSP.NormalizeDataSource(df_DataSource.to_numpy())
    [np_InputData, np_OutputData] = DSP.GenerateIODataFrame(np_DataSourceNormal, InputCol, ReferDays)
    
    # Split as multi Data set
    np_InputData_List = DSP.SplitDataFrame(np_InputData, BatchSize)
    np_OutputData_List = DSP.SplitDataFrame(np_OutputData, BatchSize)
    numOfBatch = np_InputData_List.shape[0]
    print('Total Batch Count =', numOfBatch)
    
    startTime = time()
    
    # For each Data set to training model
    for BatchId in range(numOfBatch):
    
        # Get Training Data
        batch_x = np_InputData_List[BatchId]
        batch_y = np_OutputData_List[BatchId]
        
        # Start to train model of each data set 
        SaveModelPath = ModelDirection + 'PredictModel_'+ '%02d' % (BatchId+1) + '.ckpt'
        NSM.CreateNewModelAndTraining( BatchId, batch_x, batch_y, SaveModelPath, InputDim, HiddenDim, OutputDim )

        duration = time() - startTime
        print("Train" , '%02d' % (BatchId+1), "Finished takes:", "{:.3f}".format(duration), "s\n")

        # Verify model
        TestData_x = batch_x[-10:]
        TestData_y = batch_y[-10:]
        predict_y = NSM.PredictData( TestData_x, TestData_y, SaveModelPath, InputDim, HiddenDim, OutputDim )

        # Denormalize to get real value.
        deNormal_Test_y = DSP.DenormalizeDataSource(TestData_y)
        deNormal_Predict_y = DSP.DenormalizeDataSource(predict_y)
        RMSE_Value = np.sqrt(((deNormal_Predict_y - deNormal_Test_y) ** 2).mean())

        print( "Denormalize value-> ",
               "\n\nPredict_y: ",  
               np.array2string(deNormal_Predict_y, formatter={'float_kind':lambda x: "%.2f" % x}), 
               "\n\nTest_y: ", 
               np.array2string(deNormal_Test_y, formatter={'float_kind':lambda x: "%.2f" % x}))

        print("RMSE error is: ", "{:.9f}".format(RMSE_Value))
    # End for

# End main def

if __name__ == "__main__":
    main()