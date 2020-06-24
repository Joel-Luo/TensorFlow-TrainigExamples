from time import time
import numpy as np
import pandas as pd
import sklearn.preprocessing
import tensorflow as tf

# -------------------- Define Noramliaztion -----------------------------------
def normalize( OriginalDataSource: np, min_max_scaler):
    temp = []
    ColCount = OriginalDataSource.shape[1]
    for row in range(ColCount):
        temp.append( min_max_scaler.fit_transform(OriginalDataSource[:,row].reshape(-1, 1)).flatten())
    DS_Norm = np.array(temp).T
    return DS_Norm
# End def

def denormalize( NormalizedDataSource: np, min_max_scaler):
    DS_deNorm = min_max_scaler.inverse_transform(NormalizedDataSource)
    return DS_deNorm
# End def
# -----------------------------------------------------------------------------

# -------------------- Define NN layer ----------------------------------------
def layer(output_dim, input_dim, inputs, activation=None):
    W = tf.Variable(tf.compat.v1.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.compat.v1.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs
# End def
# -----------------------------------------------------------------------------

# -------------------- Define take next group batch dataset -------------------
def Next_Batch(InputData, OutputData, startIndex, batchSize):
    batch_x = InputData[ startIndex : startIndex + batchSize ]
    batch_y = OutputData[ startIndex : startIndex + batchSize ]
    return batch_x, batch_y
# End def
# -----------------------------------------------------------------------------

def main():

    # -------------------- Define Param -------------------------------------------
    InputDim = 1
    OutputDim = 1
    HiddenDim = 10
    # -----------------------------------------------------------------------------

    # -------------------- Step1. Read Data, Read Data from CSV -------------------
    path = './Data/TestData.csv'
    df_DataSource = pd.read_csv(path)
    
    TotalDataCount = df_DataSource.__len__()
    TraningDataCount = 3000
    TestDataCount = TotalDataCount - TraningDataCount
    # -----------------------------------------------------------------------------
    
    # -------------------- Step2. Normalization, Normaliza all data ---------------
    min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    df_Norm = normalize(df_DataSource.to_numpy(), min_max_scaler)
    # -----------------------------------------------------------------------------
    
    # -------------------- Step3. Split Input/Output Data -------------------------
    InputData = df_Norm[:TraningDataCount, 0:1].reshape(-1, InputDim)
    OutputData = df_Norm[:TraningDataCount, 1].reshape(-1, OutputDim)
    # -----------------------------------------------------------------------------
    
    # -------------------- Step4. Create Network structure ------------------------
    x = tf.compat.v1.placeholder("float", [None, InputDim])
    y_target = tf.compat.v1.placeholder("float", [None, OutputDim])
    
    # Create Training network
    h1 = layer( output_dim=HiddenDim, input_dim=InputDim, inputs=x, activation=tf.nn.relu)
    y_predict = layer( output_dim=OutputDim, input_dim=HiddenDim, inputs=h1, activation=None)
    
    # Create Regression optimization param
    loss_function = tf.reduce_mean(tf.square(y_predict - y_target))
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)
    # -----------------------------------------------------------------------------
    
    # -------------------- Step5. Define Training Epoch ---------------------------
    trainEpochs = 300
    batchSize = 30
    totalBatchs = int(TraningDataCount/batchSize)
    
    with tf.device('/cpu:0'):
        with tf.compat.v1.Session() as sess:
            sess.run( tf.compat.v1.global_variables_initializer())
    
            # Restore Model
            SaveModelPath = './Model/PredictModel.ckpt'
            ModelSaver = tf.compat.v1.train.Saver()
            save_path = ModelSaver.restore(sess, SaveModelPath)
            print("Restore Predict Model from file: %s" % save_path)
    
            print('Total Data Count =', TotalDataCount)
            print('Training Data Count =', TraningDataCount)
            print('Test Data Count =', TestDataCount)
    
            # Predict Test Data       
            TestData_x = df_Norm[TraningDataCount:, 0:1].reshape(-1, InputDim)
            TestData_y = df_Norm[TraningDataCount:, 1].reshape(-1, OutputDim)
    
            Prdict_y = sess.run(y_predict, feed_dict={ x: TestData_x })
    
            # Denormalize to get real value.
            deNormal_Test_y = denormalize(TestData_y, min_max_scaler)
            deNormal_Predict_y = denormalize(Prdict_y, min_max_scaler)
            RMSE_Value = np.sqrt(((deNormal_Predict_y - deNormal_Test_y) ** 2).mean())
    
            print( "Denormalize value-> ",
                   "\n\nPredict_y: ",  
                   np.array2string(deNormal_Predict_y, formatter={'float_kind':lambda x: "%.2f" % x}), 
                   "\n\nTest_y: ", 
                   np.array2string(deNormal_Test_y, formatter={'float_kind':lambda x: "%.2f" % x}))
    
            print("RMSE error is: ", "{:.9f}".format(RMSE_Value))
    
        # End with

    # End with

    # -----------------------------------------------------------------------------
            
    print( "Finished lesson_03 Demo" )

# End main def 

if __name__ == "__main__":
    main()


