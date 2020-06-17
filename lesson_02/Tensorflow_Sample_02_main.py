from time import time
import numpy as np
import pandas as pd
import sklearn.preprocessing
import tensorflow as tf

# -------------------- Define Noramliaztion -----------------------------------
min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
def normalize( OriginalDataSource: np ):
    temp = []
    ColCount = OriginalDataSource.shape[1]
    for row in range(ColCount):
        temp.append( min_max_scaler.fit_transform(OriginalDataSource[:,row].reshape(-1, 1)).flatten())
    DS_Norm = np.array(temp).T
    return DS_Norm
# End def

def denormalize( NormalizedDataSource: np ):
    DS_deNorm = min_max_scaler.inverse_transform(NormalizedDataSource)
    return DS_deNorm
# End def
# -----------------------------------------------------------------------------

# -------------------- Define NN layer ----------------------------------------
def layer(output_dim, input_dim, inputs, activation=None):
    W = tf.Variable(tf.random_normal([input_dim, output_dim]))
    b = tf.Variable(tf.random_normal([1, output_dim]))
    XWb = tf.matmul(inputs, W) + b
    if activation is None:
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs, W, b
# End def
# -----------------------------------------------------------------------------

# -------------------- Define take next group batch dataset -------------------
def Next_Batch(startIndex, batchSize):
    batch_x = InputData[startIndex : startIndex+batchSize]
    batch_y = OutputData[ startIndex : startIndex+batchSize]
    return batch_x, batch_y
# End def
# -----------------------------------------------------------------------------

# -------------------- Define Param -------------------------------------------
InputDim = 1
OutputDim = 1
# -----------------------------------------------------------------------------

# -------------------- Step1. Read Data, Read Data from CSV -------------------
path = './Data/TestData.csv'
df_DataSource = pd.read_csv(path)

TotalDataCount = df_DataSource.__len__()
TraningDataCount = 3000
TestDataCount = TotalDataCount - TraningDataCount
# -----------------------------------------------------------------------------

# -------------------- Step2. Normalization, Normaliza all data ---------------
df_Norm = normalize(df_DataSource.to_numpy())
# -----------------------------------------------------------------------------

# -------------------- Step3. Split Input/Output Data -------------------------
InputData = df_Norm[:TraningDataCount, 0:1].reshape(-1, InputDim)
OutputData = df_Norm[:TraningDataCount, 1].reshape(-1, OutputDim)
# -----------------------------------------------------------------------------

# -------------------- Step4. Create Network structure ------------------------
x = tf.placeholder("float", [None, InputDim])
y_target = tf.placeholder("float", [None, OutputDim])

# Create Training network
y_predict, W, b = layer( output_dim=OutputDim, input_dim=InputDim, inputs=x, activation=None)

# Create Regression optimization param
loss_function = tf.reduce_mean(tf.square(y_predict - y_target))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)
# -----------------------------------------------------------------------------

# -------------------- Step5. Define Training Epoch ---------------------------
trainEpochs = 100
batchSize = 30
totalBatchs = int(TraningDataCount/batchSize)

epoch_list = []
loss_list = []

with tf.device('/cpu:0'):
    with tf.Session() as sess:
        sess.run( tf.global_variables_initializer())
        startTime = time()

        # Do the Traning epoch
        for epoch in range(trainEpochs):
            for i in range(totalBatchs):
                batch_x, batch_y = Next_Batch(i*batchSize, batchSize)
                sess.run( optimizer, feed_dict={ x: batch_x, y_target: batch_y })
            # End for 

            Curr_W, Curr_b, loss = sess.run([W, b, loss_function], feed_dict={ x: InputData, y_target: OutputData })
            epoch_list.append(epoch)
            loss_list.append(loss)
            print( "Train Epoch", '%02d' % (epoch+1), "\nLoss=", "{:.9f}".format(loss), "\nW=", Curr_W, "\nb=", Curr_b, "\n")
        # End for 

        print('Total Data Count =', TotalDataCount)
        print('Training Data Count =', TraningDataCount)
        print('Test Data Count =', TestDataCount)

        duration = time() - startTime
        print("Train Finished, takes:", "{:.3f}".format(duration), "s")


        # Predict Test Data       
        TestData_x = df_Norm[TraningDataCount:, 0:1].reshape(-1, InputDim)
        TestData_y = df_Norm[TraningDataCount:, 1].reshape(-1, OutputDim)

        Prdict_y = sess.run(y_predict, feed_dict={ x: TestData_x })

        # Denormalize to get real value.
        deNormal_Test_y = denormalize(TestData_y)
        deNormal_Predict_y = denormalize(Prdict_y)
        RMSE_Value = np.sqrt(((deNormal_Predict_y - deNormal_Test_y) ** 2).mean())

        print( "Denormalize value-> \n Predict_y: ",  
               np.array2string(deNormal_Predict_y, formatter={'float_kind':lambda x: "%.2f" % x}), "\n\n",
               "Test_y: ", deNormal_Test_y)

        print("RMSE error is: ", "{:.9f}".format(RMSE_Value))


    # End with
# End with






