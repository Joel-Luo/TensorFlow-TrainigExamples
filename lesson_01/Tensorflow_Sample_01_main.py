import numpy as np
import pandas as pd
import sklearn.preprocessing
import tensorflow as tf


def main():

    # -------------------- Step1. Read Data, Read Data from CSV -------------------
    path = './Data/TestData.csv'
    df_DataSource = pd.read_csv(path)
    print('Total DataRow Count =', df_DataSource.__len__(), "days")
    
    DataColumn = len(df_DataSource.columns)
    Input_Col =1 
    Output_Col = DataColumn - Input_Col
    df_Norm = df_DataSource.to_numpy()
    # ----------------------------------------------------------------------------
    
    # -------------------- Step2. Split Input/Output Data  ------------------------
    InputData = df_Norm[:, 0:1].reshape(-1,Input_Col)
    OutputData = df_Norm[:, 1].reshape(-1,Output_Col)
    # -----------------------------------------------------------------------------
    
    # -------------------- Step3. Create Tensorflow module(graphic) ---------------
    W = tf.Variable(tf.compat.v1.random_normal([1,1]))
    b = tf.Variable(tf.compat.v1.random_normal([1,1]))
    
    ## Create Container for attaching Input/label data
    x = tf.compat.v1.placeholder(tf.float32, [None, 1])
    y = tf.compat.v1.placeholder(tf.float32, [None, 1])
    
    # Build a simple network
    linear_model = tf.matmul(x,W) + b
    
    # loss function
    loss = tf.reduce_mean(tf.square(linear_model - y))
    
    # optimizer
    train = tf.compat.v1.train.AdamOptimizer(0.01).minimize(loss)
    # -----------------------------------------------------------------------------
    
    # -------------------- Step4. Run Graphic -------------------------------------
    with tf.device('/cpu:0'):
    	
        with tf.compat.v1.Session() as sess:
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
    
            # loop for optimization
            loop = 1000
            for i in range(loop):
                sess.run(train, {x: InputData, y: OutputData})
    
                step = loop/10
                if i % step == 0: 
                    # Print output result
                    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: InputData, y: OutputData})
                    print("Itor: %s  W: %s b: %s loss: %s" % ( i/step, curr_W, curr_b, curr_loss))
                # End if

            # End for

        # End with

    # End with    
    
    # -----------------------------------------------------------------------------
            
    print( "Finished lesson_01 Demo" )

# End main def

if __name__ == "__main__":
    main()


