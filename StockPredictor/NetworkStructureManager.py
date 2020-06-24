import tensorflow.compat.v1 as tf
import numpy as np


class NetworkStructureManager():
    """ Data Training Processor """

    #Contructor
    def __init__(self) -> None:
        return
    # End def

    # ------------------- Define NN layer -------------------------------------
    @classmethod
    def __layer(self, output_dim, input_dim, inputs, activation=None):
        W = tf.Variable(tf.random_normal([input_dim, output_dim]))
        b = tf.Variable(tf.random_normal([1, output_dim]))
        XWb = tf.matmul(inputs, W) + b
        if activation is None:
            outputs = XWb
        # End if
        else:
            outputs = activation(XWb)
        # End else
        return outputs
    # End def
    # -------------------------------------------------------------------------

    # -------------------- Create Network structure ---------------------------
    @classmethod
    def __CreateNetworkStructure(self, x_input, y_target, InputDim, HiddenDim, OutputDim):

        # Create Training network
        h1 = self.__layer(output_dim=HiddenDim, input_dim=InputDim, inputs=x_input, activation=tf.nn.relu)
        y_predict = self.__layer( output_dim=OutputDim, input_dim=HiddenDim, inputs=h1, activation=None)

        # Create Regression optimization param
        loss_function = tf.reduce_mean(tf.square(y_predict - y_target))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)

        return [y_predict, loss_function, optimizer ]
    # End def
    # -------------------------------------------------------------------------
    
    # -------------------- Training Model and Save ------------------------------
    @classmethod
    def CreateNewModelAndTraining( self, batchId, DataSet_x, DataSet_y, SaveModelPath, InputDim, HiddenDim, OutputDim ):

        # Create Input data placehodler for tf
        x_input = tf.placeholder("float", [None, InputDim])
        y_target = tf.placeholder("float", [None, OutputDim])

        # Create Network structure to generate NN mode
        [ y_predict, loss_function, optimizer ] = self.__CreateNetworkStructure(x_input, y_target, InputDim, HiddenDim, OutputDim)

        with tf.Session() as sess:
            # Initial tf sess
            sess.run( tf.global_variables_initializer())
          
            epoch = 1
            while True:
                # Do Training 
                sess.run( optimizer, feed_dict={ x_input: DataSet_x, y_target: DataSet_y })

                # Caluate loss
                loss = sess.run(loss_function, feed_dict={ x_input: DataSet_x, y_target: DataSet_y })
                  
                if epoch % 1000 == 999:
                    print( "Batch", '%02d' % (batchId+1), "Train Epoch", '%02d' % (epoch+1), "\nLoss=", "{:.9f}".format(loss), "\n")
                # if End
            
                if loss < 0.000001 and epoch >= 5000 :
                    print( "Batch", '%02d' % (batchId+1), "Train Epoch", '%02d' % (epoch+1), "\nLoss=", "{:.9f}".format(loss), "\n")
                    break;
                # if End
                epoch = epoch + 1
            # End for 

            # Save Model
            ModelSaver = tf.train.Saver()
            save_path = ModelSaver.save(sess, SaveModelPath)
            print("Predict Model saved in file: %s" % save_path)

            sess.close()

        # End with

        tf.reset_default_graph()
    # End def
    # -------------------------------------------------------------------------

    # ----------------- Use Training Model to predict data --------------------
    @classmethod
    def PredictData( self, DataSet_x, DataSet_y, SaveModelPath, InputDim, HiddenDim, OutputDim ):
        # Create Input data placehodler for tf
        x_input = tf.placeholder("float", [None, InputDim])
        y_target = tf.placeholder("float", [None, OutputDim])
        # Create Network structure to generate NN mode
        [ y_predict, loss_function, optimizer ] = self.__CreateNetworkStructure(x_input, y_target, InputDim, HiddenDim, OutputDim)

        with tf.device('/cpu:0'):
            with tf.Session() as sess:
                sess.run( tf.global_variables_initializer())
              
                # Restore Model
                ModelSaver = tf.train.Saver()
                save_path = ModelSaver.restore(sess, SaveModelPath)
                print("Restore Predict Model from file: %s" % save_path)

                # Predict Test Data
                print('Test Data Count =', DataSet_x.shape[0])
                Prdict_y = sess.run(y_predict, feed_dict={ x_input: DataSet_x })
        
                sess.close()

                return Prdict_y             
        
            # End with
        
        # End with

    # End def
    # -------------------------------------------------------------------------

# End class