# transfer_learning

<pre>
Dataset cats_vs_dogs downloaded and prepared to /root/tensorflow_datasets/cats_vs_dogs/4.0.0. Subsequent calls will reuse this data.
Number of training samples: 9305
Number of validation samples: 2326
Number of test samples: 2326
Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5
83689472/83683744 [==============================] - 0s 0us/step
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 150, 150, 3)]     0         
_________________________________________________________________
sequential (Sequential)      (None, 150, 150, 3)       0         
_________________________________________________________________
normalization (Normalization (None, 150, 150, 3)       7         
_________________________________________________________________
xception (Functional)        (None, 5, 5, 2048)        20861480  
_________________________________________________________________
global_average_pooling2d (Gl (None, 2048)              0         
_________________________________________________________________
dropout (Dropout)            (None, 2048)              0         
_________________________________________________________________
dense (Dense)                (None, 1)                 2049      
=================================================================
Total params: 20,863,536
Trainable params: 2,049
Non-trainable params: 20,861,487
_________________________________________________________________
Epoch 1/20
291/291 [==============================] - 62s 98ms/step - loss: 0.2424 - binary_accuracy: 0.8846 - val_loss: 0.0871 - val_binary_accuracy: 0.9682
Epoch 2/20
291/291 [==============================] - 26s 90ms/step - loss: 0.1221 - binary_accuracy: 0.9482 - val_loss: 0.0781 - val_binary_accuracy: 0.9682
Epoch 3/20
291/291 [==============================] - 26s 90ms/step - loss: 0.1117 - binary_accuracy: 0.9502 - val_loss: 0.0814 - val_binary_accuracy: 0.9682
Epoch 4/20
291/291 [==============================] - 26s 90ms/step - loss: 0.1033 - binary_accuracy: 0.9573 - val_loss: 0.0726 - val_binary_accuracy: 0.9712
Epoch 5/20
291/291 [==============================] - 26s 90ms/step - loss: 0.1100 - binary_accuracy: 0.9538 - val_loss: 0.0750 - val_binary_accuracy: 0.9699
Epoch 6/20
291/291 [==============================] - 26s 90ms/step - loss: 0.1063 - binary_accuracy: 0.9543 - val_loss: 0.0725 - val_binary_accuracy: 0.9725
Epoch 7/20
291/291 [==============================] - 27s 91ms/step - loss: 0.1022 - binary_accuracy: 0.9574 - val_loss: 0.0718 - val_binary_accuracy: 0.9716
Epoch 8/20
291/291 [==============================] - 27s 92ms/step - loss: 0.0973 - binary_accuracy: 0.9601 - val_loss: 0.0731 - val_binary_accuracy: 0.9721
Epoch 9/20
291/291 [==============================] - 27s 92ms/step - loss: 0.0962 - binary_accuracy: 0.9612 - val_loss: 0.0708 - val_binary_accuracy: 0.9733
Epoch 10/20
291/291 [==============================] - 27s 92ms/step - loss: 0.0905 - binary_accuracy: 0.9615 - val_loss: 0.0710 - val_binary_accuracy: 0.9716
Epoch 11/20
291/291 [==============================] - 27s 92ms/step - loss: 0.0967 - binary_accuracy: 0.9621 - val_loss: 0.0716 - val_binary_accuracy: 0.9708
Epoch 12/20
291/291 [==============================] - 27s 92ms/step - loss: 0.0959 - binary_accuracy: 0.9634 - val_loss: 0.0728 - val_binary_accuracy: 0.9712
Epoch 13/20
291/291 [==============================] - 27s 91ms/step - loss: 0.0970 - binary_accuracy: 0.9614 - val_loss: 0.0735 - val_binary_accuracy: 0.9699
Epoch 14/20
291/291 [==============================] - 27s 91ms/step - loss: 0.0941 - binary_accuracy: 0.9598 - val_loss: 0.0828 - val_binary_accuracy: 0.9682
Epoch 15/20
291/291 [==============================] - 27s 92ms/step - loss: 0.0956 - binary_accuracy: 0.9642 - val_loss: 0.0772 - val_binary_accuracy: 0.9695
Epoch 16/20
291/291 [==============================] - 27s 92ms/step - loss: 0.0913 - binary_accuracy: 0.9626 - val_loss: 0.0734 - val_binary_accuracy: 0.9725
Epoch 17/20
291/291 [==============================] - 27s 92ms/step - loss: 0.0920 - binary_accuracy: 0.9609 - val_loss: 0.0772 - val_binary_accuracy: 0.9703
Epoch 18/20
291/291 [==============================] - 27s 92ms/step - loss: 0.0903 - binary_accuracy: 0.9618 - val_loss: 0.0719 - val_binary_accuracy: 0.9703
Epoch 19/20
291/291 [==============================] - 27s 93ms/step - loss: 0.0977 - binary_accuracy: 0.9602 - val_loss: 0.0711 - val_binary_accuracy: 0.9733
Epoch 20/20
291/291 [==============================] - 27s 92ms/step - loss: 0.1026 - binary_accuracy: 0.9623 - val_loss: 0.0778 - val_binary_accuracy: 0.9703
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 150, 150, 3)]     0         
_________________________________________________________________
sequential (Sequential)      (None, 150, 150, 3)       0         
_________________________________________________________________
normalization (Normalization (None, 150, 150, 3)       7         
_________________________________________________________________
xception (Functional)        (None, 5, 5, 2048)        20861480  
_________________________________________________________________
global_average_pooling2d (Gl (None, 2048)              0         
_________________________________________________________________
dropout (Dropout)            (None, 2048)              0         
_________________________________________________________________
dense (Dense)                (None, 1)                 2049      
=================================================================
Total params: 20,863,536
Trainable params: 20,809,001
Non-trainable params: 54,535
_________________________________________________________________
Epoch 1/10
291/291 [==============================] - 116s 381ms/step - loss: 0.0879 - binary_accuracy: 0.9648 - val_loss: 0.0527 - val_binary_accuracy: 0.9772
Epoch 2/10
291/291 [==============================] - 109s 375ms/step - loss: 0.0618 - binary_accuracy: 0.9753 - val_loss: 0.0478 - val_binary_accuracy: 0.9802
Epoch 3/10
291/291 [==============================] - 109s 376ms/step - loss: 0.0511 - binary_accuracy: 0.9788 - val_loss: 0.0461 - val_binary_accuracy: 0.9828
Epoch 4/10
291/291 [==============================] - 109s 376ms/step - loss: 0.0368 - binary_accuracy: 0.9837 - val_loss: 0.0450 - val_binary_accuracy: 0.9824
Epoch 5/10
291/291 [==============================] - 109s 375ms/step - loss: 0.0278 - binary_accuracy: 0.9891 - val_loss: 0.0461 - val_binary_accuracy: 0.9832
Epoch 6/10
291/291 [==============================] - 109s 376ms/step - loss: 0.0245 - binary_accuracy: 0.9912 - val_loss: 0.0399 - val_binary_accuracy: 0.9815
Epoch 7/10
291/291 [==============================] - 110s 377ms/step - loss: 0.0180 - binary_accuracy: 0.9937 - val_loss: 0.0448 - val_binary_accuracy: 0.9837
Epoch 8/10
291/291 [==============================] - 109s 376ms/step - loss: 0.0152 - binary_accuracy: 0.9929 - val_loss: 0.0507 - val_binary_accuracy: 0.9824
Epoch 9/10
291/291 [==============================] - 110s 377ms/step - loss: 0.0157 - binary_accuracy: 0.9951 - val_loss: 0.0444 - val_binary_accuracy: 0.9841
Epoch 10/10
291/291 [==============================] - 109s 375ms/step - loss: 0.0131 - binary_accuracy: 0.9949 - val_loss: 0.0524 - val_binary_accuracy: 0.9811
<tensorflow.python.keras.callbacks.History at 0x7f076028a150>
</pre>
