# transfer learning & fine tunning with Xception model

## Xception model's position
Model | Size | Top-1 Accuracy | Top-5 Accuracy | Parameters | Depth
------|------|----------------|----------------|------------|-------
Xception | 88 MB | 0.790 | 0.945 | 22,910,480 | 126
VGG16 | 528 MB | 0.713 | 0.901 | 138,357,544 | 23
VGG19 | 549 MB | 0.713 | 0.900 | 143,667,240 | 26
ResNet50 | 98 MB | 0.749 | 0.921 | 25,636,712 | -
ResNet101 | 171 MB | 0.764 | 0.928 | 44,707,176 | -

## data augumentation
<pre>
# random horizontal flip & rotation within 0.1 angle
data_augmentation = keras.Sequential(
    [
     layers.experimental.preprocessing.RandomFlip("horizontal"),
     layers.experimental.preprocessing.RandomRotation(0.1),
     ]
     )
</pre>

<pre>
# extra examples for reize and rescaling
resize_and_rescale = tf.keras.Sequential([
      layers.experimental.preprocessing.Resizing(180, 180),
      # If instead you wanted [-1,1], you would write -> .Rescaling(1./127.5, offset=-1).
      layers.experimental.preprocessing.Rescaling(1./255) # [0,1]
      ])
</pre>

## transfer learning
<pre>
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

..........

Epoch 20/20
291/291 [==============================] - 27s 92ms/step - loss: 0.1026 - binary_accuracy: 0.9623 - val_loss: 0.0778 - val_binary_accuracy: 0.9703
</pre>

## fine tunning

<pre>
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

..........

Epoch 10/10
291/291 [==============================] - 109s 375ms/step - loss: 0.0131 - binary_accuracy: 0.9949 - val_loss: 0.0524 - val_binary_accuracy: 0.9811
</pre>
