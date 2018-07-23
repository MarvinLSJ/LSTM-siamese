# LSTM-siamese
### How to use:

Main file is **'siamese.py'**, the configuration file is **'siamese-config.yaml'**.  

#### 1. Initialization:

    First run with the original dataset, you should input parameters **'--config siamese-config.yaml'**, while configuring **make_dict** and **data_preprocessing** as **True**, it will make the embedding and preprocess the original dataset, saved them for future usage.  


#### 2. Model:

**Tuning Parameters**:
    
    a. Classifier
    
       fc_dim: classifier fully connected layer size, 
    
    b. Encoder
           
       hidden_size: lstm hidden size
       num_layers: lstm layer 
       bidirectional: bidirectional lstm can get more info
       dropout: avoid overfitting
        
    c. Embedding
           
       embedding_freeze: Set it to false, then the embedding will participate backpropogation. Not so good from my experience, especially small training dataset.
    
    d. Other

**Structure**:
    
    a. Classifier
        
        fc layers, non-linear fc layers(add ReLU)
    
    b. Encoder 
        
        Features generating method, current method is (v1, v2, abs(v1-v2), v1*v2), more features with different vector distance measurement?


​    
#### 3. Training

    a. Optimizer
    
       Default SGD, lots of optimizer in torch.
    
    b. Learning rate
    
       It should be small enough to avoid oscillation. Furthur exploration can be dynamic lr clipping.
        
    c. Other