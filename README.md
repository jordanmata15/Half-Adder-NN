# Half-Adder-NN

# Description
Simple neural network built to model the behavior of a half adder machine. To make it less trivial, we allow the inputs to have some noise (+-0.2). 
For example, the typical input would be [x1, x2] where xi = {0,1}. However, in this case, we allow xi to be a float between {-0.2,0.2} or {0.8, 1.2}. 

If the value is in the first set, we say that it's a 0. if it's in the second set, we say that it's a 1.

# Requirements
- keras
- numpy
- pandas
- tensorflow

# Running
1. Navigate to `src` directory
2. Update `Model.py` to build the model or leave it as is to load the existing model
3. Run the Model.py
4. Inspect the stdout for predicted percentage
5. (Optional) Plot the training data (after building the model)
    - Run DataPlotter.py
    - inspect the plot in `plots` directory 
