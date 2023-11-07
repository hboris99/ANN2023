# from keras.engine.saving import model_from_json


# Import other libraries
# Import tensorflow
# Load the NPZ dataset
from model import CNNModel

test_model = CNNModel()
data_arr, labels_arr = test_model.load_dataset()
prep_data, prep_labels = test_model.preprocess_data(data_arr, labels_arr)
X_train, X_val, y_train, y_val = test_model.create_data_split(prep_data, prep_labels)
history = test_model.train(X_train, y_train, X_val, y_val, batch_size=128, epochs=25)

