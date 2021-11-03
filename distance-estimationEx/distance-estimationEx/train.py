from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import pandas as pd
import numpy as np
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import tensorflow_model_optimization as tfmot
import tensorflow as tf

def main():
	# ----------- import data and scaling ----------- #
	df_data = pd.read_csv('train.csv')
	row_count = df_data.shape[0]
	train_rows = (int) (row_count * 0.9)


	df_train = df_data.iloc[:train_rows,:]
	df_test = df_data.iloc[train_rows + 1:,:]

	print(f"there are {train_rows} rows of training data and {row_count-train_rows} rows of test data.")

	# X_train = df_train[['isperson','iscycle','iscar','istruck','xmin', 'ymin', 'xmax', 'ymax']].values
	# y_train = np.clip(df_train[['zloc']].values, 0.0, 1.0)
	# y_train = df_train[['zloc']].values

	# X_test = df_test[['isperson','iscycle','iscar','istruck','xmin', 'ymin', 'xmax', 'ymax']].values
	# y_test = np.clip(df_test[['zloc']].values, 0.0, 1.0)
	# y_test = df_test[['zloc']].values



	X_train = df_train[['isperson','iscycle','iscar','istruck','relw', 'relh', 'cx', 'cy', 'yoff']].values
	y_train = df_train[['dist']].values

	X_test = df_test[['isperson','iscycle','iscar','istruck','relw', 'relh', 'cx', 'cy', 'yoff']].values
	y_test = df_test[['dist']].values



	model = Sequential()

	# ----------- create model ----------- #

	model.add(Dense(8, input_dim=9, kernel_initializer='normal', activation='relu', name='x'))
	model.add(Dense(12, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', name='output'))

	

	# model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='relu'))
	# model.add(Dropout(0.3))
	# model.add(Dense(12, kernel_initializer='normal', activation='relu'))
	# model.add(Dropout(0.5))
	# model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	# model.add(Dropout(0.3))
	# model.add(Dense(1, kernel_initializer='normal'))

	# q_model = tfmot.quantization.keras.quantize_model(model)
	q_model = model
	
	q_model.compile(loss='mean_squared_error', optimizer='adam')

	# ----------- define callbacks ----------- #
	earlyStopping = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='min', restore_best_weights=True)
	
	reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=50,
									   verbose=1, epsilon=1e-4, mode='min')

	# modelname = "model@{}".format(int(time.time()))
	modelname = "distanceestimator"

	tensorboard = TensorBoard(log_dir="logs/{}".format(modelname))

	# ----------- start training ----------- #
	history = q_model.fit(X_train, y_train,
	 							 validation_split=0.2, epochs=20000, batch_size=2048,
	 							 callbacks=[tensorboard, earlyStopping, reduce_lr_loss], verbose=1)

	

	# ----------- save model and weights ----------- #
	model_json = q_model.to_json()
	with open("generated_files/{}.json".format(modelname), "w") as json_file:
	    json_file.write(model_json)
	q_model.save_weights("generated_files/{}.h5".format(modelname))

	q_model.save("generated_files/distanceestimator")

	model_path = "keras/distanceestimator.h5"
	tf.keras.models.save_model(q_model,model_path)

	print("Saved model to disk")

	

	print("Evaluate on test data")

	results = q_model.evaluate(X_test, y_test, batch_size=64)
	print(results)

if __name__ == '__main__':
	main()
