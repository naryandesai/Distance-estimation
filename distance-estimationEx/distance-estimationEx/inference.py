import pandas as pd
import argparse
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json

argparser = argparse.ArgumentParser(description='Get predictions of test set')
argparser.add_argument('-m', '--modelname',
                       help='model name (.json)')
argparser.add_argument('-w', '--weights',
                       help='weights filename (.h5)')

args = argparser.parse_args()

# parse arguments
MODEL = args.modelname
WEIGHTS = args.weights

def main():
    print('reading data.')
    df_test = pd.read_csv('train.csv')
    sel = df_test[['isperson','iscycle','iscar','istruck', 'relw', 'relh', 'cx', 'cy', 'yoff']]
    X_test = sel.values

    # load json and create model
    print('loading model.')
    fn = 'generated_files/{}.json'.format(MODEL)
    with open(fn, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json( loaded_model_json )

    # load weights into new model
    loaded_model.load_weights("generated_files/{}.h5".format(WEIGHTS))
    print("Loaded model from disk")

    # evaluate loaded model on test data
    print('evaluating data.')
    loaded_model.compile(loss='mean_squared_error', optimizer='adam')
    y_pred = loaded_model.predict(X_test)

    # save predictions
    print('writing results.')
    df_result = df_test
    df_result['dist_pred'] = -100000

    for idx, row in df_result.iterrows():
        pred = y_pred[idx] * 100.0
        truth = df_result.at[idx, 'dist'] * 100.0
        # print(f'truth: {truth}. pred: {pred}')
        df_result.at[idx, 'dist_norm'] = float(truth)
        df_result.at[idx, 'dist_pred'] = float(pred)

    df_result.to_csv('data/predictions.csv', index=False)


if __name__ == '__main__':
    main()


# python inference.py -m model@1635291358 -w model@1635291358

# generated_files\model@1634588193.h5