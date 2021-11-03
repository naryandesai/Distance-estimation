import argparse
import os
import random
import sys
import traceback
from os.path import dirname, abspath, join
from cv2 import cv2
import numpy as np
import pandas as pd


cuda_path = os.path.join(os.environ.get('CUDA_PATH'), 'bin')
print(f'loading cuda from {cuda_path}')
os.add_dll_directory(cuda_path)
os.environ["TFHUB_CACHE_DIR"] = "/tensorflow/staging"

import tensorflow as tf
print(tf.__version__)

REP_LIMIT = 1000

def representative_dataset_gen(): 
    global REP_LIMIT
    df_data = pd.read_csv('train.csv')
    X_train = df_data[['isperson','iscycle','iscar','istruck','relw', 'relh', 'cx', 'cy', 'yoff']].values
    X_train = np.random.shuffle(X_train)
    X_train = X_train[:REP_LIMIT, :]
    for row in X_train:
        print(row)
        yield row
    

def convert(modelfile, modelout,quantize):
    try:
        km = tf.keras.models.load_model(modelfile)
        converter = tf.lite.TFLiteConverter.from_keras_model(km)
        converter.experimental_new_converter = True

        # converter = tf.lite.TFLiteConverter.from_saved_model(modelfile)
        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.allow_custom_ops = True
        # converter.experimental_new_converter = True

        # if quantize:
        #    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.TFLITE_BUILTINS]
        #    converter.inference_input_type = tf.uint8  
        #    converter.inference_output_type = tf.uint8 
        #    converter.representative_dataset = representative_dataset_gen
        # else:
        #    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        #    converter.inference_input_type = tf.float32  
        #    converter.inference_output_type = tf.float32

        tflite_model = converter.convert()
        open(modelout, "wb").write(tflite_model)
        print(f'\nquantized? {quantize} model saved as {modelout}')

    except Exception as e:
        print('\nTFLite export failure: %s' % e)
        traceback.print_exc(file=sys.stdout)

def main(opt):
    global REP_LIMIT
    REP_LIMIT = opt.repbatch
    convert(opt.model, opt.output, opt.quant)


def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model', type=str, default='generated_files/distanceestimator') 
    parser.add_argument('--model', type=str, default='keras/distanceestimator.h5')
    parser.add_argument('--output', type=str, default='keras/distanceestimator.tflite')
    parser.add_argument('--repbatch', type=int, default=1000)
    parser.add_argument('--quant', dest='quant', action='store_true')
    parser.add_argument('--noquant', dest='quant', action='store_false')
    parser.set_defaults(quant = False)
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)