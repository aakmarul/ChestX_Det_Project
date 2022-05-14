import numpy
import tensorflow as tf
#from Cython.Includes.libcpp.list import resize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax, MaxPool2D, Conv2D
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PIL
import PIL.Image
import glob
import tensorflow_datasets as tfds

from numpy import savetxt

from sklearn.model_selection import train_test_split

#regularizers
from tensorflow.keras.layers import Dropout # one of the best regularizers
from tensorflow.keras.regularizers import l1,l2,l1_l2

#optimizers
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from matplotlib import pyplot as plt




