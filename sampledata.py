import ezdatasets as ezd
import numpy as np

# This program can be used to get a sample of real images from a dataset
X = ezd.GetTrainingData("car")
ezd.sample(X,5,5)
