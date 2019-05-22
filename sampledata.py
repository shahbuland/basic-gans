import ezdatasets as ezd
import numpy as np

X = ezd.GetTrainingData("cars")
ezd.sample(X,5,5)
