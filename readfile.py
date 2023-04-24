import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField
from scipy.ndimage import zoom

def minmax_norm(df):
    result = (df - df.min()) / (df.max() - df.min())
    #result = result * 2 - 1  # Normalize between -1 and 1
    return result

i = 0
data = np.zeros([100000, 1])
for i in range(1):
    data[:, i:i+1] = minmax_norm(pd.read_csv('./data/Capture ' + str(i+1) + '.csv', sep=';'))

# (qty images x Columns x Rows)
image = np.zeros((50, 50, 40))
i, j, l, s = 0, 0, 0, 0
for i in range(1):
    for j in range(50):
        # Create Gramian angular field image using the 'difference' method
        gaf = GramianAngularField(method='difference')
        gaf_image = gaf.fit_transform(data[j:j+2000, i:i+1].reshape(1, -1))
        
        # Resize the Gramian angular field image to the target size using zoom with bilinear interpolation
        resized_image = zoom(gaf_image[0], (50/len(gaf_image[0]), 40/len(gaf_image[0])), order=1)
        
        # Assign the resized image to the appropriate slice of the output image array
        image[l,:,:] = resized_image
        
        l += 1

for fold in range(1):
    for k in range(50):
        plt.imsave('./data/'+'/img_' + str(s+1) + '.jpeg', image[s,:,:])
        s += 1
