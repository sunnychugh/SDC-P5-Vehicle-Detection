

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import glob
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import math
from scipy.ndimage.measurements import label
import pickle

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split





#################################################################################
####    small dataset images- jpeg
#################################################################################

# Read in our vehicles and non-vehicles from smallset
images=[]
images = glob.glob('dataset/non-vehicles_smallset/notcars1/*.jpeg')
images.extend(glob.glob('dataset/non-vehicles_smallset/notcars2/*.jpeg'))
images.extend(glob.glob('dataset/non-vehicles_smallset/notcars3/*.jpeg'))
images.extend(glob.glob('dataset/vehicles_smallset/cars1/*.jpeg'))
images.extend(glob.glob('dataset/vehicles_smallset/cars2/*.jpeg'))
images.extend(glob.glob('dataset/vehicles_smallset/cars3/*.jpeg'))


#print(images)
cars = []
notcars = []

for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)

print()
print('total cars images- ', len(cars))
print('total notcars images- ',len(notcars))
print(cars[0])
print(notcars[0])

# # will use these to implement better way of hog transform later on
cars1=cars
notcars1=notcars
print('total cars1 images- ', len(cars1))
print('total notcars1 images- ',len(notcars1))





# demo car and notcar image
demoCarImage = mpimg.imread(cars[0])
demoNotCarImage = mpimg.imread(notcars[0])
print(demoCarImage.shape)
print(demoNotCarImage.shape)
plt.figure(figsize=(8,5))
plt.subplot(1,2,1)
plt.imshow(demoCarImage);
plt.title('demo (first) car image')
plt.subplot(1,2,2)
plt.imshow(demoNotCarImage);
plt.title('demo (first) notcar image')
plt.show()

# random car and notcar image
print(np.random.randint(0, len(cars)))
randomCarImage = mpimg.imread(cars[np.random.randint(0, len(cars))])
randomNotCarImage = mpimg.imread(notcars[np.random.randint(0, len(notcars))])
plt.figure(figsize=(8,5))
plt.subplot(1,2,1)
plt.imshow(randomCarImage);
plt.title('random car image')
plt.subplot(1,2,2)
plt.imshow(randomNotCarImage);
plt.title('random notcar image')
plt.show()



# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  visualise=vis, feature_vector=feature_vec, transform_sqrt=True)
        #print('vis=true, feature & image')
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       visualise=vis, feature_vector=feature_vec, transform_sqrt=True)
        #print('vis=false, features only')
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        vis=False):
    # Create a list to append feature vectors to
    features = []
    hog_images=[]
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            hog_image = []
            #i=0
            for channel in range(feature_image.shape[2]):
#                hog_features.append(get_hog_features(feature_image[:,:,channel],
#                                    orient, pix_per_cell, cell_per_block,
#                                    vis, feature_vec=True))
#                print('ALL true')
#                hog_features = np.ravel(hog_features)

                #print('ALL true')
                hog_features1, hog_image1 = (get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis, feature_vec=False))
                #plt.imshow(hog_features1);plt.show()
                hog_features1 = np.ravel(hog_features1)
                hog_features.append(hog_features1)
                hog_image.append(hog_image1)

        else:
            hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis, feature_vec=True)
            #print('ALL false')
        # Append the new feature vector to the features list
        features.append(hog_features)
        hog_images.append(hog_image)
    # Return list of feature vectors
    return features, hog_images


# Reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time

sample_size = 10
cars = cars[0:sample_size]
notcars = notcars[0:sample_size]


### TODO: Tweak these parameters and see how the results change.
#colorspace = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
colorspace = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
# tells about which channel of image you want to consider
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
#hog_channel = 0 # Can be 0, 1, 2, or "ALL"
# generate the hog_image as well
vis=True

t=time.time()
car_features, car_hog_images = extract_features(cars, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel, vis=vis)
print()
notcar_features, notcar_hog_images = extract_features(notcars, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel, vis=vis)


t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')


# #  cars images, HOG, & Features
# # because for single channel code has to be written in different way
if(hog_channel=='ALL'):
#    print()
#    print(len(car_features),len(car_hog_images))
#    print(len(notcar_features),len(notcar_hog_images))
#    print(car_features[0])
#    print(car_features[0][0])
    plt.figure(figsize=(8,5))
    plt.subplot(1,3,1)
    plt.imshow(car_hog_images[0][2]);
    plt.title('demo (first) car image')
    plt.subplot(1,3,2)
    plt.imshow(car_hog_images[0][0]);
    plt.title('demo (first) notcar image')
    plt.subplot(1,3,3)
    plt.imshow(mpimg.imread(cars[0]));
    plt.title('demo (first) notcar image')
    plt.show()
    print( mpimg.imread(cars[0]).shape[0])
    print(car_hog_images[0][0].shape)
    print(car_features[0][0].shape)
    #widthOfCarImage=mpimg.imread(cars[0]).shape[0]

    #widthOfCarImage=int(math.sqrt(car_features[0][0].shape[0]))
    #print(widthOfCarImage)
    #car_features[0][0]=np.reshape(car_features[0][0], (-1, widthOfCarImage))
    #print(car_features[0][0].shape)
    #plt.imshow(car_features[0][0], cmap='gray');plt.show()



# # cars and notcars -->> images, HOG, & Features
# # because for single channel code has to be written in different way
if(hog_channel=='ALL'):
    indexCars=np.random.randint(0, len(cars))
    indexCars=0
    for channelNo in range(3):
        carImageAllChannels=mpimg.imread(cars[indexCars])
        carImagePerChannels=carImageAllChannels[:,:,channelNo]
        widthOfCarImage=int(math.sqrt(car_features[0][0].shape[0]))
    #    print('here')
    #    print(widthOfCarImage)
        carFeaturesPerChannel = np.reshape(car_features[indexCars][channelNo], (-1, widthOfCarImage))
        #print(carFeaturesPerChannel.shape)
        carHogPerChannel = car_hog_images[indexCars][channelNo]
        plt.figure(figsize=(10,5))
        plt.subplot(1,4,1)
        plt.imshow(carImageAllChannels);
        plt.title('cars[%s]'%(indexCars))
        plt.subplot(1,4,2)
        plt.imshow(carImagePerChannels, cmap='gray');
        plt.title('cars[%s] CH-%s'%(indexCars, channelNo))
        plt.subplot(1,4,3)
        plt.imshow(carHogPerChannel, cmap='gray');
        plt.title('cars[%s] CH-%s HOG'%(indexCars, channelNo))
        plt.subplot(1,4,4)
        plt.imshow(carFeaturesPerChannel, cmap='gray');
        plt.title('cars[%s] CH-%s Features'%(indexCars, channelNo))
        plt.show()

    # # because for single channel code has to be written in different way
    indexNotCars=np.random.randint(0, len(notcars))
    indexNotCars=0
    for channelNo in range(3):
        notcarImageAllChannels=mpimg.imread(notcars[indexNotCars])
        notcarImagePerChannels=notcarImageAllChannels[:,:,channelNo]
        widthOfNotCarImage=int(math.sqrt(notcar_features[0][0].shape[0]))
    #    print('here')
    #    print(widthOfCarImage)
        notcarFeaturesPerChannel = np.reshape(notcar_features[indexNotCars][channelNo], (-1, widthOfNotCarImage))
        #print(carFeaturesPerChannel.shape)
        notcarHogPerChannel = notcar_hog_images[indexNotCars][channelNo]
        plt.figure(figsize=(10,5))
        plt.subplot(1,4,1)
        plt.imshow(notcarImageAllChannels);
        plt.title('notcars[%s]'%(indexNotCars))
        plt.subplot(1,4,2)
        plt.imshow(notcarImagePerChannels, cmap='gray');
        plt.title('notcars[%s] CH-%s'%(indexNotCars, channelNo))
        plt.subplot(1,4,3)
        plt.imshow(notcarHogPerChannel, cmap='gray');
        plt.title('notcars[%s] CH-%s HOG'%(indexNotCars, channelNo))
        plt.subplot(1,4,4)
        plt.imshow(notcarFeaturesPerChannel, cmap='gray');
        plt.title('notcars[%s] CH-%s Features'%(indexNotCars, channelNo))
        plt.show()




#print(len(car_features))
#print(np.asarray(car_features[0]).shape)
#print(np.asarray(car_features[0]))
#print(np.asarray(car_features[1]))
#print(np.asarray(car_features[2]))

car_features_now=[]
notcar_features_now=[]
for j in range(len(car_features)):
    # dependng upon which & how many channel you want to append
    #car_features_now.append(car_features[j][0])
    a=[]
    #print()
    a.extend(car_features[j][0])
    #print(np.array(a))
    a.extend(car_features[j][1])
    #print(np.array(a))
    a.extend(car_features[j][2])
    #print(np.array(a))
    car_features_now.append(a)
    #print(np.array(car_features_now))

car_featuresNP = np.array(car_features_now)
#print(np.array(car_features_now))
#print('here')
#print(np.array(car_features_now).shape)
#
#print(len(notcar_features))
#print(np.asarray(notcar_features[0]).shape)
#print(np.asarray(notcar_features[0]))
#print(np.asarray(notcar_features[1]))
#print(np.asarray(notcar_features[2]))
for j in range(len(notcar_features)):
    # dependng upon which channel you want to append
    #notcar_features_now.append(notcar_features[j][0])
    a=[]
    a.extend(notcar_features[j][0])
    #print(np.array(a))
    a.extend(notcar_features[j][1])
    #print(np.array(a))
    a.extend(notcar_features[j][2])
    #print(np.array(a))
    notcar_features_now.append(a)
    #print(np.array(notcar_features_now))


notcar_featuresNP = np.array(notcar_features_now)
#print(np.array(car_features_now))
#print('here')
#print(np.array(car_features_now).shape)

# #Create an array stack of feature vectors
X = np.vstack((car_features_now, notcar_features_now)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features_now)), np.zeros(len(notcar_features_now))))
#print()
#print(X)
#print(scaled_X)
#print(y)


###############   can also shuffle the data    ##############


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)


# Use a linear SVC (support vector classifier)
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
# Train the SVC
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
## check the accuracy of your classifier on the test dataset
print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
#print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 5
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')










###############################################################################################
##   better code here for hog and color transforms
#######################################################################################





def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


def get_hog_features1(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
    '''Calculate binned color features'''
    #return cv2.resize(img, size).ravel()

def color_hist(img, nbins=32,  bins_range=(0, 256)):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Making the features arrangement to train the model
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features1(imgs, cspace='RGB', orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        vis=False, spatial_size=(32, 32),hist_bins=32,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
#            print('extract_spatial_features:',spatial_features)
#            print('max(extract_spatial_features):',max(spatial_features))
#            print('min(extract_spatial_features):',min(spatial_features))

        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
#            print('extract_hist_features:',hist_features)
#            print('max(extract_hist_features):',max(hist_features))
#            print('min(extract_hist_features):',min(hist_features))
            hist_features=((np.array(hist_features))/(max(hist_features))).astype('float32')
#            print('extract_hist_features:',hist_features)
#            print('max(extract_hist_features):',max(hist_features))
#            print('min(extract_hist_features):',min(hist_features))

            file_features.append(hist_features)

        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features1(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features1(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
#            print('extract_hog_features:',hog_features)
#            print('max(extract_hog_features):',max(hog_features))
#            print('min(extract_hog_features):',min(hog_features))
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features







# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

    draw_img = np.copy(img)
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features1(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features1(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features1(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    windows=[]
    for xb in range(nxsteps):
        #cvprint()
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

           # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            hist_features=((np.array(hist_features))/(max(hist_features))).astype('float32')

#            print()
#            print('spatf- ', spatial_features)
#            print('histf- ', hist_features)
#            print('max(spatf)-', max(spatial_features))
#            print('max(histf)-', max(hist_features))
#            print('max(hogf)-', max(hog_features))
#            print('histf- ', hist_features)
#            print('hogf- ', hog_features)

#            # Scale features and make a prediction
#            #test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
#            test_features = X_scaler.transform(np.hstack((hog_features)).reshape(1, -1))
#            #test_features = X_scaler.transform(np.hstack((spatial_features, hog_features)).reshape(1, -1))
#            test_prediction = svc.predict(test_features)
#            print('old test features length-', len(test_features[0]))


            # Scale features and make a prediction
            final_features=[]
            if (spatial_feat==True):
                final_features=np.hstack((spatial_features))
            if(hist_feat==True):
                final_features=np.hstack((final_features, hist_features))
            if(hog_feat==True):
                final_features=np.hstack((final_features, hog_features))
            test_features = X_scaler.transform(final_features.reshape(1,-1))
            test_prediction = svc.predict(test_features)
            #print('new test features length- ', len(test_features[0]))


            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

    return draw_img, windows




def drawRectanglesOnFrame(image):
    # #Create an empty list to receive positive detection windows
    on_windows = []
    imageCopy=np.copy(image)
    carColor4Scale=((0,255,0), (0,0,255), (255,0,0), (255,0,255), (255,0,255))
    icarColor4Scale=0
    # worked
    #for scale in np.arange(1.5,1.0,-0.25):
    #for scale in np.arange(1.5,0.75,-0.5):
    #for scale in [1.5, 1.60, 1.40]:
    for scale in [1.5]:
        currentcarColor4Scale=carColor4Scale[icarColor4Scale]
        print('currentcarColor4Scale:', currentcarColor4Scale)
        print('icarColor4Scale:', icarColor4Scale)
        print()
        print('scale:',scale)
        out_img, windows = find_cars(image, ystart, ystop, scale, svc1, X1_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        #plt.imshow(out_img);plt.show()
        #print(windows)
        for window in windows:
            cv2.rectangle(image,(window[0]),(window[1]),(carColor4Scale[icarColor4Scale]),6)
        #print('scale: ',icarColor4Scale)
        #plt.imshow(image);plt.show()
        icarColor4Scale+=1
        on_windows.extend(windows)

    #print(on_windows)
    # on_windows will have all the detected windows
    for window in on_windows:
        #print(window)
        cv2.rectangle(imageCopy,(window[0]),(window[1]),(0,0,255),6)
    #plt.imshow(imageCopy);plt.show()
    return on_windows, imageCopy

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):

    carBoxColor=((0,255,0), (0,255,255), (0,0,255), (255,0,0), (255,0,255))

    previousRectangleCenter=((0,0))
    iForCarColorIndex=0
    previousCarBoxColor=carBoxColor[iForCarColorIndex]
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
    #if (car_number<4):
        # Find pixels with each car_number label value
        print('carNumber- ', car_number)
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))


        rectangleCenter = ((bbox[0][0]+bbox[1][0])//2, (bbox[0][1]+bbox[1][1])//2)

        xdiff12=abs(rectangleCenter[0]-previousRectangleCenter[0])
        ydiff12=abs(rectangleCenter[1]-previousRectangleCenter[1])

        currentCarBoxColor=carBoxColor[car_number-1]

        ## to consider nearby pixels
        if(int(math.hypot(xdiff12, ydiff12))<70):
            print('rectangle centres nearby----------------------------------')
            print('int(math.hypot(xdiff12, ydiff12))- ', int(math.hypot(xdiff12, ydiff12)))
            #rectangleCenter=(previousRectangleCenter+rectangleCenter)//2
            rectangleCenter=previousRectangleCenter
            currentCarBoxColor=previousCarBoxColor
        else:
            print('rectangle centres nottttttttt nearby---------------')
            print('int(math.hypot(xdiff12, ydiff12))- ', int(math.hypot(xdiff12, ydiff12)))
            previousRectangleCenter=rectangleCenter
            previousCarBoxColor=currentCarBoxColor

        # draw right size rectangle for different cars, with diff. colors
        if (rectangleCenter[1]<460):
            print('rectangleCenter1- ', (rectangleCenter[0],rectangleCenter[1]))
            if (rectangleCenter[0]>300 and rectangleCenter[0]<900  ):
                xmin_cen=rectangleCenter[0]-50
                xmax_cen=rectangleCenter[0]+50
                ymin_cen=rectangleCenter[1]-40
                ymax_cen=rectangleCenter[1]+40
                bbox = ((xmin_cen, ymin_cen), (xmax_cen, ymax_cen))
            else:
                xmin_cen=rectangleCenter[0]-80
                xmax_cen=rectangleCenter[0]+80
                ymin_cen=rectangleCenter[1]-50
                ymax_cen=rectangleCenter[1]+40
                bbox = ((xmin_cen, ymin_cen), (xmax_cen, ymax_cen))

        elif (rectangleCenter[1]>=460 and rectangleCenter[1]<550):
            print('rectangleCenter2- ', (rectangleCenter[0],rectangleCenter[1]))
            if (rectangleCenter[0]>200 and rectangleCenter[0]<1100  ):
                print('here')
                xmin_cen=rectangleCenter[0]-70
                xmax_cen=rectangleCenter[0]+70
                ymin_cen=rectangleCenter[1]-60
                ymax_cen=rectangleCenter[1]+60
                bbox = ((xmin_cen, np.min(nonzeroy)), (xmax_cen, np.max(nonzeroy)))
            else:
                print('here1')
                xmin_cen=rectangleCenter[0]-100
                xmax_cen=rectangleCenter[0]+100
                ymin_cen=rectangleCenter[1]-90
                ymax_cen=rectangleCenter[1]+90
                bbox = ((xmin_cen, np.min(nonzeroy)), (xmax_cen, np.max(nonzeroy)))

        else:
            print('rectangleCenter3- ', (rectangleCenter[0],rectangleCenter[1]))
            xmin_cen=rectangleCenter[0]-120
            xmax_cen=rectangleCenter[0]+120
            ymin_cen=rectangleCenter[1]-100
            ymax_cen=rectangleCenter[1]+100
            bbox = ((xmin_cen, np.min(nonzeroy)), (xmax_cen, np.max(nonzeroy)))


        # Draw the box on the image
        #cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)
        #cv2.rectangle(img, bbox[0], bbox[1], (carBoxColor[car_number-1]), 6)
        cv2.rectangle(img, bbox[0], bbox[1], (currentCarBoxColor), 6)


    # Return the image
    return img



#import collections
#dequeMaxLength=5
#heatmapDeque = collections.deque(maxlen=dequeMaxLength)
#heatmapDeque.append(np.zeros((720,1280)).astype(np.float64))

heatsList = []
heatClippedThresh_sum = np.zeros((720,1280)).astype(np.float)
heatClipped=[]
heatClippedThresh=[]

import collections
maxLengthDeque=15
heatClippedThresh_Deque=collections.deque(maxlen=maxLengthDeque)
for i in range(heatClippedThresh_Deque.maxlen):
    heatClippedThresh_Deque.append(np.zeros((720,1280)).astype(np.float))
heatClippedThresh_Deque_AllAdd = np.zeros((720,1280)).astype(np.float)

def car_RectBox_HeatMap(img,threshold4NoOfBoxes, maxHeatmapsList=10, video=False, ):

    print('\n===============================================')
    print('new frame')
    global heatClipped
    heatClippedThresh=[]
    copy_image=np.copy(img)
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    #plt.imshow(img);plt.show()
    box_list, imageWithBoxes = drawRectanglesOnFrame(img)
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)
    print("max(heat):", np.amax(heat))
    print("min(heat):", np.amin(heat))
    plt.imshow(heat);plt.colorbar();plt.title('heat');plt.show()


    # Visualize the heatmap when displaying
    ## heatmap only have positive values now, hence easy to apply threshold
    heatClipped = np.clip(heat, 0, 255)
    print("max(heatClipped):", np.amax(heatClipped))
    print("min(heatClipped):", np.amin(heatClipped))
    plt.imshow(heatClipped);plt.colorbar();plt.title('heatClipped');plt.show()

    # Apply threshold to help remove false positives
    heatClippedThresh = apply_threshold(heatClipped, threshold4NoOfBoxes)
    plt.imshow(heatClippedThresh);plt.colorbar();plt.title('heatClippedThresh');plt.show()
    ######check
    # Apply threshold to help remove false positives
    #heat = apply_threshold(heat, threshold4NoOfBoxes)
    print("max(heatClippedThresh):", np.amax(heatClippedThresh))
    print("min(heatClippedThresh):", np.amin(heatClippedThresh))

    #print('heatmap-', heatmap)
    #plt.imshow(heatmap, cmap='hot');plt.colorbar();plt.show();

    #fig1=plt.figure(figsize=(12,12))
    #plt.subplot(121)
    #plt.imshow(heatmap);plt.colorbar();plt.title('single heatmap')
    #plt.show()



    if(video==True):

#        if (len(heatmapsList)<=maxHeatmapsList):
#            box_list, imageWithBoxes = drawRectanglesOnFrame(img)
#            # Add heat to each box in box list
#            heat = add_heat(heat,box_list)
#            print("max(heat):", np.amax(heat))
#            print("min(heat):", np.amin(heat))
#
#            # Apply threshold to help remove false positives
#            heat = apply_threshold(heat, threshold4NoOfBoxes)
#            # Visualize the heatmap when displaying
#            heatmap = np.clip(heat, 0, 255)
 #           heatmap=heat

        global heatClippedThresh_sum
        #global heatmap
        #heatmapDeque.append(heatmap)

        # checkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
        #heatmap_sum = np.add(heatmap_sum, heatmap)
        ########## heatmap_sum is actually heat_sum
        #heatClippedThresh_sum = np.add(heatClippedThresh_sum, heatClippedThresh)

        #global heatClippedThresh_Deque_AllAdd
        heatClippedThresh_Deque_AllAdd = np.zeros((720,1280)).astype(np.float)

        heatClippedThresh_Deque.append(np.array(heatClippedThresh))
        print('len(heatClippedThresh_Deque) before if loop: ',len(heatClippedThresh_Deque))
        print("max(heatClippedThresh_Deque) before AllAdd:", np.amax(heatClippedThresh_Deque))
        print("min(heatClippedThresh_Deque) before AllAdd:", np.amin(heatClippedThresh_Deque))
        for ele in heatClippedThresh_Deque:
            #print(ele.shape)
            #print(heatClippedThresh_Deque_AllAdd.shape)
            #print(np.array(ele))
            heatClippedThresh_Deque_AllAdd=np.add(heatClippedThresh_Deque_AllAdd, np.array(ele))

        print("max(heatClippedThresh_Deque_AllAdd) after AllAdd:", np.amax(heatClippedThresh_Deque_AllAdd))
        print("min(heatClippedThresh_Deque_AllAdd) after AllAdd:", np.amin(heatClippedThresh_Deque_AllAdd))

#        heatClippedThresh_sum += heatClippedThresh
#        heatsList.append(heatClippedThresh)
#        print("max(heatClippedThresh_sum):", np.amax(heatClippedThresh_sum))
#        print("min(heatClippedThresh_sum):", np.amin(heatClippedThresh_sum))
#        print('len(heatsList) before if loop: ', len(heatsList))
#        print('len(maxHeatmapsList)', maxHeatmapsList)

        # subtract off old heat map to keep running sum of last n heatmaps
        #if (len(heatsList)>=maxHeatmapsList):
        if ((np.amax(heatClippedThresh_Deque_AllAdd))>=((2*threshold4NoOfBoxes)+1)):
        #if ((np.amax(heatClippedThresh_sum))>=7):
            print('\nin loop------->>>>>>>>>>>>>>>>>>>>')
            #print('len(heatsList) before pop(0): ', len(heatsList))
            #old_heatmap = heatsList.pop(0)
            #print('len(heatsList) after pop(0): ', len(heatsList))
            #heatmap_sum=np.array(heatmap_sum)
            #old_heatmap=np.array(old_heatmap)
            #heatClippedThresh_sum -= old_heatmap

            ## deque line
            #heatClippedThresh_sum=heatClippedThresh_Deque_AllAdd
            #print("max(heatClippedThresh_sum):", np.amax(heatClippedThresh_sum))
            #print("min(heatClippedThresh_sum):", np.amin(heatClippedThresh_sum))

            print("max(heatClippedThresh_Deque_AllAdd): ", np.amax(heatClippedThresh_Deque_AllAdd))
            print("min(heatClippedThresh_Deque_AllAdd): ", np.amin(heatClippedThresh_Deque_AllAdd))

            ## check this line
            ##either both clips should be up or down
            ## minus values of heatmap are subtracting & creating issues for threshold
            ## i guess no need of this as already clippped
            #heatmap_sum = np.clip(heatmap_sum,0,255)

            #plt.imshow(heatClippedThresh_sum);plt.colorbar();plt.title('heatClippedThresh_sum_before')
            #plt.show();
            plt.imshow(heatClippedThresh_Deque_AllAdd);plt.colorbar();plt.title('heatClippedThresh_Deque_AllAdd--before')
            plt.show();

            #heatmap_sum = apply_threshold(heatmap_sum, (mulFactor4threshold)*threshold4NoOfBoxes)
            #heatmap = apply_threshold(heatmap_sum, (mulFactor4threshold)*threshold4NoOfBoxes)
            heatClippedThresh_Deque_AllAdd = apply_threshold(heatClippedThresh_Deque_AllAdd, (mulFactor4threshold)*(threshold4NoOfBoxes+1))
            plt.imshow(heatClippedThresh_Deque_AllAdd);plt.colorbar();plt.title('heatClippedThresh_Deque_AllAdd--after')
            plt.show();

            ## check this line
            ## minus values of heatmap are subtracting & creating issues for threshold
            #heatmap = np.clip(heatmap,0,255)


            #plt.subplot(122)
            #plt.imshow(heatmap_sum);plt.colorbar();plt.title('heatmap_sum_after')
            #plt.show();
        plt.imshow(heatClippedThresh_Deque_AllAdd);plt.colorbar();plt.title('heatClippedThresh_Deque_AllAdd--after')
        plt.show();
        heatClippedThresh=heatClippedThresh_Deque_AllAdd
        #heatmap=heatmap_sum

#        addDequeValues=[]
#        for i in range(len(heatmapDeque)-1):
#            print()
#            print('i-',i)
#            print('len(heatmapDeque)-', len(heatmapDeque))
#            #print('i+1-',i+1)
#            print(heatmapDeque[i])
#            addDequeValues=np.add(heatmapDeque[i],heatmapDeque[i+1])
#            print('iplusi+1-', )

    # Find final boxes from heatmap using label function
    labels = label(heatClippedThresh)
    print(labels[1], 'cars found')
    #plt.imshow(labels[0], cmap='gray');plt.colorbar();plt.show()
    #print('labels[0]-', labels[0])



    draw_img = draw_labeled_bboxes(copy_image, labels)


    plt.figure(figsize=(8,5))
    plt.subplot(121)
    plt.imshow(imageWithBoxes)
    plt.title('imageWithBoxes')
    plt.subplot(122)
    plt.imshow(draw_img)
    plt.title('final_frame(draw_img)')
    #fig.tight_layout()
    plt.show()

    return imageWithBoxes, draw_img, heatClippedThresh, labels




#################################################################################
####    big dataset images - png
#################################################################################



# Read in our vehicles and non-vehicles
images1 = glob.glob('dataset/non-vehicles/**/*.png', recursive=True)
images1 += glob.glob('dataset/vehicles/**/*.png', recursive=True)
print('len- ', len(images1))
print(images1[0])
print(images1[-1])

# images are divided up into vehicles and non-vehicles
cars1 = []
notcars1 = []

for image in images1:
    if 'non-vehicles' in image:
        #notcars1.append(mpimg.imread(image))
        notcars1.append(image)
    else:
        #cars1.append(mpimg.imread(image))
        cars1.append(image)

print()
print('total cars1 images- ', len(cars1))
print('total notcars1 images- ',len(notcars1))
#print(cars1[0])
#print(notcars1[0])
#plt.imshow(cars1[0]);plt.show()
#plt.imshow(notcars1[0]);plt.show()
#plt.imshow(cars1[0]);plt.show()
#plt.imshow(notcars1[0]);plt.show()
plt.imshow(mpimg.imread(cars1[0]));plt.show()
plt.imshow(mpimg.imread(notcars1[0]));plt.show()


# #Reduce the sample size because HOG features are slow to compute
# #The quiz evaluator times out after 13s of CPU timecentroid = ((bbox[0][0]+bbox[1][0])//2, (bbox[0][1]+bbox[1][1])//2)

#sample_size1 = 50
#cars1 = cars1[0:sample_size1]
#notcars1 = notcars1[0:sample_size1]



### TODO: Tweak these parameters and see how the results change.
# YCrCb not working with all three features (spacial, hist & hog)
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
# tells about which channel of image you want to consider
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
#hog_channel = 0 # Can be 0, 1, 2, or "ALL"
# keep vis=False here, so as not to generate the hog_image
vis=False
spatial_size = (32, 32)  # (32,32)
hist_bins = 16  # 16

spatial_feat=True
hist_feat=True
hog_feat=True
#spatial_feat=False
#hist_feat=False


# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
#image = image.astype(np.float32)/255
##########################################


t=time.time()

car_features1 = extract_features1(cars1, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel, vis=vis, spatial_size=spatial_size,
                        hist_bins=hist_bins, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

notcar_features1 = extract_features1(notcars1, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel, vis=vis, spatial_size=spatial_size,
                        hist_bins=hist_bins, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)


t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract features1 ...')

# #Create an array stack of feature vectors
X1 = np.vstack((car_features1, notcar_features1)).astype(np.float64)
# Fit a per-column scaler
X1_scaler = StandardScaler().fit(X1)
# Apply the scaler to X
scaled_X1 = X1_scaler.transform(X1)

# Define the labels vector
y1 = np.hstack((np.ones(len(car_features1)), np.zeros(len(notcar_features1))))
#print()
#print(X1)
#print(scaled_X1)
#print(y1)


###############   can also shuffle the data    ##############
from sklearn.utils import shuffle
# Shuffle the training data
scaled_X1, y1 = shuffle(scaled_X1, y1)
print('scaled_X1[0]-', scaled_X1[0])
print('max(scaled_X1[0])-', max(scaled_X1[0]))
print('min(scaled_X1[0])-', min(scaled_X1[0]))
print('y1[0]-', y1[0])

# # Training the model here
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X1_train, X1_test, y1_train, y1_test = train_test_split(
    scaled_X1, y1, test_size=0.1, random_state=rand_state)


# Use a linear SVC (support vector classifier)
svc1 = LinearSVC()
# Check the training time for the SVC
t=time.time()
# Train the SVC
svc1.fit(X1_train, y1_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC1 ...')
## check the accuracy of your classifier on the test dataset
print('Test Accuracy of SVC = ', svc1.score(X1_test, y1_test))
#print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t=time.time()
n_predict = 10
print('My SVC1 predicts: ', svc1.predict(X1_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y1_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC1')





# Saving the data in pickle format for later use
import pickle
veh_det_pickle = {}
veh_det_pickle["svc"] = svc1
veh_det_pickle["X_scaler"] = X1_scaler
veh_det_pickle["colorspace"] = colorspace
veh_det_pickle["orient"] = orient
veh_det_pickle["pix_per_cell"] = pix_per_cell
veh_det_pickle["cell_per_block"] = cell_per_block
veh_det_pickle["spatial_size"] = spatial_size
veh_det_pickle["hist_bins"] = hist_bins
veh_det_pickle["spatial_feat"] = spatial_feat
veh_det_pickle["hist_feat"] = hist_feat
veh_det_pickle["hog_feat"] = hog_feat
pickle.dump(veh_det_pickle, open( "saved_veh_det_pickle.p", "wb" ) )




# # Restroing the saved pickle data
veh_pickle = pickle.load( open("saved_veh_det_pickle.p", "rb" ) )
svc1 = veh_pickle["svc"]
X1_scaler = veh_pickle["X_scaler"]
colorspace = veh_pickle["colorspace"]
orient = veh_pickle["orient"]
pix_per_cell = veh_pickle["pix_per_cell"]
cell_per_block = veh_pickle["cell_per_block"]
spatial_size = veh_pickle["spatial_size"]
hist_bins = veh_pickle["hist_bins"]
spatial_feat = veh_pickle["spatial_feat"]
hist_feat = veh_pickle["hist_feat"]
hog_feat = veh_pickle["hog_feat"]


ystart = 400
ystop = 656
#scale = 1.5
## can try 5, 6, 7, 8 also
#threshold4NoOfBoxes = 1
#mulFactor4threshold=8
#maxHeatmapsList = 15

threshold4NoOfBoxes = 1
mulFactor4threshold=7
#maxHeatmapsList = 15


#img = mpimg.imread('test_images/test4.jpg')
#plt.imshow(img);plt.show()
#final_image_with_boxes, final_frame, final_heatmap, final_labels = car_RectBox_HeatMap(img, threshold4NoOfBoxes)
#plt.imshow(final_image_with_boxes);plt.show()
#print(final_labels[1], 'cars found')
#fig = plt.figure(figsize=(8,5))
#plt.subplot(121)
#plt.imshow(final_frame)
#plt.title('Car Positions')
#plt.subplot(122)
#plt.imshow(final_heatmap, cmap='gray')
#plt.title('Heat Map')
##fig.tight_layout()
#plt.show()


### Make a list of test images
imagesPaths = glob.glob('test_images/*.jpg')
print(len(imagesPaths))
import os
for imagePath in sorted(imagesPaths):
    imgTest = mpimg.imread(imagePath)
    #plt.imshow(imgTest);plt.show()
    imageName = os.path.basename(imagePath)
    print(imageName)

    final_image_with_boxes,final_frame, final_heatmap, final_labels = car_RectBox_HeatMap(imgTest, threshold4NoOfBoxes)
    #plt.imshow(final_image_with_boxes);plt.show()
    #saving image
    imageName = os.path.splitext(imageName)[0]
    print(imageName)
    imageSavePath = "output_images/" + imageName + "_output.jpg"
    print(imageSavePath)
    #cv2.imwrite(imageSavePath, final_frame)
    mpimg.imsave(imageSavePath, final_frame)
    print(final_labels[1], 'cars found')
    fig = plt.figure(figsize=(8,5))
    plt.subplot(121)
    plt.imshow(final_frame)
    plt.title('Car Positions- '+ imageName)
    plt.subplot(122)
    plt.imshow(final_heatmap, cmap='hot')
    plt.title('Heat Map')
    #fig.tight_layout()
    plt.show()
    print("========================================================")




# # skipping frames to reduce processing time for videos
iForSkipFrames=0
final_labels=[]
def process_image(image):
    global iForSkipFrames
    global final_labels

    # # Skip video to required frames as first 100 frames are without cars
    if(iForSkipFrames<(75) or iForSkipFrames>1260):
        iForSkipFrames+=1
        return image

    print()
    print('iForSkipFrames- ', iForSkipFrames)
    ## if want to skip frames
    if((iForSkipFrames)%3 == 0):
        print("threshold4NoOfBoxes: ", threshold4NoOfBoxes)
        #print("maxHeatmapsList: ", maxHeatmapsList)
        final_image_with_boxes, final_frame, final_heatmap, final_labels = car_RectBox_HeatMap(image, threshold4NoOfBoxes, maxHeatmapsList, video=True)
    else:
        final_frame = draw_labeled_bboxes(image, final_labels)
        #plt.imshow(final_frame);plt.show()
    #plt.imshow(final_frame);plt.title('final_frame');plt.show()
    iForSkipFrames+=1
    return final_frame



def resetGlobalVariables():
    global heatmapsList
    global heatmap_sum
    global heatmap
    global iForSkipFrames
    global final_labels
    heatmapsList = []
    heatmap_sum = np.zeros((720,1280)).astype(np.float64)
    heatmap=[]
    iForSkipFrames=0
    final_labels=[]


# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


iForSkipFrames=0
resetGlobalVariables()
test_video_output = 'test_video_solution.mp4'
clip1 = VideoFileClip("test_video.mp4")
test_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
test_clip.write_videofile(test_video_output, audio=False)

HTML("""
<video width="960" height="540" controls>
<source src="{0}">
</video>
""".format(test_video_output))


iForSkipFrames=0
resetGlobalVariables()
project_video_output = 'project_video_solution.mp4'
clip1 = VideoFileClip("project_video.mp4")
project_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
project_clip.write_videofile(project_video_output, audio=False)

HTML("""
<video width="960" height="540" controls>
<source src="{0}">
</video>
""".format(project_video_output))