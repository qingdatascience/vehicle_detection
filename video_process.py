import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import pickle
import imageio
imageio.plugins.ffmpeg.download()
import numpy as np
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from lesson_functions import *
from scipy.ndimage.measurements import label


np.random.seed(2)

dist_pickle = pickle.load( open("svm_dic.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]


class vehicle():
    def __init__(self,svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, filter_factor = 6):
        # store all the boxes found
        self.recentbox = []   
        # store all the vehicle heatmap
        self.recentvehicles =[]
        # 
        self.svc = svc
        # 
        self.X_scaler = X_scaler 
        #
        self.orient = orient   
        #
        self.pix_per_cell = pix_per_cell  
        #
        self.cell_per_block = cell_per_block 
        #
        self.spatial_size = spatial_size

        self.hist_bins = hist_bins
        # 
        self.filter_factor = filter_factor

 
    def find_cars(self,img, ystart, ystop, scale, window=64, cells_per_step =2):   
        
        
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
        nxblocks = (ch1.shape[1] // self.pix_per_cell)-1
        nyblocks = (ch1.shape[0] // self.pix_per_cell)-1 
        nfeat_per_block = self.orient*self.cell_per_block**2
        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = window
        nblocks_per_window = (window // self.pix_per_cell)-1 
        cells_per_step = cells_per_step  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step
        
        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, self.orient, self.pix_per_cell, self.cell_per_block, feature_vec=False)
        
        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb*cells_per_step
                xpos = xb*cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos*self.pix_per_cell
                ytop = ypos*self.pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
              
                # Get color features
                spatial_features = bin_spatial(subimg, size=self.spatial_size)
                hist_features = color_hist(subimg, nbins=self.hist_bins)

                # Scale features and make a prediction
                test_features = self.X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
                #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
                test_prediction = self.svc.predict(test_features)
                
                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    
                    ytop_draw = np.int(ytop*scale)
                    
                    win_draw = np.int(window*scale)
                    
                    self.recentbox.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    

    def car_in_scales(self,img): 
        self.recentbox = []
        ystart = 400
        ystop = 432
        scale = 0.5
        small = self.find_cars(img, ystart, ystop, scale, cells_per_step =4)

        ystart = 400
        ystop = 560
        scale = 1
        medium = self.find_cars(img, ystart, ystop, scale)

        ystart = 400
        ystop = 592
        scale = 1.5
        big = self.find_cars(img, ystart, ystop, scale)

        ystart = 464
        ystop = 656
        scale = 3
        huge = car.find_cars(img, ystart, ystop, scale )
        
    def add_heat_(self,heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap# Iterate through list of bboxes
        
    def apply_threshold_(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        #heatmap[heatmap > threshold] = 255
        # Return thresholded map
        return heatmap


    def heatmap_(self,image):
        # find all the boxes
        self.car_in_scales(image)

        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        # Add heat to each box in box list
        heat = self.add_heat_(heat,self.recentbox)
            
        # Apply threshold to help remove false positives
        heat = self.apply_threshold_(heat,1)

        heatmap = np.clip(heat,0, 255)

        # Find final boxes from heatmap using label function
        self.recentvehicles.append(heatmap)
        #if len(self.recentvehicles) == 2:
         #   plt.imshow(self.recentvehicles[0])
          #  plt.show()
         #   print (self.recentvehicles[0].shape)
          

        return np.clip(np.sum(self.recentvehicles[-self.filter_factor:], axis = 0),0,255).astype(np.int)



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
    heatmap[heatmap > threshold] = 255
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img):
    heatmap = car.heatmap_(img)
        # Apply threshold to help remove false positives
    heat = apply_threshold(heatmap,2)

    # Visualize the heatmap when displaying    
    #heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heat)

    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 5)
    # Return the image
    return img

def heatmap(image):
    # Read in image similar to one shown above 
    box_list = findCar(image)


    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

 



if __name__ == '__main__':
    from sys import argv
    output_video = argv[2]
    input_video = argv[1]

    clip1 = VideoFileClip(input_video)

    car = vehicle(svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    video_clip = clip1.fl_image(draw_labeled_bboxes)

    video_clip.write_videofile(output_video, audio=False)


