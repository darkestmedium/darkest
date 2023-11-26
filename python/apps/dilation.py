import cv2
import numpy as np




# Main
if __name__ == '__main__':

  im = np.zeros((10,10), dtype='uint8')

  im[0,1] = 1
  im[-1,0]= 1
  im[-2,-1]=1
  im[2,2] = 1
  im[5:8,5:8] = 1

  print(im)

  element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
  print(element)

  ksize = element.shape[0]
  height, width = im.shape[:2]

  dilatedEllipseKernel = cv2.dilate(im, element)
  print(dilatedEllipseKernel)


  border = ksize//2
  paddedIm = np.zeros((height + border*2, width + border*2))
  paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value = 0)
  paddedDilatedIm = paddedIm.copy()

  # Create a VideoWriter object
  # Use frame size as 50x50
  ###
  ### YOUR CODE HERE
  ###
  frame_size = (50, 50)
  video = cv2.VideoWriter("dilationScratch.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 10, frame_size)
  for h_i in range(border, height+border):
    for w_i in range(border, width+border):
      if im[h_i-border,w_i-border]:
        ###
        ### YOUR CODE HERE
        ###
        paddedDilatedIm[ h_i - border : (h_i + border)+1, w_i - border : (w_i + border)+1] = cv2.bitwise_or(paddedDilatedIm[ h_i - border : (h_i + border)+1, w_i - border : (w_i + border)+1],element)
        ###
        ### YOUR CODE HERE
        ###
        # Convert resizedFrame to BGR before writing
        resized_frame = cv2.resize(paddedDilatedIm, frame_size, cv2.INTER_LINEAR)
        resized_frame=resized_frame*255
        ###
        ### YOUR CODE HERE
        ###
        resized_frame = cv2.cvtColor(resized_frame.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        video.write(resized_frame)
        cv2.waitKey(25)

  # Release the VideoWriter object
  ###
  ### YOUR CODE HERE
  ###
  video.release()


  dilatedImage = paddedDilatedIm[border:border+height,border:border+width]
  plt.imshow(dilatedImage)




# border = ksize//2
# paddedIm = np.zeros((height + border*2, width + border*2))
# paddedIm = cv2.copyMakeBorder(im, border, border, border, border, cv2.BORDER_CONSTANT, value = 1)
# paddedErodedIm = paddedIm.copy()
# paddedErodedIm2= paddedIm.copy()

# frame_size = (50, 50)
# video2 = cv2.VideoWriter("erosionScratch.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 10, frame_size)
# roi=0
# temp=0
# for h_i in range(border, height+border):
#     for w_i in range(border,width+border):
#         if im[h_i-border,w_i-border]:
#             roi=paddedErodedIm2[h_i-border  : (h_i + border)+1, w_i - border : (w_i + border)+1] 
#             temp = cv2.bitwise_or(roi,element)
#             paddedErodedIm[h_i,w_i]=np.min(temp)
            

#             resized_frame2=cv2.resize(paddedErodedIm,(50,50))
#             resized_frame2=resized_frame2*255
#             resized_frame2=cv2.cvtColor(resized_frame2,cv2.COLOR_BAYER_GB2BGR)
            
#             video2.write(resized_frame2)
#             cv2.waitKey(25)


# video.release()