# Built-in imports
import sys
import argparse
import math
from enum import Enum
from typing import overload, final

# Third-party imports
import numpy as np
# import cv2 as cv
import cv2
import matplotlib.pyplot as plt

# Darkest APi imports

winname = "OpenCV Window"
cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
imagePath = "/home/ccpcpp/Dropbox/code/darkest/resources/images/blemish.png"

number_of_candidates = 16
radius = 16




styles = {
  # BGR
  "light": {
    "cola": (255, 255, 255),	# almost white
    "colb": (225, 225, 225),	# light grey
    "colc": (35, 35, 35),			# dark grey
    "cold": (0, 0, 0),				# almost black
    "fontr": ["resources/fonts/intel/IntelOneMono-Regular.ttf", 0],
    "fontb": ["resources/fonts/intel/IntelOneMono-Bold.ttf", 0],
    "fontl": ["resources/fonts/intel/IntelOneMono-Light.ttf", 0],
    "body-01": 12,
    "body-02": 16,
  },
  "dark": {
    "cola": (0, 0, 0),				# almost black
    "colb": (39, 39, 39),			# dark grey
    "colc": (225, 225, 225),	# light grey
    "cold": (245, 245, 245),	# almost white
    "fontr": ["resources/fonts/intel/IntelOneMono-Regular.ttf", 0],
    "fontb": ["resources/fonts/intel/IntelOneMono-Bold.ttf", 0],
    "fontl": ["resources/fonts/intel/IntelOneMono-Light.ttf", 0],
  },

  "cpr": {
    # "cola": (50, 10, 250),	# red
    "cola": (75, 10, 250),	# red
    "colb": (10, 10, 105),	# dark red
    "colc": (235, 225, 10),	# cyan
    # "cold": (245, 245, 150),		# dark blue
    "cold": (65, 5, 5),		# dark blue
    "fontr": ["resources/fonts/intel/IntelOneMono-Regular.ttf", 0],
    "fontb": ["resources/fonts/intel/IntelOneMono-Bold.ttf", 0],
    "fontl": ["resources/fonts/intel/IntelOneMono-Light.ttf", 0],
  },
  "cpc": {
    "cola": (235, 225, 10),	# cyan
    "colb": (65, 5, 5),		# dark blue
    "colc": (50, 10, 250),	# red
    "cold": (5, 5, 105),	# dark red
    "fontr": ["resources/fonts/intel/IntelOneMono-Regular.ttf", 0],
    "fontb": ["resources/fonts/intel/IntelOneMono-Bold.ttf", 0],
    "fontl": ["resources/fonts/intel/IntelOneMono-Light.ttf", 0],
  },
}


class DBox():

  # Constructors
  def __init__(self, left:int, top:int, right:int, bottom:int) -> None:
    self.lefttop = (left, top)
    self.righttop = (right, top)
    self.rightbottom = (right, bottom)
    self.leftbottom = (left, bottom)

    self.left = (left, (top+bottom)//2)
    self.top = ((left+right)//2, top)
    self.right = (right, (top+bottom)/2)
    self.bottom = ((left+right)//2, bottom)

    self.center = ((left+right)//2, (top+bottom)//2)

    self.points_to_array()


  def points_to_array(self):
    self.points = []
    self.points.append(self.lefttop)
    self.points.append(self.righttop)
    self.points.append(self.rightbottom)
    self.points.append(self.leftbottom)

    self.points.append(self.left)
    self.points.append(self.top)
    self.points.append(self.right)
    self.points.append(self.bottom)

    self.points.append(self.center)




class DrawOCVUi():
  """Class for drawing annotations with the opencv framework.

  The origin, (0, 0), is located at the top-left of the image. OpenCV images are zero-indexed,
  where the x-values go left-to-right (column number) and y-values go top-to-bottom (row number).

  Reference:
    cv2.MARKER_CROSS,
    cv2.MARKER_TILTED_CROSS,
    cv2.MARKER_STAR,
    cv2.MARKER_DIAMOND,
    cv2.MARKER_SQUARE,
    cv2.MARKER_TRIANGLE_UP,
    cv2.MARKER_TRIANGLE_DOWN,

  
  TODO:
    Fix clipping of label - out of frame - automatic placement?
    cleanup / optimize text method

  """

  class StyleFlag(Enum):
    IBMPlexMonoLight 	: 0
    IBMPlexMonoDark 	: 1

  # log = logging.getLogger("DrawOCVUi")

  # Frame attributes
  imcv = None
  imsh = None
  imtxt = None
  imout = None

  imheight = None
  imwidth = None
  dnn = None

  bbox = None

  # Style attributes
  style = "light"
  # Edges
  thicka = 1
  thickb = thicka*2
  thickc = thicka*4
  linet = cv2.LINE_AA
  # Font
  ft2b = cv2.freetype.createFreeType2()
  ft2l = cv2.freetype.createFreeType2()
  ft2r = cv2.freetype.createFreeType2()
  heading = 18
  body = 14
  fontt = -1
  # Colors
  cola = styles[style]["cola"]
  colb = styles[style]["colb"]
  colc = styles[style]["colc"]
  cold = styles[style]["cold"]

  class_index = {
    1: 'person',
    2: 'face',
  }
  class_id = 2
  class_name = class_index[class_id]


  # Constructor
  @overload
  def __init__(self, imcv:np.ndarray=None, dnn=None, style:str="cpr") -> None: ...
  def __init__(self, imcv=None, dnn=None, style:str="light") -> None:
    """Overloads on constructors??."""
    if imcv is None: imcv = np.zeros((720, 1280, 3), dtype=np.uint8)

    DrawOCVUi.image(imcv)
    DrawOCVUi.dnn = dnn
    # Set style
    DrawOCVUi.set_style(style)
    DrawOCVUi.set_scaling()


  @classmethod
  def set_style(cls, style:str="noosl") -> str or None:
    """Sets the visual style based on the given input.

    Args:
      style (string): Name of the style ["noosl", "cp"]

    """
    if style not in styles:
      cls.log.warning(f"Style '{style}' not found, available styles: {', '.join(list(styles.keys()))}.")
      return style

    # Set style
    cls.style = style
    cls.cola = styles[style]["cola"]
    cls.colb = styles[style]["colb"]
    cls.colc = styles[style]["colc"]
    cls.cold = styles[style]["cold"]
    cls.ft2b.loadFontData(fontFileName=styles[style]["fontb"][0], idx=styles[style]["fontb"][1])
    cls.ft2l.loadFontData(fontFileName=styles[style]["fontl"][0], idx=styles[style]["fontl"][1])
    cls.ft2r.loadFontData(fontFileName=styles[style]["fontr"][0], idx=styles[style]["fontr"][1])

    return style


  @classmethod
  def set_scaling(cls, mode:str="auto"):
    """Sets the ui thicnkess values based on the input image resolution.

    Reference https://optiviewusa.com/cctv-video-resolutions/ :
      QCIF						176 x 120		Quarter CIF (half the height and width as CIF)
      CIF							352 x 240
      2CIF						704 x 240		  2 times CIF width
      4CIF						704 x 480		  2 times CIF width and 2 times CIF height
      D1							720 x 480		  aka "Full D1"
      720p HD					1280 x 720		720p High Definition aka "HD-SDI"
      960p HD					1280 x 960		960p High Definition - a Sony specific HD standard
      1.3 MP					1280 x 1024		aka "1 Megapixel" or "1MP"
      2 MP						1600 x 1200		2 Megapixel
      1080p HD				1920 x 1080		1080p High Definition
      3 MP						2048 x 1536		3 Megapixel
      4 MP						2688 x 1520		4 Megapixel
      5 MP						2592 x 1944		5 Megapixel
      6 MP						3072 x 2048		6 Megapixel
      8 MP / 4K (Coax)			3840 x 2160		8 Megapixel
      12 MP / 4K (IP)				4000 x 3000

    Args:
      mode (string): Mode which scales the ui, auto, small, medium, large, custom

    """
    if mode == "auto":
      if cls.imwidth <= 1920 and cls.imwidth >= 1280 and cls.imheight <= 1080 and cls.imheight >= 720:
        cls.thicka = 2
        cls.heading = 16
        cls.body = 12

    cls.thickb = cls.thicka*2
    cls.thickc = cls.thicka*4
    # cls.ft2 = cv2.FONT_HERSHEY_DUPLEX
  

  @classmethod
  def randomize_colors(cls,):
    """Randomizes the give colors by the specified range.
    """
    # Colors BGR
    R = np.array(np.arange(96, 256, 32))
    G = np.roll(R, 1)
    B = np.roll(R, 2)
    colIDS = np.array(np.meshgrid(R, G, B)).T.reshape(-1, 3)
    # color = tuple(colIDS[class_id % len(colIDS)].tolist())[::-1]


  @classmethod
  def detection(cls, bbox, draw_bbox:bool=True, draw_vf:bool=True):
    """Method for drawing the annotation.

    Args:
      bbox (tuple): Tuple with points for the bounding box - left, top, right, bottom.

    """
    if draw_bbox: cls.bbox(cls.imcv, bbox)


  @classmethod
  def get_bbox_ss(cls, bbox:tuple):
    """Returns the bbox points in screen space - unnormalized.

    left = int(bbox[0] * cls.imwidth)
    top = int(bbox[1] * cls.imheight)
    right = int(bbox[2] * cls.imwidth)
    bottom = int(bbox[3] * cls.imheight)

    Args:
      bbox (tuple): Tuple of four ints with the normalized points for the bounding box.

    """
    cls.bbox = (int(bbox[0] * cls.imwidth), int(bbox[1] * cls.imheight), int(bbox[2] * cls.imwidth), int(bbox[3] * cls.imheight))
    return cls.bbox


  @classmethod
  def get_bbox_center(cls, bbox:tuple=None):
    """Returns the center point of the bbox."""
    if bbox is None: bbox = cls.bbox
    return ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)


  @classmethod
  def get_bbox_edge_centers(cls, bbox:tuple=None):
    """Calculate the center of each edge of the bounding box.

    Should be used with the unnormalized - screen space coords.

    """
    if bbox is None: bbox = cls.bbox
    # Calculate the center of each edge
    left_center = (bbox[0], (bbox[1] + bbox[3]) // 2)
    top_center = ((bbox[0] + bbox[2]) // 2, bbox[1])
    right_center = (bbox[2], (bbox[1] + bbox[3]) // 2)
    bottom_center = ((bbox[0] + bbox[2]) // 2, bbox[3])
    return(left_center, top_center, right_center, bottom_center)


  @classmethod
  def get_bbox_scaled_edge(cls, bbox:tuple=None, scale:float=0.1):
    """Returns the scaled length of the shorter edge of the bbox.
    """
    if bbox is None: bbox = cls.bbox
    return int(min((bbox[2] - bbox[0]), (bbox[1] - bbox[3])) * scale)


  @classmethod
  def get_bbox_pts(cls, bbox:tuple=None) -> tuple:
    """Gets all anchor points of the bounding box including centers.

    Note:
      lefttop, righttop, rightbottom, leftbottom, top, right, bottom, left, center

    """
    if bbox is None: bbox = cls.bbox
    # First unpack single bbox points
    left, top, right, bottom = bbox
    # Corners 
    lefttop, righttop, rightbottom, leftbottom = (left, top), (right, top), (right, bottom), (left, bottom)
    # Edge centers
    left, top, right, bottom = cls.get_bbox_edge_centers(bbox)
    center = cls.get_bbox_center(bbox)

    return (lefttop, righttop, rightbottom, leftbottom, top, right, bottom, left, center)


  @classmethod
  def get_text_size(cls, text:str, fonth:int=None, padding:int=0, weight:str="regular"):
    """Gets the width and height of the bounding box for the specified text string.
    """
    if fonth is None: fonth = cls.heading + padding
    else: fonth + padding
    if weight == "regular": ft2 = cls.ft2r
    if weight == "light": ft2 = cls.ft2l
    if weight == "bold": ft2 = cls.ft2b

    return ft2.getTextSize(text, fonth, cls.fontt)[0]


  @classmethod
  def get_inference_time(cls, unit:str="ms", dnn:cv2.dnn=None) -> float:
    """Returns the inference of the model as a float value."""
    if dnn is None: dnn = cls.dnn
    timer = dnn.getPerfProfile()[0]
    time_eval = 0
    if unit == "ms":
      time_eval = timer * 1000.0 / cv2.getTickFrequency()
    elif unit == "fps":
      time_eval = cv2.getTickFrequency() / timer

    return time_eval


  @classmethod
  def bbox_outline(cls, bbox:tuple=None, color=None, opacity:int=255):
    """Method for drawing the detection bounding box.
    """
    if bbox is None: bbox = cls.bbox
    if color is None: color = cls.cola
    color = (color[0], color[1], color[2], opacity)
    cv2.rectangle(cls.imsh, (bbox[0], bbox[1]), (bbox[2], bbox[3]),	color, cls.thicka)


  @classmethod
  def bbox_corners(cls, bbox:tuple=None, color=None, scale:float=0.05, opacity:int=255):
    """Draws the bbox_corners corners for the given detection.

    Note:
      Unpack: left, top, right, bottom = bbox

    TODO:
      if style == "cross":
        marker = cv2.MARKER_CROSS
        # corners = [plt, prt, prb, plb]
        corners = [(left-scaleh, top-scaleh), (right+scaleh, top-scaleh), (right+scaleh, bottom+scaleh), (left-scaleh, bottom+scaleh)]
        [cv2.drawMarker(cls.imcv, pos, color, marker, scale, cls.thickb) for pos in corners]

    """
    if bbox is None: bbox = cls.bbox
    if color is None: color = cls.cola
    color = (color[0], color[1], color[2], opacity)

    # Get shortest edge from detected bbox
    scale = cls.get_bbox_scaled_edge(bbox, scale)
    # left, top, right, bottom = bbox
    left, top, right, bottom = (bbox[0]-cls.thicka), (bbox[1]-cls.thicka), (bbox[2]+cls.thicka), (bbox[3]+cls.thicka)
    leftsc, topsc, rightsc, bottomsc = (left-scale), (top-scale), (right+scale), (bottom+scale)
    plt, prt, prb, plb = (left, top), (right, top), (right, bottom), (left, bottom)

    # top left corner
    cv2.line(cls.imsh, plt, (leftsc, top), color, cls.thickb)
    cv2.line(cls.imsh, plt, (left, topsc), color, cls.thickb)
    # top right corner
    cv2.line(cls.imsh, prt, (rightsc, top), color, cls.thickb)
    cv2.line(cls.imsh, prt, (right, topsc), color, cls.thickb)
    # bottom right corner
    cv2.line(cls.imsh, prb, (rightsc, bottom), color, cls.thickb)
    cv2.line(cls.imsh, prb, (right, bottomsc), color, cls.thickb)
    # bottom left corner
    cv2.line(cls.imsh, plb, (leftsc, bottom), color, cls.thickb)
    cv2.line(cls.imsh, plb, (left, bottomsc), color, cls.thickb)


  @classmethod
  def bbox_frame(cls, bbox:tuple=None, color=None, scale:float=0.05, opacity:int=255):
    """Draws the bbox_corners corners for the given detection.

    Note:
      Unpack: left, top, right, bottom = bbox

    TODO:
      if style == "cross":
        marker = cv2.MARKER_CROSS
        # corners = [plt, prt, prb, plb]
        corners = [(left-scaleh, top-scaleh), (right+scaleh, top-scaleh), (right+scaleh, bottom+scaleh), (left-scaleh, bottom+scaleh)]
        [cv2.drawMarker(cls.imcv, pos, color, marker, scale, cls.thickb) for pos in corners]

    """
    if bbox is None: bbox = cls.get_bbox_pts()[4:8]
    if color is None: color = cls.cola

    color = (color[0], color[1], color[2], opacity)
    edgs = cls.get_bbox_scaled_edge(scale=scale)

    # top line
    cv2.line(cls.imsh, bbox[0], (bbox[0][0], bbox[0][1]-edgs), color, cls.thickb)
    # right line
    cv2.line(cls.imsh, bbox[1], (bbox[1][0]+edgs, bbox[1][1]), color, cls.thickb)
    # bottom line
    cv2.line(cls.imsh, bbox[2], (bbox[2][0], bbox[2][1]+edgs,), color, cls.thickb)
    # left line
    cv2.line(cls.imsh, bbox[3], (bbox[3][0]-edgs, bbox[3][1]), color, cls.thickb)


  @classmethod
  def text(cls, text:str="label", pxy:tuple=None, alignh:str="center", alignv:str="above", weight:str="light",
    fonth:float=None, fontc=None, padding:int=8, bboxc=None, bboxo:int=255, draw_bbox:bool=True
  ):
    """Draws a text label.

    """
    if pxy is None: pxy = (cls.bbox[0], cls.bbox[1])
    if fonth is None: fonth = cls.heading
    if fontc is None: fontc = cls.cold
    if bboxc is None: bboxc = cls.cola
    if weight == "regular": ft2 = cls.ft2r
    if weight == "light": ft2 = cls.ft2l
    if weight == "bold": ft2 = cls.ft2b

    fontc = (fontc[0], fontc[1], fontc[2], 255)
    bboxc = (bboxc[0], bboxc[1], bboxc[2], bboxo)

    twh = cls.get_text_size(text, fonth, padding)
    twhh = [pt // 2 for pt in twh]
    padh = padding // 2

    # Shift align - don't ask -.-
    if alignh == "left" or alignv == "above":
      bxy = (pxy[0], pxy[1]-twh[1]-padding)
      bwh = (pxy[0]+twh[0]+padding, pxy[1])
      txy = (bxy[0]+padh, bwh[1]-padh)

    if alignh == "center":
      bxy = (pxy[0]-twhh[0]-padh, pxy[1]-twh[1]-padding)
      bwh = (pxy[0]+twhh[0]+padh, pxy[1])
      txy = (bxy[0]+padh, bwh[1]-padh)
      
    if alignh == "right":
      bxy = (pxy[0]-twh[0]-padding, pxy[1])
      bwh = (pxy[0], pxy[1]-twh[1]-padding)
      txy = (bxy[0]+padh, bxy[1]-padh)

    if alignv == "center":
      bxy = (bxy[0], bxy[1] + twhh[1] + padh)
      bwh = (bwh[0], bwh[1] + twhh[1] + padh)
      txy = (txy[0], txy[1] + twhh[1] + padh)

    if alignv == "below":
      bxy = (bxy[0], bxy[1] + twh[1] + padding)
      bwh = (bwh[0], bwh[1] + twh[1] + padding)
      txy = (txy[0], txy[1] + twh[1] + padding)

    # Handle case when the label is above the image frame.
    # if pxy[1] < twh[1]: shift_down = int(2 * twh[1])
    # else:	shift_down = 0

    if draw_bbox: cv2.rectangle(cls.imsh, bxy, bwh, bboxc, -1)
    ft2.putText(cls.imtxt, text, txy, fonth, fontc, cls.fontt, cls.linet, True)
    # cv2.putText(cls.imtxt, text, txy, cls.ft2b, fonth, fontc, cls.fontt, cls.linet, True)
    return twh


  @classmethod
  def stats(cls, pos:str="topleft", decimals:int=2, pads:tuple=None, opacity:bool=True, model_eval:bool=True, imwh:bool=True):
    """Draws stats for debbuging.
    
    Reference:
      30 FPS - 33.33 MS		|		60 FPS - 16.66 MS		|		120 FPS - 8.333 MS		|		240 FPS - 4.166 MS

    """
    if pos == "topleft":
      if pads is None: pads = int(min(cls.imwidth, cls.imheight) * 0.025)
      px, py = pads, pads
    
    padt = 6
    thp = cls.body + padt

    str_imwh = f"Image size: {cls.imwidth} x {cls.imheight} px"
    cls.text(str_imwh, (px, py), alignh="left", alignv="below", fonth=cls.body, padding=padt, bboxo=opacity)

    py += thp
    fps = f"Inference: {cls.get_inference_time('fps'):{decimals}.{decimals}f} fps"
    cls.text(fps, (px, py), alignh="left", alignv="below", fonth=cls.body, padding=padt, bboxo=opacity)


  @classmethod
  def image(cls, imcv):
    """Sets the image on which the ui elements will be drawn.
    """
    cls.imcv = imcv
    cls.imheight = imcv.shape[0]
    cls.imwidth = imcv.shape[1]
    # Create blend layers
    cls.imsh = np.zeros((cls.imheight, cls.imwidth, 4), cls.imcv.dtype)
    cls.imtxt = np.zeros((cls.imheight, cls.imwidth, 4), cls.imcv.dtype)
    cls.imout = np.zeros((cls.imheight, cls.imwidth, 4), cls.imcv.dtype)


  @classmethod
  def combine(cls):
    """Gets the blended out image."""
    masks = cls.imsh[:, :, 3] / 255
    maskt = cls.imtxt[:, :, 3] / 255
    for channel in range(3):
      cls.imout[:, :, channel] = (
        ((cls.imcv[:, :, channel] * (1-masks))
         +(cls.imsh[:, :, channel] * masks)*(1-maskt))
        +(cls.imtxt[:, :, channel] * maskt)
      )
    return cls.imout.astype(np.uint8)




def get_patches(position):
  """Returns patches of image from around our blemished area.
  """
  posx, posy = position
  # Finding specific number of evenly spaced points around our patch
  t = np.linspace(0, 2 * np.pi, number_of_candidates + 1)  # +1 because, the first and the last one will be the same
  tCos = (np.round(2 * radius * np.cos(t)) + posx).astype(int)
  tSin = (np.round(2 * radius * np.sin(t)) + posy).astype(int)
  candidate_centers = np.c_[
    (np.round(2 * radius * np.cos(t)) + posx).astype(int),
    (np.round(2 * radius * np.sin(t)) + posy).astype(int)
  ][:-1]  # We remove the last one
  # Remove the patches that dont fit into our image
  condition = np.array([
    0 + radius < center[0] < image.shape[1] - radius
    and 0 + radius < center[1] < image.shape[0] - radius
    for center in candidate_centers
  ])
  
  patches = [image[posy - radius:posy + radius, posx - radius:posx + radius] for [posx, posy] in candidate_centers[condition]]
  return patches




def calculate_candidate_gradient_measures(patches):
  sobel_x = [calculate_mean_gradient(patch, 1, 0) for patch in patches]
  sobel_y = [calculate_mean_gradient(patch, 0, 1) for patch in patches]

  total_gradients = np.add(sobel_x, sobel_y)
  return total_gradients


# Calculates a single number gradient score (vertical or horizontal) for a single image
def calculate_mean_gradient(image_patch, xorder, yorder):
  sobel = cv2.Sobel(image_patch, cv2.CV_64F, xorder, yorder, ksize=3)
  abs_sobel = np.abs(sobel)  # Because sobel results can be negative
  mean_gradient = np.mean(np.uint8(abs_sobel))
  return mean_gradient



def get_gradient_measures(patches):
  """Calculates single number gradient score for every patch.
  """
  for patch in patches:
    gradient_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)

    mean_gradient_x = np.mean(np.uint8(np.abs(gradient_x)))
    mean_gradient_y = np.mean(np.uint8(np.abs(gradient_y)))

  return np.add(mean_gradient_x, mean_gradient_x)




def remove_blemish(action, x, y, flags, userdata):
  """Clears the blemish around left mouse click.
  """
  global image
  # Action to be taken when left mouse button is pressed
  # Find and replace the blemish
  if action == cv2.EVENT_LBUTTONDOWN:
    patch_center = (x, y)

    # dont do blemish removal if the patch doesnt fit in the image
    patch_fits = 0 + radius < x < image.shape[1] - radius and 0 + radius < y < image.shape[0] - radius

    if not patch_fits: return

    patches = get_patches(patch_center)
    gradients = calculate_candidate_gradient_measures(patches)

    gradient_min_idx = np.argmin(gradients)
    print(gradients)
    print(gradient_min_idx)
    gradient_min_patch = patches[int(gradient_min_idx)]

    mask = np.full_like(gradient_min_patch, 255, dtype=gradient_min_patch.dtype)
    image = cv2.seamlessClone(gradient_min_patch, image, mask, patch_center, cv2.NORMAL_CLONE)

    # cv2.imshow(winname, gradient_min_patch)
    cv2.imshow(winname, image)




if __name__ == "__main__":
  image = cv2.imread(imagePath, 1)
  dummy = image.copy()

  postxt = (image.shape[1]//2, image.shape[0]-25)

  uidraw = DrawOCVUi(image)
  uidraw.text("Press 'esc' to quit, 'c' to reset the image", postxt, alignh="center", alignv="center", fonth=uidraw.body, padding=6, bboxo=192)
  cv2.imshow(winname, uidraw.combine())

  cv2.setMouseCallback(winname, remove_blemish)

  while True:  
    key = cv2.waitKey(1)
    match key:
      case 99:  # c is pressed
        image = dummy.copy()
        uidraw.image(image)
        uidraw.text("Press 'esc' to quit, 'c' to reset the image", postxt, alignh="center", alignv="center", fonth=uidraw.body, padding=6, bboxo=192)
        cv2.imshow(winname, uidraw.combine())
      case 27:  # esc is pressed 
        break

  cv2.destroyAllWindows()