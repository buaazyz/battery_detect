import cv2




#统一输入尺寸
def Resize_Input(img,size = (1040,2000,3)):
    """
    输入: 待转换的图片img，目标尺寸size,默认为1040*2000*3
    输出：转换好后的图片img_new
    """
    if img.shape<=size:
        img_new = cv2.copyMakeBorder(img,0,size[0]-img.shape[0],0,size[1]-img.shape[1],cv2.BORDER_CONSTANT,value=[255,255,255])
    else:
        img_new = img[:size[0],:size[1],:]
    #print(img_new.shape)
    return img_new

img = Resize_Input( cv2.imread('small.png') )
print(img.shape)