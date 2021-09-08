import cv2
import numpy
import os

# 한글경로 쓰기
def imwrite(image, savePath, parameters = None): 
        
    try: 
        
        result, array = cv2.imencode(os.path.splitext(savePath)[1], image, parameters) 
        
        if result: 
            
            with open(savePath, mode = 'w+b') as f:
                
                array.tofile(f)

                return True 
                
        else: 
            
            return False 
            
    except Exception as exception: 
        
        print(exception) 
        
        return False



if __name__ == '__main__':

    baseDirectory = os.path.abspath(os.path.dirname(__file__))

    imageDirectory = f'{baseDirectory}/MBTI/ISTP/W'

    haarcascade = cv2.CascadeClassifier(f'{baseDirectory}/haarcascades/haarcascade_frontalface_alt2.xml') # 검출 대상 : 정면 얼굴 검출 
    # OpenCV에서 제공하는 하르 기반 분류기 XML파일 

    savePath = f'{baseDirectory}/Cropped_ISTP_W' 
    os.makedirs(savePath, exist_ok = True)

    filenameList = os.listdir(imageDirectory)

    for filename in filenameList:

        image = numpy.fromfile(f'{imageDirectory}/{filename}', numpy.uint8) # 한글경로 읽기
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)

        #cv2.imshow('Original', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # scaleFactor : 이미지 피라미드 스케일, minNeighbors : 인접 객체 최소 거리 픽셀, minSize : 탐지 객체 최소 크기
        boundingBoxes = haarcascade.detectMultiScale(image, scaleFactor =  1.3, minNeighbors = 3, minSize = (10,10))
        # image : 검출하고자 하는 원본 이미지 // defalt = 1.1
        # minNeighbors : 인접 객체 최소 거리 픽셀로 1~3정도로 설정을 하면 얼굴인식율 높았고 4이상으로 설정할 경우 인식율이 떨어지는것을 확인했다. defalut = 3
        # minsize : 탐지 객체 최소 크기로, 10*10 이하의 객체는 인식을 하지 않는다. maxSize는 설정하지 않음으로써 사진에서 크게 나온 얼굴도 검출이 가능하다. 
        for boundingBox in boundingBoxes:
                
            x, y, width, height = boundingBox

    

            croppedImage = image[y : y + height, x : x + width,  : ] 
            # 얼굴 인식 후 크롭 사이즈 결정
            

            #cv2.imshow(f'Cropped', croppedImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            imwrite(image = croppedImage, savePath = f'{savePath}/{filename}')