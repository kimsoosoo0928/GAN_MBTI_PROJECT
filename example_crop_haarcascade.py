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

    haarcascade = cv2.CascadeClassifier(f'{baseDirectory}/haarcascades/haarcascade_frontalface_alt2.xml')

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

        for boundingBox in boundingBoxes:
                
            x, y, width, height = boundingBox

            # cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), thickness = 2) # 경계상자 그리기

            # cv2.imshow(f'Haarcascade', image)
            # cv2.waitKey(0)

            croppedImage = image[y : y + height, x : x + width,  : ]
            #croppedImage = image[y : y + height + int(height/4), x -int(width/4) :x + width + int(width/4)]

            #cv2.imshow(f'Cropped', croppedImage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            imwrite(image = croppedImage, savePath = f'{savePath}/{filename}')