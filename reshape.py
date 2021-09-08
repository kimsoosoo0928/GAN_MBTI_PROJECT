from PIL import Image # PIL 다양한 이미지 파일 형식을 지원하고 강력한 이미지 처리와 그래픽 기능을 제공하는 자유-오픈 소스 소프트웨어 라이브러리이다
import os.path #  os.path 내에는 경로반환, 경로추출 등 파일/디렉토리 경로와 관련된 많은 함수를 제공해준다

targerdir = r"D:\TEAM_PROJECT\data\ISFP\W" # original data folder  
newpath = r"D:\TEAM_PROJECT\data\ISFP\_W" # resize data folder 

files = os.listdir(targerdir) # 지정한 디렉토리 내의 모든 파일과 디렉토리의 리스트를 반환 

format = [".jpg",".png",".jpeg","bmp",".JPG",".PNG","JPEG","BMP"] # 지원하는 파일 형태의 확장자들
for (path,dirs,files) in os.walk(targerdir): # os.walk()를 사용하면 어떤 경로의 모든 하위 디렉토리를 탐색 할 수 있다. 
    for file in files:
         if file.endswith(tuple(format)): # endswith는 특정 문자로 끝이나는 문자열을 찾는 메소드 이기때문에, if문에서 앞서 지정한 확장자를 찾을 수 있다.
             image = Image.open(path+"\\"+file)
             print(image.filename)
             print(image.size)

             image=image.resize((150, 150)) # 이미지를 150 * 150 으로 리사이즈
             image.save(newpath+"\\"+file) # 앞서 지정한 새로운 경로로 파일을 저장 
             print(image.size)

         else: # 지정한 확장자가 아니면 InValid를 출력하고 종료
             print(path)
             print("InValid",file)