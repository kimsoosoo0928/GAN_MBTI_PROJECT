from PIL import Image
import os.path

targerdir = r"D:\01_TEAM_PROJECT\_data\ISFJ\M" #해당 폴더 설정 
newpath = r"D:\01_TEAM_PROJECT\_data\ISFJ\_M"

files = os.listdir(targerdir)

format = [".jpg",".png",".jpeg","bmp",".JPG",".PNG","JPEG","BMP"] #지원하는 파일 형태의 확장자들
for (path,dirs,files) in os.walk(targerdir):
    for file in files:
         if file.endswith(tuple(format)):
             image = Image.open(path+"\\"+file)
             print(image.filename)
             print(image.size)

             image=image.resize((150, 150))
             image.save(newpath+"\\"+file)
             print(image.size)

         else:
             print(path)
             print("InValid",file)