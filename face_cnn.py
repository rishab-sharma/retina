import pandas as pd
from PIL import Image

imgSize = 48 ** 2

#df = pd.read_csv("fer2013/fer2013.csv", sep=",")
imgStr = df['pixels'][0]

imgarr= map(int,imgStr.split(" "))

img = Image.frombytes("L" , (48 , 48) , imgStr , 'raw')

img.show()