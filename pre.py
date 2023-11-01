from ultralytics import YOLO

model=YOLO('best2.pt')

model.predict('03.mp4',save=True)


#result = model('01.jpeg')

#print(result)
