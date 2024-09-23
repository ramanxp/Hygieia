from roboflow import Roboflow
rf = Roboflow(api_key="l73bIMnhr4Mui2reJSPx")
project = rf.workspace("raman-0g3ds").project("hygieiafracdetect")
model = project.version(1).model

# infer on a local image
#print(model.predict("your_image.jpg", confidence=40, overlap=30).json())

# visualize your prediction
#model.predict("your_image1.jpg", confidence=40, overlap=30).save("prediction1.jpg")

# infer on an image hosted elsewhere
print(model.predict(r'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTriIydoOvTBvLAANbtntIQJs10qWgCMc_JsQ&s', hosted=True, confidence=40, overlap=30).save("prediction2.jpg"))

'''
from roboflow import Roboflow
rf = Roboflow(api_key="l73bIMnhr4Mui2reJSPx")
project = rf.workspace("raman-0g3ds").project("hygieiafracdetect")
dataset = project.version(2).download("yolov8")
'''