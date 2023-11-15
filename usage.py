from PIL import Image

from face_recognition_ai import match_faces, show_detections

unknown_with_multiple_faces = Image.open("images/multi.jpeg")

known = Image.open("images/sakib.jpg")
person_name: str = "Sakib"

print(True if True in match_faces(unknown_with_multiple_faces, known) else False)
img = show_detections(unknown_with_multiple_faces, known, person_name)
img.show()
