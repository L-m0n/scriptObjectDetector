import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') 

webcam = cv2.VideoCapture(0)

while True:
    status, frame = webcam.read()
    if not status:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, width, height) in faces:
        cv2.rectangle(frame,  (x, y), (x+width, y+height), (250, 0, 0), 2)

    cv2.putText(
        frame, f"Faces Detected: {len(faces)}", (5, 25),
        cv2.FONT_HERSHEY_COMPLEX,0.8, (0,0,0), 2
    )

    cv2.imshow('Detector de Rostros', frame)

    if cv2.waitKey(1) == 113: # El Valor ASCII que representa la letra "q"
        cv2.imwrite("haar_resultado.jpg", frame)
        break

webcam.release()
cv2.destroyAllWindows()