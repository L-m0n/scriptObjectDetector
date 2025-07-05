import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

webcam = cv2.VideoCapture(0) 

object_interest = [
    'person',
    'cell phone',
    'mouse'
    ]

selected_object = 'person'

executed = False

while True:
    status, frame = webcam.read()
    if not status:
        break

    bbox, label, conf = cv.detect_common_objects(frame)

    filtered_bbox = []
    filtered_label = []
    filtered_conf = []
    objectCount = dict.fromkeys(object_interest, 0)

    for idx, etiqueta in enumerate(label):
        if etiqueta in object_interest:
            filtered_bbox.append(bbox[idx])
            filtered_label.append(label[idx])
            filtered_conf.append(conf[idx])
            objectCount[etiqueta] += 1

    output = draw_bbox(frame, filtered_bbox, filtered_label, filtered_conf)

    displayed_text = ""
    for obj, count in objectCount.items():
        displayed_text += f"{obj}: {count} | "

    cv2.putText(output, displayed_text, (5, 25), cv2.FONT_HERSHEY_COMPLEX,0.6, (0,0,0), 2)

    if objectCount[selected_object] >= 2 and not executed:
        cv2.imwrite("mobilenet_filtro_resultado.jpg", output)
        executed = True
        print("Saved mobilenet_filtro_resultado")


    cv2.imshow("Detector Filtrado", output)

    if cv2.waitKey(1) == 113: # El Valor ASCII que representa la letra "q"
        break

webcam.release()
cv2.destroyAllWindows()