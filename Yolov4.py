
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:24:59 2023

@author: asonm
"""

#%% 1.Bölüm
import cv2
import numpy as np

cap = cv2.VideoCapture("C:/Users/asonm/Desktop/Atlas Robot/New dog-like robot from Boston Dynamics can open doors.mp4")
#print(img)

while True:
    ret,frame = cap.read()





    frame_width = frame.shape[1]
    frame_height = frame.shape[0]




    frame_blob = cv2.dnn.blobFromImage(frame,1/255,(416,416),swapRB=True,crop=False)

    labels = ["Atlas Robot"]


    colors = ["0,255,255","0,0,255","255,255,0","0,255,0","255,0,0","255,0,255"]
    colors = [np.array(color.split(",")).astype("int") for color in colors]
    colors = np.array(colors)
    colors = np.tile(colors,(18,1))



    model = cv2.dnn.readNetFromDarknet("C:/Users/asonm/Desktop/Atlas Robot/spot_yolov4.cfg","C:/Users/asonm/Desktop/Atlas Robot/spot_yolov4_last.weights")

    layers = model.getLayerNames()
    output_layer = [layers[layer -1] for layer in model.getUnconnectedOutLayers()]
    
    model.setInput(frame_blob)
    
    detection_layers = model.forward(output_layer)


    ################# NON-MAXIMUM-SUPPRESSION - OP1 ################
    
    ids_list = []
    boxes_list = []
    confidence_list = []
    
    ################## END OF OP1 ####################









    for detection_layer in detection_layers:
        for object_detection in detection_layer:
            
            scores = object_detection[5:]
            predicted_id = np.argmax(scores)
            confidence = scores[predicted_id]
    
            if confidence > 0.35:
                
                label = labels[predicted_id]
                bounding_box = object_detection[0:4] * np.array([frame_width,frame_height,frame_width,frame_height])
                (box_center_x,box_center_y,box_width,box_height) = bounding_box.astype("int")
    
                start_x = int(box_center_x - (box_width/2))
                start_y = int(box_center_y - (box_height/2))
                
                ################# NON-MAXIMUM-SUPPRESSION - OP2 ################            
                
                ids_list.append(predicted_id)
                confidence_list.append(float(confidence))
                boxes_list.append([start_x,start_y,int(box_width),int(box_height)])
                
                ################## END OF OP2 ####################
                
                
            
################# NON-MAXIMUM-SUPPRESSION - OP3 ################           
            
    max_ids = cv2.dnn.NMSBoxes(boxes_list, confidence_list, 0.5, 0.4)

    for max_id in max_ids:
        max_class_id = max_id
        box = boxes_list[max_class_id]

        start_x = box[0]
        start_y = box[1]
        box_width = box[2]
        box_height = box[3]

        predicted_id = ids_list[max_class_id]
        label = labels[predicted_id]
        confidence = confidence_list[max_class_id]
    
    
    
    
 ################## END OF OP3 ####################
 

        end_x = start_x + box_width
        end_y = start_y + box_height

        box_color = colors[predicted_id]
        box_color = [int(each) for each in box_color]


        label = "{}: {:.2f}%".format(label,confidence * 100)
        print("predincted object {}".format(label))


        cv2.rectangle(frame, (start_x,start_y), (end_x,end_y), box_color,2)
        cv2.putText(frame, label, (start_x,start_y -10), cv2.FONT_HERSHEY_COMPLEX, 0.5, box_color,2)



    cv2.imshow("Boston Robot", frame)

    if cv2.waitKey(100) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()


















