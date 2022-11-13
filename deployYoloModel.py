import torch
import numpy as np
import cv2
import pafy
from time import time
from plyer import notification
from win10toast import ToastNotifier


class ObjectDetection:
    """
    Class implements Yolo5 model to make inferences on a youtube video/webcam using OpenCV.
    """
    
    def __init__(self, url, out_file):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        self._URL = url
        self.model = self.load_model()
        self.classes = self.model.names
        self.out_file = out_file
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\n\nDevice Used:",self.device)


    def get_video_from_url(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
        play = pafy.new(self._URL).streams[-1]
        assert play is not None
        return MP4V(play.url)

    def get_video_from_file(self, video_path):
        return cv2.VideoCapture(video_path)

    def capture_from_webcam(self):
        return cv2.VideoCapture(1)

    def load_model(self):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=r'C:\Users\LENOVO\Desktop\Code\AIP_project\Human_Fall_Detection_Using_Pretrained_YoLov5_final\weights\320_epochs_32_batch.pt', force_reload=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        return labels, cord


    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]


    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                if labels[i] == 0: bgr = (0, 0, 255)
                else: bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame, self.class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

        return frame

    '''
    def __call__(self): write bounding box on video
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        player = self.get_video_from_url()
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        four_cc = cv2.VideoWriter_fourcc(*"MP4V")
        out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))
        while True:
            start_time = time()
            ret, frame = player.read()
            if not ret:
                break
            results = self.score_frame(frame)
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = 1/np.round(end_time - start_time, 3)
            print(f"Frames Per Second : {fps}")
            out.write(frame)
        '''
    def __call__(self): #real-time detection
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        player = self.capture_from_webcam()
        #player = self.get_video_from_file('Input_videos/fall_detection_4.mp4')
        assert player.isOpened()
        x_shape = int(player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #four_cc = cv2.VideoWriter_fourcc(*"MJPG")
        #out = cv2.VideoWriter(self.out_file, four_cc, 20, (x_shape, y_shape))
        total_fps = 0
        frame_count = 0
        while True:
            start_time = time()
            ret, frame = player.read()
            if not ret:
                break
            results = self.score_frame(frame)
            ''' 
            if True: 
                if falling_time_start != 0:
                    falling_time_start = time()
                falling_time_end = time()
                if falling_time_end - falling_time_start == 3:
                    notification.notify(
                        title = "Sample Notification",
                        message = "This is a sample notification",
                        timeout = 10
                    )
                    toaster = ToastNotifier()
                    toaster.show_toast("Demo notification",
                    "Hello world",
                    duration=10)
            '''
            frame = self.plot_boxes(results, frame)
            end_time = time()
            fps = np.round(1/np.round(end_time - start_time, 3),3)
            total_fps += fps
            frame_count += 1
            print(f"Frame {frame_count} Processing. Frames Per Second : {fps}")
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        player.release()
        avg_fps = total_fps / frame_count
        print(f"Average FPS: {avg_fps:.3f}")
        cv2.destroyAllWindows()    

# Create a new object and execute.
detection = ObjectDetection("", "")
detection()