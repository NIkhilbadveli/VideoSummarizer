import cv2
import os


class FrameExtractor:
    """Uses OpenCV to extract ExtractedFrames of a video at a specified fps and save them into the path given."""

    def extract_frames(self, video_path, frames_path, n=10):
        # n is the number of ExtractedFrames from which you select 1 frame. So, once every n ExtractedFrames.

        # Create directory if frames_path doesn't exist
        try:
            if not os.path.exists(frames_path):
                print('Creating the directory to save ExtractedFrames')
                os.makedirs(frames_path)
        except OSError:
            print('Error! Could not create a directory')

        # reading the video from specified path
        vid_cap = cv2.VideoCapture(video_path)

        # reading the number of ExtractedFrames at that particular second
        # fps = vid_cap.get(cv2.CAP_PROP_FPS)

        current_frame_no = 0
        frames_captured = 0

        # Read frame by frame using while loop
        print('Extracting ExtractedFrames...')
        while True:
            grabbed, frame = vid_cap.read()
            if grabbed:
                if current_frame_no % n == 0:
                    cv2.imwrite(frames_path + '/frame_' + str(frames_captured) + '.jpg', frame)
                    frames_captured += 1
            else:
                break
            current_frame_no += 1
        print('Extracting ExtractedFrames done...')
