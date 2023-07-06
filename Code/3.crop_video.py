import cv2
import os

def crop_video(video_path, trim_coordinates):
    # Read video file
    video_capture = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define output video filename
    output_path = os.path.join(os.path.dirname(video_path), 'cropped_' + os.path.basename(video_path))

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (trim_coordinates[2] - trim_coordinates[0], trim_coordinates[3] - trim_coordinates[1]))

    # Set starting frame
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Read and crop frames
    for _ in range(total_frames):
        ret, frame = video_capture.read()
        if ret:
            cropped_frame = frame[trim_coordinates[1]:trim_coordinates[3], trim_coordinates[0]:trim_coordinates[2]]
            output_video.write(cropped_frame)

    # Release resources
    video_capture.release()
    output_video.release()

    print('Cropping complete. Cropped video saved as:', output_path)

# Define video file path
video_file = 'Visualization/run1/run1.mp4'

# Define trim coordinates
trim_coordinates = (100, 300, 1200, 700)

# Call the crop_video function
crop_video(video_file, trim_coordinates)
