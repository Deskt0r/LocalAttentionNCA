import os
import cv2
from pdf2image import convert_from_path
from natsort import natsorted
import numpy as np

# Folder path containing the PDF files
folder_path = 'Visualization/splitted_pdfs'

# Output video file path
output_path = 'Visualization/run1/run1.mp4'

# Get a list of PDF files in the folder and sort them alphanumerically
pdf_files = natsorted([file for file in os.listdir(folder_path) if file.endswith('.pdf')])

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = None
frame_width, frame_height = 0, 0

# Process each PDF file
for pdf_file in pdf_files:
    # Convert PDF to images
    images = convert_from_path(
        os.path.join(folder_path, pdf_file),
        size=(frame_width, frame_height)  # Set the original size of the PDF
    )

    # Create video writer based on the first image's dimensions
    if video_writer is None:
        frame_width, frame_height = images[0].size
        output_size = (frame_width, frame_height)
        video_writer = cv2.VideoWriter(output_path, fourcc, 5, output_size)

    # Process each image and write it as a frame in the video
    for image in images:
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

# Release the video writer
video_writer.release()
