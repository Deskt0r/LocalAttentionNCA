

import os
from PyPDF2 import PdfReader, PdfWriter

def trim_pdf(pdf_path, left, bottom, right, top):
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        pdf = PdfReader(file)
        output_pdf = PdfWriter()

        # Iterate over each page
        for page in pdf.pages:
            # Set the new dimensions for trimming
            page.trimbox.lower_left = (left, bottom)
            page.trimbox.upper_right = (right, top)
            page.cropbox.lower_left = (left, bottom)
            page.cropbox.upper_right = (right, top)

            # Add the modified page to the output PDF
            output_pdf.add_page(page)

        # Save the trimmed PDF
        output_folder = 'Visualization/splitted_pdfs'  # Specify the output folder path
        output_filename = os.path.splitext(os.path.basename(pdf_path))[0] + '_trimmed.pdf'
        output_path = os.path.join(output_folder, output_filename)
        with open(output_path, 'wb') as output_file:
            output_pdf.write(output_file)

        print(f'Trimmed PDF saved: {output_path}')

# Specify the input folder path containing PDF files
input_folder = 'Visualization/Static_Positioning/Static_Positioning-1'

# Specify the output folder path
output_folder = 'Visualization/splitted_pdfs'

# Specify the trimming coordinates (left, bottom, right, top)
trim_coordinates = (50, 100, 420, 230)

# Iterate over each file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith('.pdf'):
        # Construct the full file paths
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)

        # Trim the PDF using the specified coordinates
        trim_pdf(input_file_path, *trim_coordinates)
