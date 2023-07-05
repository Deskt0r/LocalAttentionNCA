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
        output_path = os.path.splitext(pdf_path)[0] + '_trimmed.pdf'
        with open(output_path, 'wb') as output_file:
            output_pdf.write(output_file)

        print(f'Trimmed PDF saved: {output_path}')

# Provide the folder path containing PDF files
folder_path = 'Visualization/Static_Positioning/Static_Positioning-1'

# Specify the trimming coordinates (left, bottom, right, top)
trim_coordinates = (50, 100, 420, 230)

# Iterate over each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.pdf'):
        # Construct the full file path
        file_path = os.path.join(folder_path, filename)

        # Trim the PDF using the specified coordinates
        trim_pdf(file_path, *trim_coordinates)
