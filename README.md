# License-Plate-Detection-Recogntion
License Plate Detection is done using YOLO(You Only Look Once) model 
Recognition of characters from License Plate is done using Tesseract OCR

### Installation Guide
1. Clone or download the repository https://github.com/Pulkitg64/License-Plate-Detection-Recogntion <br>
2. Install tesseract in your computer using link: https://github.com/UB-Mannheim/tesseract/wiki <br>
3. Copy the installed location path where tesseract is installed and paste in line no. 22 of finalcode.py file. <br>
4. Move the eng.user-patterns file to the tessdata folder Tesseract OCR and copy the location and replace the path in line no. 119 of finalcode.py file.<br>
5. Download the trained weights file from the link: https://drive.google.com/open?id=1eIrbGhLcGrSt2-LB8Tt_YOjDva9pWkOl
6. Place the Video Camera in front of license plate of car and run the code. The code will generate License plate numbers and append to list1.
