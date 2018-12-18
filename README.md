# NEKO
Neural End-to-end Kuzushiji OCR

The image size is 576 x 576
All characters detected is in charlist.csv (409 characters at the moment)

I am thinking about adding config.py to put all variables there.

When doing OCR, just put images in decoder/source and run ocr.py
in ocr.py there are 3 steps

1. source image resize to 576
2. detection with torch
3. clustering with dbscan

all output are in decoder/ocr/
