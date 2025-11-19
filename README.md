## pixelator

trying to calculate pixels of an object in an image, both horizontally and vertically, uses yolo3 model, although it being a weak ass model doesn't do much now

But it can easily detect each circular object as it uses hough circle transform function of opencv.

To use the detect circle use the tag:
```bash
python pixelator.py --detect-circles
```

made for the SIH finals, but can be used otherwise as well.

And you need the model files for it to be useful, listed in the gitignore file.