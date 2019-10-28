import sys

defaultcfg = [32, 64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

split_groups = [ 0, 0, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4 ]

max_splitcfg = [ 32, 80, (160, 2), 160, (320, 2), 320, (640, 2), 640, 640, 640, 640, 640, (1280, 2), 1280 ]

# input and output resolutions
resolutions = [ (32, 32), (32, 32), (32, 16), (16, 16), (16, 8), (8, 8), (8, 4), (4, 4), (4, 4), (4, 4),  (4, 4),  (4, 4), (4, 2), (2, 2) ]


