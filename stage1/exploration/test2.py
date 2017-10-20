import segment as sg

APS_FILE_NAME = '00360f79fd6e02781457eda48f85da90'
zones = sg.get_cropped_zones('./', filelist = [APS_FILE_NAME], file_extension='aps', angle=1)
images,label = zones[0]
print(images[0])