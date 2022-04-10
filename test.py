file1 = open('tmp.txt', 'r')
lines = file1.readlines()

print("SCALE_FACTOR,SUBSAMPLING_MODE,VIDEO_MODE,CPU,OMP,CUDA")
for i in range(1,len(lines),7):
    SCALE_FACTOR,SUBSAMPLING_MODE,VIDEO_MODE = lines[i].replace('\n','').split(',')
    SCALE_FACTOR,SUBSAMPLING_MODE,VIDEO_MODE = float(SCALE_FACTOR),int(SUBSAMPLING_MODE),int(VIDEO_MODE)
    CPU = float(lines[i+2].replace('\n','').replace('AVG_FPS=', ''))
    OMP = float(lines[i+4].replace('\n','').replace('AVG_FPS=', ''))
    CUDA = float(lines[i+6].replace('\n','').replace('AVG_FPS=', ''))
    print(SCALE_FACTOR,SUBSAMPLING_MODE,VIDEO_MODE,CPU,OMP,CUDA, sep=',')
