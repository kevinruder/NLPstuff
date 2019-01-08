TRAIN_DATA = []
import glob

file_directory = glob.glob("C:/Users/kvru/Downloads/Nytår/*.txt")

for txt in file_directory:
    with open(txt) as f:
        content = f.readlines()
        for line in content:
            if(line != "\n"):
                TRAIN_DATA.append(line)
                
with open("compiledSpeech","w") as file:
    for line in TRAIN_DATA:
        file.write(line)

