from pathlib import Path
from sphfile import SPHFile

if __name__ == "__main__":
    path = 'C:\\Users\\Dekel\\Downloads\\לימודים\\deep learning\\an4_sphere\\an4\\wav\\an4test_clstk\\'
    path_list = Path(path).glob('**/*.sph')
    files_amount = 0
    for path in path_list:
        # because path is object not string
        path_str = str(path)
        sph = SPHFile(path_str)
        # Note that the following loads the whole file into ram
        path_tuple = path_str.rsplit('\\', 1)
        path_to_dir = path_tuple[0]# path to dir
        file_name_with_extension = path_tuple[1]# full file name
        file_name = file_name_with_extension.split('.')[0]
        wav_file = path_to_dir + '\\' + file_name + '.wav'
        print(wav_file)
        sph.write_wav(wav_file)
        files_amount += 1

    print("Amount of files: " + str(files_amount))
