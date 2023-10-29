import subprocess
import os
import shutil
import pandas as pd
from tqdm import tqdm

JOERNPATH="..\script\joern-cli"
PARSEPATH="..\script\joern-cli"
root_dir = '..\data'

source_dir = "sourcecode_dir"

import subprocess

def move_c_files_to_folders(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith('.c'):
            folder_name = os.path.splitext(file)[0]
            os.makedirs(os.path.join(folder_path, folder_name), exist_ok=True)
            shutil.move(os.path.join(folder_path, file), os.path.join(folder_path, folder_name, file))


def parse_source_code_to_dot(file_path, f, out_dir_ast='ast_dir', out_dir_cpg='cpg_dir'):
    try :
        os.makedirs(out_dir_ast)
        os.makedirs(out_dir_cpg)
    except:
        pass
    print('parseing source code into ast...')
    shell_str = JOERNPATH + "joern-parse " + file_path
    print("shell_str:", shell_str)
    subprocess.call(shell_str, shell=True)
    # subprocess.run(shell_str, shell=True, capture_output=True, text=True, encoding='gbk')
    print('exporting cpg from cpg root...')
    print(out_dir_cpg)
    print(f.split('.')[0])
    # shell_export_cpg = "cmd /c " + JOERNPATH + "joern-export " + "--repr cpg14 --out " + out_dir_cpg + os.sep + f.split('.')[0]
    shell_export_cpg = f"{JOERNPATH}joern-export --repr cpg14 --out {out_dir_ast}{os.sep}{f.split('.')[0]}"
    print("shell_export_cpg:", shell_export_cpg)
    # subprocess.call(shell_export_cpg, shell=True)
    subprocess.run(shell_export_cpg, shell=True, capture_output=True, text=True, encoding='gbk')
    # output = subprocess.check_output(shell_export_cpg, shell=True, encoding='gbk')
    # with open(os.path.join(out_dir_cpg, f"{f.split('.')[0]}.txt"), 'w', encoding='gbk') as f:
    #     f.write(output)

import os
def main_func(source_dir = "sourcecode_dir", out_dir_cpg="cpg_dir"):
    dirs = os.listdir(source_dir)
    for c_folder in tqdm(dirs):
        file_path = source_dir + '\\' + c_folder
        cpg_path = out_dir_cpg + '\\' + c_folder
        if os.path.exists(cpg_path) and len(os.listdir(cpg_path)) > 0:
            print(f'{file_path} file has been processed')
            continue
        print(f'starting to process {file_path}')
        f = os.listdir(file_path)[0]
        parse_source_code_to_dot(file_path, f)

if __name__ == "__main__":
    move_c_files_to_folders('cfile_dir')
    main_func()
