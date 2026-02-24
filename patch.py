import json
f=open('colab_runner.ipynb', encoding='utf-8')
d=json.load(f)
f.close()
d['cells'][3]['source'][6] = '    !git clone -b part-2 {REPO_URL}\n'
d['cells'][3]['source'][10] = '    !git pull origin part-2\n'
d['cells'][2]['source'][6] = 'DRIVE_ROOT = "/content/drive/MyDrive/project3"\n'
d['cells'][4]['source'][1] = 'ZIP_PATH = os.path.join(DRIVE_ROOT, "part2/datasets/part2_dataset.zip")\n'
f=open('colab_runner.ipynb', 'w', encoding='utf-8')
json.dump(d, f, indent=1)
f.close()
