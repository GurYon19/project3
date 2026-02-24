import json
f=open('colab_runner.ipynb', encoding='utf-8')
d=json.load(f)
f.close()

for cell in d['cells']:
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        for i, line in enumerate(source):
            if 'cmd = f"python part2/train.py --data-dir \\"{found_path}\\" --format coco"' in line:
                # Add argument appending
                append_line = 'cmd += f" --checkpoint-dir \\"{os.path.join(DRIVE_ROOT, \'checkpoints/part2\')}\\" --log-dir \\"{os.path.join(DRIVE_ROOT, \'logs/part2\')}\\""\\n'
                if append_line not in source:
                    source.insert(i+1, append_line)
                break

f=open('colab_runner.ipynb', 'w', encoding='utf-8')
json.dump(d, f, indent=1)
f.close()
