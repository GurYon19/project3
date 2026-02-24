import json
import os

f=open('colab_runner.ipynb', encoding='utf-8')
d=json.load(f)
f.close()

for cell in d['cells']:
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        # Check if it's the right cell
        if any('python part2/train.py' in s for s in source):
            new_source = []
            for s in source:
                # Remove the buggy lines
                if '--checkpoint-dir' not in s:   
                    new_source.append(s)
            
            # Find insertion point
            idx = -1
            for j, s in enumerate(new_source):
                if 'cmd = f"python part2/train.py' in s:
                    idx = j
                    break
                    
            if idx != -1:
                # Add the correct line (without the trailing literal \n inside code syntax)
                correct_append_code = 'cmd += f" --checkpoint-dir \\"{os.path.join(DRIVE_ROOT, \'checkpoints/part2\')}\\" --log-dir \\"{os.path.join(DRIVE_ROOT, \'logs/part2\')}\\""\n'
                new_source.insert(idx+1, correct_append_code)
            
            cell['source'] = new_source

f=open('colab_runner.ipynb', 'w', encoding='utf-8')
json.dump(d, f, indent=1)
f.close()
