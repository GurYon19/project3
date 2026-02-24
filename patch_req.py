import json
f=open('colab_runner.ipynb', encoding='utf-8')
d=json.load(f)
f.close()

# Find the cell we inserted that just has "!pip install tensorboard" and replace it
found = False
for cell in d['cells']:
    if cell.get('cell_type') == 'code':
        source = cell.get('source', [])
        if len(source) > 0 and 'pip install tensorboard' in source[0]:
            cell['source'] = [
                "%cd /content/{REPO_NAME}\n",
                "!pip install -r requirements.txt\n"
            ]
            found = True
            break

if not found:
    # Append it right before the last cells if not found
    print("Could not find the pip install tensorboard cell, inserting anew.")
    new_cell = {
       "cell_type": "code",
       "execution_count": None,
       "metadata": {},
       "outputs": [],
       "source": [
        "%cd /content/{REPO_NAME}\n",
        "!pip install -r requirements.txt\n"
       ]
    }
    d['cells'].insert(10, new_cell)

f=open('colab_runner.ipynb', 'w', encoding='utf-8')
json.dump(d, f, indent=1)
f.close()
