import json
f=open('colab_runner.ipynb', encoding='utf-8')
d=json.load(f)
f.close()
# Add a new cell to install dependencies
new_cell = {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install tensorboard\n"
   ]
}
d['cells'].insert(10, new_cell)
f=open('colab_runner.ipynb', 'w', encoding='utf-8')
json.dump(d, f, indent=1)
f.close()
