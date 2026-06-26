import json

path = "/Users/davidscerri/Library/Mobile Documents/com~apple~CloudDocs/Studies/Masters/Work Placement/Work-Placement-Assignment/src/integration_pipeline/integration_pipeline.ipynb"

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'verdict = " Probable Deepfake' in line:
                line = '                verdict = " Potential Deepfake"\\n",\n'
            elif 'verdict = " Probably AI-generated' in line:
                line = '                verdict = " Likely AI Generated"\\n",\n'
            elif 'verdict = f" Probably Real' in line:
                line = '            verdict = f" Likely Real (confidence: {1 - fusion_result.ai_probability:.2%})"\\n",\n'
            
            new_source.append(line)
        cell['source'] = new_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook verdicts finally fixed properly.")
