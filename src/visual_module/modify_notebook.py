import json

nb_path = '/Users/davidscerri/Library/Mobile Documents/com~apple~CloudDocs/Studies/Masters/Proof-of-Concept/Seeing-Through-Deepfakes/src/visual_module/visual_classifier_eval.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find if there is an empty code cell at the end.
if nb['cells'][-1]['cell_type'] == 'code' and not nb['cells'][-1]['source']:
    # Replace the empty cell
    new_cell = nb['cells'][-1]
else:
    # Append a new cell
    new_cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": []}
    nb['cells'].append(new_cell)

source_code = [
    "from PIL import Image\n",
    "import IPython.display as display_lib\n",
    "\n",
    "def test_custom_image(image_path):\n",
    "    print(f\"Testing image: {image_path}\")\n",
    "    try:\n",
    "        image = Image.open(image_path)\n",
    "    except Exception as e:\n",
    "        print(f\"Error loading image: {e}\")\n",
    "        return\n",
    "        \n",
    "    # Make prediction using the fine-tuned model\n",
    "    result = fine_tuned_classifier.predict(image)\n",
    "    \n",
    "    print(\"\\n--- Results ---\")\n",
    "    print(f\"Prediction: {result['prediction']}\")\n",
    "    print(f\"Confidence: {result['confidence']*100:.2f}%\")\n",
    "    print(f\"Raw Label:  {result['raw_label']}\")\n",
    "    print(f\"All Scores: {result['all_scores']}\")\n",
    "    \n",
    "    # Display the image scaled down for the notebook\n",
    "    image.thumbnail((400, 400))\n",
    "    display_lib.display(image)\n",
    "\n",
    "# Enter your image path here\n",
    "image_path = \"/path/to/your/image.jpg\"\n",
    "# test_custom_image(image_path)\n"
]

new_cell['source'] = source_code

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
