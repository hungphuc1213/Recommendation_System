import json

with open('d:/Bao_cao_de_an/Code.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('d:/Bao_cao_de_an/eda_output.txt', 'w', encoding='utf-8') as out:
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            out.write("--- MARKDOWN ---\n")
            out.write(''.join(cell['source']) + "\n")
        elif cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if "plt" in source or "sns" in source or "info(" in source or "describe(" in source or "print(" in source:
                out.write("--- CODE (EDA RELATED) ---\n")
                out.write(source + "\n")
