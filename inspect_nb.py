import json, sys

with open('Code.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

with open('nb_content.txt', 'w', encoding='utf-8') as out:
    out.write(f"=== TONG SO CELL: {len(cells)} ===\n\n")
    for i, c in enumerate(cells):
        src = ''.join(c['source'])
        out.write(f"\n{'='*60}\n")
        out.write(f">>> CELL {i:02d} [{c['cell_type'].upper()}]\n")
        out.write('='*60 + '\n')
        out.write(src[:3000] + '\n')

print("Done! Saved to nb_content.txt")
