"""Render all report pages and run final non-training artifact checks.

Optional QA dependency: PyMuPDF, Pillow. Run after build.ps1.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess

import pymupdf
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
QA = HERE / 'qa'
QA.mkdir(exist_ok=True)
doc = pymupdf.open(HERE / 'report.pdf')
assert 6 <= len(doc) <= 8, f'Unexpected page count: {len(doc)}'
text = '\n'.join(page.get_text() for page in doc)
assert '\ufffd' not in text and '\x00' not in text, 'Replacement or null characters in extracted PDF text'
assert len(re.findall(r'[\u4e00-\u9fff]',text)) > 4000, 'Chinese text extraction is incomplete'
for token in ['99.05', '97.64', '98.35', '96.27']:
    assert token in text, f'Missing audited result {token}'
assert '??' not in text, 'Unresolved cross reference in PDF'
assert all(name in text for name in ['Devlin','Mikolov','Pennington','Wang']), 'Missing references'
log = (HERE/'report.log').read_text(encoding='utf-8',errors='replace')
for bad in ['Overfull', 'Missing character', 'undefined', 'LaTeX Error', 'Package fontspec Error']:
    assert bad not in log, f'Unresolved LaTeX issue: {bad}'
font_info = subprocess.run(['pdffonts',str(HERE/'report.pdf')],capture_output=True,text=True,check=True).stdout
font_rows = font_info.splitlines()[2:]
assert font_rows and all(re.search(r'yes\s+yes\s+yes\s+\d+\s+\d+\s*$',line) for line in font_rows), 'Fonts must be embedded, subset and have Unicode maps'
source_audit = json.loads((HERE/'audit.json').read_text(encoding='utf-8'))
missing_sources = []
for name,expected in source_audit['source_sha256'].items():
    source = HERE.parent/name
    if not source.exists():
        missing_sources.append(name)
        continue
    assert hashlib.sha256(source.read_bytes()).hexdigest()==expected, f'Source file changed: {name}'
for entry in source_audit['files']:
    assert hashlib.sha256((HERE.parent/entry['path']).read_bytes()).hexdigest()==entry['sha256'], f'Result file changed: {entry["path"]}'
review = {'pages':len(doc),'cjk_characters':len(re.findall(r'[\u4e00-\u9fff]',text)),
    'all_fonts_embedded':True,'unresolved_references':False,'overfull_boxes':False,
    'present_original_source_hashes_unchanged':True,
    'missing_original_sources':missing_sources, 'page_bounds':[]}
canvas = Image.new('RGB',(1350,1920),'#e5e7eb')
draw = ImageDraw.Draw(canvas)
for i,page in enumerate(doc):
    page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(QA/f'page-{i+1:02d}.png')
    canvas.paste(Image.open(QA/f'page-{i+1:02d}.png').resize((430,608)),((i%3)*450+10,(i//3)*640+22))
    draw.text(((i%3)*450+10,(i//3)*640+5),f'Page {i+1}',fill='black')
    spans = [span for block in page.get_text('dict')['blocks'] if 'lines' in block
             for line in block['lines'] for span in line['spans'] if span['text'].strip()]
    bounds = [min(s['bbox'][0] for s in spans),min(s['bbox'][1] for s in spans),
              max(s['bbox'][2] for s in spans),max(s['bbox'][3] for s in spans)]
    # xeCJK compresses punctuation advance while PDF text boxes retain full
    # em width. Test actual body characters separately from punctuation boxes
    # and the official ACL page-number footer (baseline near 803 pt).
    punctuation = set('，。、；：？！）》】」』,.?!;: ')
    chars = [ch for block in page.get_text('rawdict')['blocks'] if 'lines' in block
             for line in block['lines'] for span in line['spans'] for ch in span['chars']
             if ch['c'] not in punctuation and ch['bbox'][1]<780]
    assert all(ch['bbox'][0]>=68 and ch['bbox'][2]<=527 and ch['bbox'][1]>=55 and ch['bbox'][3]<=780 for ch in chars), f'Body character outside ACL page area: page {i+1}'
    assert bounds[3]<=815, f'Footer outside page area: page {i+1}'
    review['page_bounds'].append(bounds)
canvas.save(QA/'contact.png')
for stale in QA.glob('page-*.png'):
    match = re.fullmatch(r'page-(\d+)\.png',stale.name)
    if match and int(match.group(1))>len(doc):
        stale.unlink()
(QA/'report_text.txt').write_text(text,encoding='utf-8')
(QA/'font_inventory.txt').write_text(font_info,encoding='utf-8')
(QA/'verification.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(review,ensure_ascii=False,indent=2))
