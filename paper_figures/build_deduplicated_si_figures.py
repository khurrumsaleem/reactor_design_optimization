"""Compose SI-only panels from archived figures without changing source assets.

Coordinates are in PDF points, measured from the upper-left corner. Retained
panels keep their original drawing contents. Only panel letters and one heading
are replaced. Pass an input directory and a separate output directory.
"""
from pathlib import Path
from io import BytesIO
from copy import deepcopy
import hashlib
import json
import argparse

from pypdf import PdfReader, PdfWriter, PageObject, Transformation
from pypdf.generic import RectangleObject
from reportlab.pdfgen import canvas

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-dir', required=True, type=Path,
                    help='Directory containing the archived source PDFs')
parser.add_argument('--output-dir', required=True, type=Path)
parser.add_argument('--include-attention', action='store_true',
                    help='Also build a historical attention panel omitted from the current SI')
args = parser.parse_args()
SOURCE = args.source_dir.resolve()
OUTPUT = args.output_dir.resolve()
if OUTPUT == SOURCE or SOURCE in OUTPUT.parents:
    raise ValueError('Outputs must be outside the read-only source directory')
OUTPUT.mkdir(parents=True, exist_ok=True)
manifest = {}


def make(name, width, height, panels, masks, labels):
    page = PageObject.create_blank_page(width=width, height=height)
    records = []
    for source, box, dest, scale in panels:
        path = SOURCE / source
        original = PdfReader(path).pages[0]
        part = deepcopy(original)
        x0, y0, x1, y1 = box
        source_height = float(part.mediabox.height)
        part.cropbox = RectangleObject((x0, source_height-y1, x1, source_height-y0))
        part.mediabox = part.cropbox
        part.trimbox = part.cropbox
        dx, dy = dest
        transform = Transformation().translate(-x0, -(source_height-y1)).scale(scale).translate(dx, height-dy-(y1-y0)*scale)
        page.merge_transformed_page(part, transform)
        records.append({'source':source, 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'crop_top_left_points':box, 'destination_top_left_points':dest, 'scale':scale})
    buffer = BytesIO()
    overlay = canvas.Canvas(buffer, pagesize=(width,height))
    for x,y,w,h in masks:
        overlay.setFillColorRGB(1,1,1)
        overlay.rect(x,height-y-h,w,h,stroke=0,fill=1)
    for x,y,text,size,bold in labels:
        overlay.setFillColorRGB(0,0,0)
        overlay.setFont('Helvetica-Bold' if bold else 'Helvetica',size)
        overlay.drawString(x,height-y-size,text)
    overlay.showPage()
    overlay.save()
    page.merge_page(PdfReader(buffer).pages[0])
    writer=PdfWriter();writer.add_page(page)
    path=OUTPUT/(name+'.pdf')
    with path.open('wb') as f:writer.write(f)
    manifest[name]={'width':width,'height':height,'panels':records,
                    'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


# Only the three attention maps, with their original head identifiers.
if args.include_attention:
    make('si_attention_maps',420,177,
         [('workflow.pdf',(16,493,417,657),(9,5),1)],[],[])

# Fitness and multiplication-factor convergence, followed by six selected layouts.
# Inventory trajectories are already shown in main Figure 2. Aggregate endpoints
# are already in main Figures 2–3 and the SI high-statistics table.
make('si_convergence_layouts',519,597,
     [('fig2_discovery.pdf',(0,0,174,166),(0,0),1.47),
      ('fig2_discovery.pdf',(349,0,519,166),(268,0),1.47),
      ('fig2_discovery.pdf',(0,177,519,215),(0,251),1),
      ('fig2_discovery.pdf',(0,216,519,513),(0,300),1)],
     [(0,0,18,18),(268,0,18,18)]+
     [(x-1,y-0.5,9,10.2) for x,y in [(57.6,303.3),(219.8,303.3),(382,303.3),(57.6,449.5),(219.8,449.6),(382,449.6)]],
     [(1.5,0,'a',11,True),(269.5,0,'b',11,True)]+
     [(x,y,label,10,True) for x,y,label in [(57.6,303.3,'c'),(219.8,303.3,'d'),(382,303.3,'e'),(57.6,449.5,'f'),(219.8,449.6,'g'),(382,449.6,'h')]])

# Keep first-passage data and compact seedwise diagnostics. Repeated incumbent
# trajectories and the bar-chart rendering of the same random counts are omitted.
make('si_first_passage_counts',519,361,
     [('ed3_inventory_sampling.pdf',(14,10,239,170),(123,0),1.2),
      ('ed3_inventory_sampling.pdf',(36,443,511,509),(22,200),1),
      ('ed3_inventory_sampling.pdf',(36,524,511,579),(22,287),1)],
     [(123,0,15,18),(22,200,14,14),(22,287,310,14)],
     [(124,0,'a',11,True),(24.4,202.3,'b',11,True),(24.4,287.8,'c',11,True),
      (44.2,289,'Seedwise random-sampling counts',8,False)])

(OUTPUT/'si_deduplicated_figures_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Created '+', '.join(manifest))
