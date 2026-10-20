"""Build the two versioned Word manuals from the reviewed Markdown sources.

Documentation tooling only. Requires python-docx==1.2.0 and Matplotlib.
Run from any directory; no model, case data, GPU or repository mutation occurs.
"""
from pathlib import Path
import datetime
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.opc.constants import RELATIONSHIP_TYPE as RT

HERE = Path(__file__).resolve().parent
COMMIT = 'd05d8c6c66d9fd2969c007681916254e8d1c2c7c'
NAVY, TEAL, INK = '18344A', '147D92', '243746'


def shade(element, color):
    shading = OxmlElement('w:shd')
    shading.set(qn('w:fill'), color)
    element.append(shading)


def field(paragraph, instruction):
    r = OxmlElement('w:r')
    fld = OxmlElement('w:fldChar')
    fld.set(qn('w:fldCharType'), 'begin')
    r.append(fld)
    paragraph._p.append(r)
    r = OxmlElement('w:r')
    text = OxmlElement('w:instrText')
    text.set(qn('xml:space'), 'preserve')
    text.text = instruction
    r.append(text)
    paragraph._p.append(r)
    for kind in ('separate', 'end'):
        r = OxmlElement('w:r')
        f = OxmlElement('w:fldChar')
        f.set(qn('w:fldCharType'), kind)
        r.append(f)
        paragraph._p.append(r)


def hyperlink(p, label, target=None, anchor=None):
    h = OxmlElement('w:hyperlink')
    if anchor:
        h.set(qn('w:anchor'), anchor)
    else:
        h.set(qn('r:id'), p.part.relate_to(target, RT.HYPERLINK, is_external=True))
    r = OxmlElement('w:r')
    pr = OxmlElement('w:rPr')
    c = OxmlElement('w:color')
    c.set(qn('w:val'), TEAL)
    pr.append(c)
    r.append(pr)
    t = OxmlElement('w:t')
    t.text = label
    r.append(t)
    h.append(r)
    p._p.append(h)


INLINE = re.compile(r'(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^\s)]+\)|https?://[^\s<>]+)')


def inline(p, text):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            p.add_run(part[2:-2]).bold = True
        elif part.startswith('`') and part.endswith('`'):
            r = p.add_run(part[1:-1])
            r.font.name = 'Consolas'
            r.font.size = Pt(7.5 if re.fullmatch(r'[0-9a-f]{64}', part[1:-1]) else 9)
        elif part.startswith('[') and '](' in part:
            m = re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)', part)
            hyperlink(p, m[1], m[2])
        elif part.startswith(('https://', 'http://')):
            hyperlink(p, part.rstrip('.,;'), part.rstrip('.,;'))
        else:
            p.add_run(part)


def styles(doc, subtitle):
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.top_margin = sec.bottom_margin = Inches(0.7)
    sec.left_margin = sec.right_margin = Inches(0.75)
    sec.header_distance = sec.footer_distance = Inches(0.3)
    sec.different_first_page_header_footer = True
    normal = doc.styles['Normal']
    normal.font.name = 'Calibri'
    normal.font.size = Pt(10.5)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.line_spacing = 1.12
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.widow_control = True
    for level, size in ((1, 16), (2, 12.5), (3, 11)):
        s = doc.styles[f'Heading {level}']
        s.font.name = 'Calibri'
        s.font.size = Pt(size)
        s.font.color.rgb = RGBColor.from_string(NAVY if level == 1 else TEAL)
        s.font.bold = True
        s.paragraph_format.keep_with_next = True
        s.paragraph_format.space_before = Pt(14 if level == 1 else 10)
        s.paragraph_format.space_after = Pt(5)
    for name, font, size in [('Code Block', 'Consolas', 8.3), ('Equation', 'Cambria Math', 10.5),
                              ('Table Text', 'Calibri', 8.5), ('Caption Text', 'Calibri', 9)]:
        s = doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        s.base_style = normal
        s.font.name, s.font.size = font, Pt(size)
        s.paragraph_format.line_spacing = 1.02 if name in ('Code Block', 'Table Text') else 1.1
        s.paragraph_format.space_after = Pt(3 if name == 'Table Text' else 6)
    doc.styles['Caption Text'].font.color.rgb = RGBColor.from_string(TEAL)
    header = sec.header.paragraphs[0]
    header.text = 'MAPLE-SYRUP  |  ' + subtitle
    header.style = doc.styles['Caption Text']
    footer = sec.footer.paragraphs[0]
    footer.paragraph_format.space_before = Pt(3)
    footer.add_run('Source d05d8c6 • 20 October 2026                                      ')
    field(footer, ' PAGE ')
    footer.add_run(' / ')
    field(footer, ' NUMPAGES ')
    for run in footer.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor.from_string(TEAL)
    doc.core_properties.author = 'MAPLE-SYRUP project'
    doc.core_properties.title = 'MAPLE-SYRUP — ' + subtitle
    doc.core_properties.subject = 'Documentation of committed SYRUP ' + COMMIT
    doc.core_properties.keywords = 'SYRUP, MAPLE, runoff, sediment, CPU, GPU'
    doc.core_properties.created = datetime.datetime.now(datetime.timezone.utc)
    doc.core_properties.modified = doc.core_properties.created
    doc.core_properties.last_modified_by = 'MAPLE-SYRUP project'
    doc.core_properties.language = 'en-US'
    doc.core_properties.version = 'd05d8c6'
    update = OxmlElement('w:updateFields')
    update.set(qn('w:val'), 'true')
    doc.settings.element.append(update)


def cover(doc, subtitle, description):
    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run('MAPLE-SYRUP')
    r.font.name, r.font.size = 'Calibri', Pt(34)
    r.font.bold = True
    r.font.color.rgb = RGBColor.from_string(NAVY)
    p = doc.add_paragraph()
    r = p.add_run(subtitle)
    r.font.size = Pt(22)
    r.font.color.rgb = RGBColor.from_string(TEAL)
    p.paragraph_format.space_after = Pt(20)
    doc.add_paragraph(description)
    p = doc.add_paragraph('Sediment Yield, Runoff, and Uptake by Plants')
    p.style = doc.styles['Caption Text']
    doc.add_paragraph()
    for label, value in [('Model version', 'SYRUP d05d8c6'), ('Qualified MAPLE', '39bcf79'),
                         ('Prepared', '20 October 2026'), ('Document type', 'Practical reference' if 'User' in subtitle else 'Scientific methods')]:
        p = doc.add_paragraph()
        p.add_run(label + '   ').bold = True
        p.add_run(value)
    doc.add_paragraph('The acronym names future plant uptake; growth and nutrient/ecohydrological feedbacks are outside the implemented water-model scope.')
    doc.add_page_break()


def rows(line):
    return [x.strip().replace('\\|', '|') for x in re.split(r'(?<!\\)\|', line.strip().strip('|'))]


def table(doc, lines):
    head = rows(lines[0])
    data = [head] + [rows(line) for line in lines[2:]]
    tb = doc.add_table(rows=0, cols=len(head))
    tb.style = 'Light Shading Accent 1'
    tb.autofit = False
    widths = ([1.65, 1.45, 3.9] if len(head) == 3 else
              [1.45, 1.8, 2.4, 1.35] if len(head) == 4 else [7 / len(head)] * len(head))
    for col, width in zip(tb.columns, widths):
        col.width = Inches(width)
    for i, row in enumerate(data):
        cells = tb.add_row().cells
        for c, width in zip(cells, widths):
            c.width = Inches(width)
        pr = cells[0]._tc.getparent().get_or_add_trPr()
        cant = OxmlElement('w:cantSplit')
        pr.append(cant)
        if i == 0:
            repeat = OxmlElement('w:tblHeader')
            pr.append(repeat)
        for cell, content in zip(cells, row):
            p = cell.paragraphs[0]
            p.style = doc.styles['Table Text']
            inline(p, content)
            if i == 0:
                shade(cell._tc.get_or_add_tcPr(), NAVY)
                for run in p.runs:
                    run.bold = True
                    run.font.color.rgb = RGBColor(255, 255, 255)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_equation(doc, lines):
    for line in lines:
        if not line.strip():
            continue
        p = doc.add_paragraph(style='Equation')
        math = OxmlElement('m:oMath')
        r = OxmlElement('m:r')
        txt = OxmlElement('m:t')
        txt.set(qn('xml:space'), 'preserve')
        txt.text = line
        r.append(txt)
        math.append(r)
        p._p.append(math)


def markdown(doc, text, figures=True):
    lines = text.splitlines()
    tops = [(i, l[3:].replace('`', '')) for i, l in enumerate(lines) if l.startswith('## ')]
    p = doc.add_paragraph('Contents', style='Heading 1')
    for index, (_, title) in enumerate(tops):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        hyperlink(p, title, anchor=f'section_{index}')
    doc.add_page_break()
    top_id = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.strip() == '---' or line.startswith('# '):
            i += 1
            continue
        if line.startswith('```'):
            code = []
            i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i])
                i += 1
            p = doc.add_paragraph('\n'.join(code), style='Code Block')
            shade(p._p.get_or_add_pPr(), 'F0F4F7')
            p.paragraph_format.left_indent = p.paragraph_format.right_indent = Inches(0.08)
            p.paragraph_format.keep_together = len(code) <= 12
            i += 1
            continue
        if line.strip() == '$$':
            block = []
            i += 1
            while i < len(lines) and lines[i].strip() != '$$':
                block.append(lines[i])
                i += 1
            add_equation(doc, block)
            i += 1
            continue
        if line.startswith('|'):
            block = []
            while i < len(lines) and lines[i].startswith('|'):
                block.append(lines[i])
                i += 1
            table(doc, block)
            continue
        match = re.match(r'^(#{2,4}) (.+)$', line)
        if match:
            level = len(match[1]) - 1
            p = doc.add_paragraph(style=f'Heading {level}')
            inline(p, match[2])
            if level == 1:
                start = OxmlElement('w:bookmarkStart')
                start.set(qn('w:id'), str(top_id))
                start.set(qn('w:name'), f'section_{top_id}')
                end = OxmlElement('w:bookmarkEnd')
                end.set(qn('w:id'), str(top_id))
                p._p.insert(1 if p._p.pPr is not None else 0, start)
                p._p.append(end)
                top_id += 1
            i += 1
            continue
        image_match = re.match(r'^!\[([^\]]*)\]\(([^)]+)\)$', line)
        if image_match and figures:
            p = doc.add_paragraph()
            p.add_run().add_picture(str(HERE / image_match[2]), width=Inches(6.8))
            doc.add_paragraph(image_match[1], style='Caption Text')
            i += 1
            continue
        bullet = re.match(r'^[-*] (.+)$', line)
        numbered = re.match(r'^\d+\. (.+)$', line)
        if bullet or numbered:
            # Numbered sources/procedures retain their literal source numbering.
            p = doc.add_paragraph(style='List Bullet' if bullet else 'Normal')
            if numbered:
                p.paragraph_format.left_indent = Inches(0.16)
                p.paragraph_format.first_line_indent = Inches(-0.16)
            inline(p, bullet[1] if bullet else line)
            i += 1
            continue
        block = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r'^(#|\||```|\$\$|!\[|[-*] |\d+\. )', lines[i]):
            block.append(lines[i])
            i += 1
        inline(doc.add_paragraph(), ' '.join(block))


def create_figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    assets = HERE / 'assets'
    assets.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.4), layout='constrained')
    ax.set(xlim=(0, 10), ylim=(0, 4.4))
    ax.axis('off')
    boxes = [(0.15, 2.7, 'Rainfall + soil column\nIntake, drainage, return'),
             (3.5, 2.7, 'Water movement\nDirected graph OR pond faces'),
             (6.85, 2.7, 'Detachment demand\nCurrent surface + wet laws'),
             (6.85, 0.8, 'Native MAPLE pickup\nActual available grain supply'),
             (3.5, 0.8, 'Distance + speed\nTravel, deposition, export'),
             (0.15, 0.8, 'Native MAPLE bed\nExchange, accounting, net gate')]
    for x, y, label in boxes:
        box = FancyBboxPatch((x, y), 3, 0.95, boxstyle='round,pad=0.08',
                             facecolor='#EDF5F7', edgecolor='#147D92', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 1.5, y + 0.48, label, ha='center', va='center', fontsize=12, color='#18344A')
    for start, end in [((3.2, 3.17), (3.4, 3.17)), ((6.55, 3.17), (6.75, 3.17)),
                       ((8.35, 2.6), (8.35, 1.9)), ((6.75, 1.27), (6.6, 1.27)),
                       ((3.4, 1.27), (3.25, 1.27)), ((1.65, 1.85), (1.65, 2.55))]:
        ax.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle': '-|>', 'lw': 1.6, 'color': '#147D92'})
    ax.text(5, 0.17, 'One accepted step • failed candidates leave the accepted landscape unchanged',
            ha='center', va='center', fontsize=11, color='#18344A')
    fig.savefig(assets / 'model_flow.png', dpi=180)
    plt.close(fig)


def build(source, filename, subtitle, description):
    doc = Document()
    styles(doc, subtitle)
    cover(doc, subtitle, description)
    markdown(doc, source)
    destination = HERE / filename
    doc.save(destination)
    # Drop stale layout counters and authoring-application claims from the default template.
    # Word calculates pagination when opened; no layout engine was used here.
    ns = 'http://schemas.openxmlformats.org/officeDocument/2006/extended-properties'
    with zipfile.ZipFile(destination) as archive:
        members = [(info, archive.read(info.filename)) for info in archive.infolist()]
    for i, (info, data) in enumerate(members):
        if info.filename != 'docProps/app.xml':
            continue
        tree = ET.fromstring(data)
        for child in list(tree):
            if child.tag.rsplit('}', 1)[-1] in ('Template', 'Pages', 'Words', 'Characters', 'Lines', 'Paragraphs',
                                               'CharactersWithSpaces', 'HeadingPairs', 'TitlesOfParts'):
                tree.remove(child)
        app = tree.find('{' + ns + '}Application')
        app.text = 'MAPLE-SYRUP documentation (python-docx 1.2.0)'
        members[i] = info, ET.tostring(tree, encoding='utf-8', xml_declaration=True)
    temporary = destination.with_suffix('.docx.partial')
    with zipfile.ZipFile(temporary, 'w') as archive:
        for info, data in members:
            archive.writestr(info, data)
    temporary.replace(destination)
    return {'file': filename, 'bytes': (HERE / filename).stat().st_size,
            'sha256': hashlib.sha256((HERE / filename).read_bytes()).hexdigest(),
            'paragraphs': len(doc.paragraphs), 'tables': len(doc.tables),
            'headings': sum(p.style.name.startswith('Heading') for p in doc.paragraphs)}


def main():
    create_figure()
    guide = (HERE / 'user_guide.md').read_text() + '\n\n' + (HERE / 'cli_reference.md').read_text()
    theory = (HERE / 'model_structure_and_physics.md').read_text()
    records = [build(guide, 'SYRUP_CLI_User_Guide.docx', 'CLI User Guide',
                     'Installation, input preparation, CPU and GPU runs, restart, outputs, plotting and the complete command reference.'),
               build(theory, 'SYRUP_Model_Structure_and_Physics.docx', 'Model Structure and Physics',
                     'Governing relationships, numerical methods and assumptions for a reproducible MAHLERAN-inspired water and sediment model.')]
    (HERE / 'document_build.json').write_text(json.dumps({'source_commit': COMMIT, 'documents': records}, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
