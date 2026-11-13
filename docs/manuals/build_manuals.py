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
OFFICIAL_NAME = 'Sediment Yield, Runoff, and Underground Plumbing'
METADATA_SCHEMA = 'syrup.manual_release_metadata.v1'
_SHA40, _SHA64 = re.compile(r'^[0-9a-f]{40}$'), re.compile(r'^[0-9a-f]{64}$')
# Explicitly versioned closed extension (v2): v1 fields keep their meaning unchanged (the historical C1 subsurface release); v2 adds ONE closed block that names the
# SEPARATE native Phase 1.5 (N15) source package and its Root acceptance. It is not an open key bag, and nothing in it rebinds a C1 identity.
METADATA_SCHEMA_V2 = 'syrup.manual_release_metadata.v2'
N15_FIELDS = {'syrup_package_sha256': _SHA64, 'package_file_count': int, 'source_manifest_sha256': _SHA64, 'maple_native_commit': _SHA40,
              'maple_native_cpu_package_sha256': _SHA64, 'root_s4_acceptance_sha256': _SHA64, 'root_s4_acceptance_status': str,
              'whole_phase_receipt_status': ('pending', 'recorded'), 'whole_phase_receipt_sha256': _SHA64, 'source_commit': _SHA40, 'published': bool}
# Release fields that must be populated for a final build; their pending state is kept in the metadata file as @RELEASE_...@ tokens.
# The production recipe is identified by the SHA-256 of the recipe file (the value pinned as source.recipe_sha256), the receipt by the SHA-256 of its file.
# Explicitly versioned closed extension (v3) = v2 + ONE closed `native_phase2` block naming the NEW Native Phase 2 source (the accepted model-code commit; the document's own
# publication may follow it and is never named here). Nullable fields are pending in a draft and refused by --final. Nothing here rebinds the C1 or N15 identities.
METADATA_SCHEMA_V3 = 'syrup.manual_release_metadata.v3'
P2_FIELDS = {'source_commit': _SHA40, 'source_branch': str, 'syrup_package_sha256': _SHA64, 'package_file_count': int, 'source_manifest_sha256': _SHA64,
             'maple_native_commit': _SHA40, 'maple_native_package_sha256': _SHA64, 'whole_phase_acceptance_sha256': _SHA64, 'published': bool}
P2_NULLABLE = frozenset({'source_commit', 'source_branch', 'syrup_package_sha256', 'package_file_count', 'source_manifest_sha256', 'whole_phase_acceptance_sha256'})
RELEASE_FIELDS = {'syrup_commit': _SHA40, 'syrup_branch': None, 'syrup_water_package_sha256': _SHA64, 'leaf_commit': _SHA40, 'leaf_branch': None,
                  'leaf_package_sha256': _SHA64, 'maple_native_commit': _SHA40, 'maple_native_package_sha256': _SHA64,
                  'masy_commit': _SHA40, 'masy_branch': None, 'masy_package_sha256': _SHA64,
                  'production_cpu_recipe_sha256': _SHA64, 'production_cpu_receipt_sha256': _SHA64,
                  'production_cuda_recipe_sha256': _SHA64, 'production_cuda_receipt_sha256': _SHA64}
# Release state flags (booleans) that a final build must find true.
RELEASE_FLAGS = ('sources_published',)


def load_metadata(path=None, require_final=False):
    """Read the closed release metadata. A draft build tolerates null release fields (rendered as their @RELEASE_...@ token); a final build refuses them.
    The metadata names source identities only; it never contains this documentation's own commit or hash."""
    meta = json.loads((path or HERE / 'release_metadata.json').read_text())
    top = {'schema', 'official_name', 'release_status', 'prepared_date', 'qualification_statement', 'surface_reference', 'subsurface_release', 'pending_tokens'}
    if meta.get('schema') in (METADATA_SCHEMA_V2, METADATA_SCHEMA_V3):
        top = top | {'native_phase15'}
    if meta.get('schema') == METADATA_SCHEMA_V3:
        top = top | {'native_phase2'}
    if set(meta) != top or meta['schema'] not in (METADATA_SCHEMA, METADATA_SCHEMA_V2, METADATA_SCHEMA_V3):
        raise SystemExit('release_metadata.json: unknown or missing top-level keys or wrong schema')
    if 'native_phase15' in meta:
        n15 = meta['native_phase15']
        if set(n15) != set(N15_FIELDS):
            raise SystemExit('release_metadata.json: unknown or missing native_phase15 fields')
        for key, rule in N15_FIELDS.items():
            value = n15[key]
            if key == 'source_commit':
                # Nullable: the FIRST phase commit that holds the accepted code and both manuals cannot name itself, so it is filled in by a later docs-only refresh
                # after the real commit and push; `published` is true only together with that exact 40-hex commit.
                if value is not None and not rule.match(value):
                    raise SystemExit('release_metadata.json: native_phase15.source_commit must be null or 40 lowercase hex')
                if (value is None) == n15['published']:
                    raise SystemExit('release_metadata.json: native_phase15.published must be true exactly when source_commit is set')
            elif key == 'whole_phase_receipt_sha256':
                if (n15['whole_phase_receipt_status'] == 'pending') != (value is None):
                    raise SystemExit('release_metadata.json: whole_phase_receipt_sha256 must be null exactly while the receipt status is pending')
                if value is not None and not rule.match(value):
                    raise SystemExit(f'release_metadata.json: native_phase15.{key} is malformed')
            elif isinstance(rule, tuple):
                if value not in rule:
                    raise SystemExit(f'release_metadata.json: native_phase15.{key} must be one of {rule}')
            elif rule is int or rule is bool or rule is str:
                if type(value) is not rule:
                    raise SystemExit(f'release_metadata.json: native_phase15.{key} has the wrong type')
            elif not isinstance(value, str) or not rule.match(value):
                raise SystemExit(f'release_metadata.json: native_phase15.{key} is malformed')
    if 'native_phase2' in meta:
        p2 = meta['native_phase2']
        if type(p2) is not dict or set(p2) != set(P2_FIELDS):
            raise SystemExit('release_metadata.json: unknown or missing native_phase2 fields')
        for key, rule in P2_FIELDS.items():
            value = p2[key]
            if value is None:
                if key not in P2_NULLABLE:
                    raise SystemExit(f'release_metadata.json: native_phase2.{key} may not be null')
            elif rule is bool or rule is int or rule is str:
                if type(value) is not rule or (rule is int and value < 1) or (rule is str and not value.strip()):
                    raise SystemExit(f'release_metadata.json: native_phase2.{key} is malformed')
            elif not isinstance(value, str) or not rule.match(value):
                raise SystemExit(f'release_metadata.json: native_phase2.{key} is malformed')
        if (p2['source_commit'] is None) == p2['published']:
            raise SystemExit('release_metadata.json: native_phase2.published must be true exactly when source_commit is set')
    if meta['official_name'] != OFFICIAL_NAME:
        raise SystemExit('release_metadata.json: official_name must be ' + OFFICIAL_NAME)
    if meta['release_status'] not in ('draft', 'final'):
        raise SystemExit('release_metadata.json: release_status must be draft or final')
    datetime.date.fromisoformat(meta['prepared_date'])
    if set(meta['surface_reference']) != {'syrup_commit', 'maple_commit'} or set(meta['subsurface_release']) != set(RELEASE_FIELDS) | set(RELEASE_FLAGS):
        raise SystemExit('release_metadata.json: unknown or missing release fields')
    for flag in RELEASE_FLAGS:
        if type(meta['subsurface_release'][flag]) is not bool:
            raise SystemExit(f'release_metadata.json: {flag} must be a Boolean')
    for key, value in ((k, v) for k, v in meta['subsurface_release'].items() if k in RELEASE_FIELDS):
        if value is None:
            if key not in meta['pending_tokens'] or not re.fullmatch(r'@RELEASE_[A-Z0-9_]+@', meta['pending_tokens'][key]):
                raise SystemExit(f'release_metadata.json: {key} is null without an @RELEASE_...@ token')
        elif RELEASE_FIELDS[key] is not None and not RELEASE_FIELDS[key].match(value):
            raise SystemExit(f'release_metadata.json: {key} is malformed')
    for key, value in meta['surface_reference'].items():
        if not _SHA40.match(value):
            raise SystemExit(f'release_metadata.json: surface_reference.{key} is malformed')
    pending = sorted([k for k, v in meta['subsurface_release'].items() if v is None] + [f for f in RELEASE_FLAGS if not meta['subsurface_release'][f]])
    if 'native_phase2' in meta:
        pending += ['native_phase2.' + k for k in sorted(P2_NULLABLE) if meta['native_phase2'][k] is None]
        pending += [] if meta['native_phase2']['published'] else ['native_phase2.published']
    if require_final and (pending or meta['release_status'] != 'final'):
        raise SystemExit('final build refused: release_status must be final and every release field populated and the sources published; pending: ' + ', '.join(pending))
    if meta['release_status'] == 'final' and pending:
        raise SystemExit('release_status final with pending fields: ' + ', '.join(pending))
    return meta


def substitutions(meta):
    rel = meta['subsurface_release']
    out = {'official_name': meta['official_name'], 'release_status': meta['release_status'], 'prepared_date': meta['prepared_date'],
           'qualification_statement': meta['qualification_statement'],
           'surface_syrup_commit': meta['surface_reference']['syrup_commit'], 'surface_maple_commit': meta['surface_reference']['maple_commit']}
    for key, value in rel.items():
        if key in RELEASE_FLAGS:
            out[key] = 'yes' if value else 'no'
        else:
            out[key] = value if value is not None else meta['pending_tokens'][key]
    out['surface_syrup_short'], out['surface_maple_short'] = out['surface_syrup_commit'][:7], out['surface_maple_commit'][:7]
    out['syrup_short'] = out['syrup_commit'][:7]
    n15 = meta.get('native_phase15')
    if n15 is not None:  # only the explicit separate N15 block yields n15_* markers; the C1 markers above are never overwritten
        for key, value in n15.items():
            out['n15_' + key] = 'yes' if value is True else 'no' if value is False else (('not yet recorded' if key == 'source_commit' else 'pending') if value is None else value)
        out['n15_syrup_package_short'] = n15['syrup_package_sha256'][:12]
    p2 = meta.get('native_phase2')
    for key in P2_FIELDS:  # v1/v2 metadata carries no Phase 2 identity: the markers say so instead of failing or inventing one
        value = None if p2 is None else p2[key]
        out['p2_' + key] = ('not recorded (metadata schema before v3)' if p2 is None else 'yes' if value is True else 'no' if value is False else 'pending' if value is None else value)
    out['p2_source_short'] = 'pending' if p2 is None or p2['source_commit'] is None else p2['source_commit'][:7]
    out['p2_package_short'] = 'pending' if p2 is None or p2['syrup_package_sha256'] is None else p2['syrup_package_sha256'][:12]
    return out


def substitute(text, subs):
    """Replace {{key}} markers of the Markdown sources; an unknown marker is an error, never silently kept."""
    def one(m):
        if m[1] not in subs:
            raise SystemExit(f'unknown release marker {{{{{m[1]}}}}}')
        return str(subs[m[1]])
    return re.sub(r'\{\{([a-z0-9_]+)\}\}', one, text)


META = load_metadata()
SUBS = substitutions(META)
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
    if 'native_phase2' in META:  # v3: the new Phase 2 source and the historical CLI release are different identities and are labelled as such
        footer.add_run(f"{META['release_status'].capitalize()} • Native Phase 2 source {SUBS['p2_source_short']} • historical CLI C1 {SUBS['syrup_short']} • prepared {META['prepared_date']}      ")
    else:
        footer.add_run(f"{META['release_status'].capitalize()} • SYRUP {SUBS['syrup_short']} • prepared {META['prepared_date']}                          ")
    field(footer, ' PAGE ')
    footer.add_run(' / ')
    field(footer, ' NUMPAGES ')
    for run in footer.runs:
        run.font.size = Pt(8)
        run.font.color.rgb = RGBColor.from_string(TEAL)
    doc.core_properties.author = 'MAPLE-SYRUP project'
    doc.core_properties.title = 'MAPLE-SYRUP — ' + subtitle
    doc.core_properties.subject = (f"{OFFICIAL_NAME}: documentation of historical CLI release C1 {SUBS['syrup_commit']} and Native Phase 2 source {SUBS['p2_source_commit']} ({META['release_status']})"
                                   if 'native_phase2' in META else f"{OFFICIAL_NAME}: documentation of SYRUP {SUBS['syrup_commit']} ({META['release_status']})")
    doc.core_properties.keywords = 'SYRUP, MAPLE, LEAF, runoff, sediment, soil water, CPU, GPU'
    doc.core_properties.created = datetime.datetime.now(datetime.timezone.utc)
    doc.core_properties.modified = doc.core_properties.created
    doc.core_properties.last_modified_by = 'MAPLE-SYRUP project'
    doc.core_properties.language = 'en-US'
    doc.core_properties.version = (f"P2-{SUBS['p2_source_short']}-C1-{SUBS['syrup_short']}-{META['release_status']}" if 'native_phase2' in META else f"{SUBS['syrup_short']}-{META['release_status']}")
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
    p = doc.add_paragraph(OFFICIAL_NAME)
    p.style = doc.styles['Caption Text']
    doc.add_paragraph()
    for label, value in [('Release status', META['release_status']), (('SYRUP subsurface release (historical CLI C1)' if 'native_phase2' in META else 'SYRUP subsurface release'), SUBS['syrup_commit']),
                         *([('Native Phase 2 source (Python API, not in the historical CLI)', SUBS['p2_source_commit'])] if 'native_phase2' in META else []),
                         ('LEAF', SUBS['leaf_commit']), ('Native MAPLE (pinned, historical)', SUBS['maple_native_commit']),
                         ('Scheduler (MASY)', SUBS['masy_commit']),
                         ('Surface-erosion reference', f"SYRUP {SUBS['surface_syrup_short']}, MAPLE {SUBS['surface_maple_short']}"),
                         ('Prepared', META['prepared_date']), ('Document type', 'Practical reference' if 'User' in subtitle else 'Scientific methods')]:
        p = doc.add_paragraph()
        p.add_run(label + '   ').bold = True
        p.add_run(value)
    doc.add_paragraph(META['qualification_statement'])
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
    markdown(doc, substitute(source, SUBS))
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


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Build the two manuals. Default is a draft build; --final requires populated release metadata.')
    parser.add_argument('--final', action='store_true', help='refuse unless release_metadata.json is final and fully populated')
    args = parser.parse_args(argv)
    global META, SUBS
    META = load_metadata(require_final=args.final)
    SUBS = substitutions(META)
    create_figure()
    guide = (HERE / 'user_guide.md').read_text() + '\n\n' + (HERE / 'cli_reference.md').read_text()
    theory = (HERE / 'model_structure_and_physics.md').read_text()
    records = [build(guide, 'SYRUP_CLI_User_Guide.docx', 'CLI User Guide',
                     'Installation, input preparation, CPU and GPU runs, restart, outputs, plotting and the complete command reference.'),
               build(theory, 'SYRUP_Model_Structure_and_Physics.docx', 'Model Structure and Physics',
                     'Governing relationships, numerical methods and assumptions for a reproducible MAHLERAN-inspired water and sediment model.')]
    (HERE / 'document_build.json').write_text(json.dumps({'release_status': META['release_status'], 'prepared_date': META['prepared_date'], 'metadata_file': 'release_metadata.json',
                                                 'metadata_sha256': hashlib.sha256((HERE / 'release_metadata.json').read_bytes()).hexdigest(),
                                                 'documents': records}, indent=2) + '\n')
    print(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
