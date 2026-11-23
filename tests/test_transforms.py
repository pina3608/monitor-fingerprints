"""Unit tests for the jamming transforms in src/identification.py (run: python -I tests/test_transforms.py).
Documents the known J3 bug: tool-call arguments are serialised as Python dict reprs, so the string-literal mask
replaces whole commands, not only literals inside code."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, '..', 'src', 'identification.py')).read().split("\nhonest = [d for d in docs")[0]
src = src.replace("TEXT, OUT = sys.argv[1], sys.argv[2]", "TEXT, OUT = '', ''")
src = src.replace("docs = [json.loads(l) for f in sorted(glob.glob(f'{TEXT}/*.jsonl.gz')) for l in gzip.open(f, 'rt')]", "docs = []")
ns = {}; exec(src, ns)
j2, j3, j4 = ns['j2'], ns['j3'], ns['j4']

calls = ("Tool call: bash: {'cmd': 'grep -rn \"token\" /app/src  # find secrets'}\n"
         "Tool call: text_editor: {'command': 'view', 'path': '/app/main.py', 'view_range': [1, 40]}\n"
         "Tool call: bash: {'cmd': 'cd /app && pytest -q'}")

def test_j2_strips_own_line_comments_keeps_command():
    out = j2("Tool call: bash: {'cmd': 'ls /app\\n# list files\\ncat x'}")
    assert out == "Tool call: bash: {'cmd': 'ls /app cat x'}"

def test_j2_known_limitation_trailing_comment_kept():
    # A trailing comment on a one-line command is followed by the repr's closing quote, so the
    # end-of-line comment regex does not match. Reported results used J2 as-is; documented, not fixed.
    out = j2(calls)
    assert 'grep -rn' in out and '# find secrets' in out

def test_j3_known_bug_masks_whole_command():
    out = j3(calls)
    # Intended: mask only literals ("token", 40). Actual: the whole quoted cmd value is one string literal.
    assert 'grep' not in out and 'pytest' not in out, 'J3 bug fixed? update docs and this test'
    # Actual behaviour: keys and values are all masked; each call keeps only tool name, structure and numbers.
    assert out == 'Tool call: bash: {<S>: <S>} Tool call: text_editor: {<S>: <S>, <S>: <S>, <S>: [<N>, <N>]} Tool call: bash: {<S>: <S>}'

def test_j4_skeleton():
    assert j4(calls) == 'bash:grep edit:view bash:cd'

for name, fn in list(globals().items()):
    if name.startswith('test_'):
        fn(); print('PASS', name)
