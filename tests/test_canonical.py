"""Unit tests for src/canonical_calls.py (run: python -I tests/test_canonical.py)."""
import importlib.util, os
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location('cc', os.path.join(HERE, '..', 'src', 'canonical_calls.py'))
cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)

def test_reordered_keys_identical():
    a = cc.canon_calls("Tool call: text_editor: {'command': 'view', 'path': '/a'}")[0]
    b = cc.canon_calls("Tool call: text_editor: {'path': '/a', 'command': 'view'}")[0]
    assert a == b == 'Tool call: text_editor: {"command":"view","path":"/a"}'

def test_nested_dicts_sorted_lists_kept():
    out = cc.canon_calls("Tool call: t: {'z': {'b': 1, 'a': [3, {'y': 0, 'x': None}]}, 'a': True}")[0]
    assert out == 'Tool call: t: {"a":true,"z":{"a":[3,{"x":null,"y":0}],"b":1}}'

def test_missing_arguments():
    assert cc.canon_calls("Tool call: bash: {}") == ('Tool call: bash: {}', 0)
    assert cc.canon_calls("Tool call: bash: ") == ('Tool call: bash: {}', 0)

def test_quoted_and_multiline_commands():
    line = "Tool call: bash: {'cmd': 'cat <<\\'EOF\\' > f\\necho \"a|b\" && ls\\nEOF'}"
    out, f = cc.canon_calls(line)
    assert f == 0 and out == 'Tool call: bash: {"cmd":"cat <<\'EOF\' > f\\necho \\"a|b\\" && ls\\nEOF"}'

def test_empty_trajectory():
    assert cc.canon_calls('') == ('', 0)

def test_redaction_preserved():
    # formatter output: str_replace strings already replaced by '[...]' when the editor returned output
    out = cc.canon_calls("Tool call: text_editor: {'command': 'str_replace', 'path': '/p', 'old_str': '[...]', 'new_str': '[...]'}")[0]
    assert out == 'Tool call: text_editor: {"command":"str_replace","new_str":"[...]","old_str":"[...]","path":"/p"}'

def test_planning_dropped_and_failure_placeholder():
    out, f = cc.canon_calls("Tool call: start_phase1_planning: {}\nTool call: bash: {'cmd': 'ls'\nTool call: bash: {'cmd': 'pwd'}")
    assert f == 1 and out == 'Tool call: bash: <UNPARSED>\nTool call: bash: {"cmd":"pwd"}'
    assert cc.strip_planning("Tool call: run_phase_x: {}\nTool call: bash: {'cmd': 'ls'}") == "Tool call: bash: {'cmd': 'ls'}"

for name, fn in list(globals().items()):
    if name.startswith('test_'):
        fn(); print('PASS', name)
