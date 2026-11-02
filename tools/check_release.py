"""Check publication files without importing GPU dependencies."""
import argparse
import ast
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict-publication',action='store_true')
    args=parser.parse_args()
    issues=[];count=0
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or any(p in {'.git','__pycache__','.pytest_cache','.venv','outputs'} for p in path.relative_to(ROOT).parts):continue
        if path.suffix.lower() not in {'.py','.md','.txt','.json','.cff','.sh','.yml','.yaml'}:continue
        count+=1;text=path.read_text(encoding='utf-8');relative=str(path.relative_to(ROOT))
        for pattern in [r'/mnt/(?:bn|shenzhen|shaoguan)',r'/Users/[^\s]+',r'hf_[a-zA-Z0-9]{20,}']:
            if path.resolve()!=Path(__file__).resolve() and re.search(pattern,text):issues.append(f'{relative}: local infrastructure path or token pattern')
        for address in (re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b',text) if path.suffix in {'.py','.sh'} else []):
            if address not in {'127.0.0.1','0.0.0.0'}:issues.append(f'{relative}: hard-coded network address')
        if path.suffix=='.py':
            try:tree=ast.parse(text,filename=relative)
            except SyntaxError as error:issues.append(f'{relative}: {error}');continue
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and ((isinstance(node.func,ast.Name) and node.func.id=='breakpoint') or
                    (isinstance(node.func,ast.Attribute) and node.func.attr=='set_trace')):
                    issues.append(f'{relative}:{node.lineno}: interactive debugger call')
                if isinstance(node,ast.ImportFrom) and node.level and node.module:
                    base=path.parent
                    for _ in range(node.level-1):base=base.parent
                    target=base.joinpath(*node.module.split('.'))
                    if not target.is_dir() and not target.with_suffix('.py').is_file():
                        issues.append(f'{relative}:{node.lineno}: missing relative import {node.module}')
        if path.suffix=='.md':
            for target in re.findall(r'\]\(([^)]+)\)',text):
                target=target.split('#',1)[0]
                if not target or '://' in target or target.startswith('mailto:'):continue
                if not (path.parent/target).exists():issues.append(f'{relative}: missing local link {target}')
    blockers=[]
    if not (ROOT/'LICENSE').is_file():blockers.append('Missing code license')
    if (ROOT/'licenses/SnapMoGen.txt').is_file():blockers.append('SnapMoGen-derived components retain non-commercial research terms; Apache-2.0 covers original contributions only')
    result=dict(files_checked=count,technical_issues=issues,publication_blockers=blockers)
    print(json.dumps(result,indent=2))
    return 1 if issues else 2 if args.strict_publication and blockers else 0


if __name__=='__main__':sys.exit(main())
