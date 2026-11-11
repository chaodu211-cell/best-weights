#!/usr/bin/env python3
"""B 核验：改动前后完整生成页面，只允许 validity.retest 新增。登记写入隔离到临时副本。"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
PROTECTED = ('_红点预登记.csv', '_漏报预登记.csv', '_信号快照.csv')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=('before', 'after'))
    args = ap.parse_args()
    manifest = HERE / 'B核验记录.json'
    if args.phase == 'before':
        work = Path(tempfile.mkdtemp(prefix='retest-b-'))
        m = {'work': str(work), 'generated_at': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
             'protected_before': {n: sha(ROOT/n) for n in PROTECTED},
             'original_outputs': {n: sha(ROOT/n) for n in ('data_alt.json', 'dashboard_alt.html')},
             'sources_before': {n: sha(ROOT/n) for n in ('engine.py', 'alt_engine.py', 'make_alt_page.py', 'alt_tpl.html', '说明.md', 'test_engine.py', '.github/workflows/daily.yml')}}
        for n in PROTECTED:
            shutil.copy2(ROOT/n, work/n)
    else:
        m = json.loads(manifest.read_text())
        work = Path(m['work'])
    import engine as E
    import make_alt_page as M
    class FixedDatetime:
        @staticmethod
        def utcnow():
            return datetime.strptime(m['generated_at'], '%Y-%m-%dT%H:%M:%SZ')
    E.datetime = FixedDatetime
    os.environ.pop('ALT_SNAPSHOT', None)
    M.PREREG_CSV = str(work/'_红点预登记.csv')
    M.RECALL_CSV = str(work/'_漏报预登记.csv')
    M.SNAP_CSV = str(work/'_信号快照.csv')
    if hasattr(M, 'RETEST_CSV'):
        M.RETEST_CSV = str(work/'_二次顶预登记.csv')
    phase_dir = work/args.phase
    phase_dir.mkdir(exist_ok=True)
    sys.argv = ['make_alt_page.py', '--data-out', str(phase_dir/'data_alt.json'), '--out', str(phase_dir/'dashboard_alt.html')]
    M.main()
    d = json.loads((phase_dir/'data_alt.json').read_text())
    m[args.phase] = {'sha256': sha(phase_dir/'data_alt.json'), 'as_of': d['as_of'], 'days': len(d['series']['dates'])}
    if args.phase == 'after':
        before = json.loads((work/'before/data_alt.json').read_text())
        after = json.loads((work/'after/data_alt.json').read_text())
        added = after['validity'].pop('retest')
        assert before == after, '移除 validity.retest 后完整 JSON 不一致'
        m['comparison'] = {'equal': True, 'added_paths': ['validity.retest'], 'days': len(d['series']['dates']),
                           'all_existing_fields_equal': True, 'generated_at_compared': True}
        m['retest'] = {'periods': added['periods'], 'current': added['current'], 'trigger_days': [day for day, on in zip(d['series']['dates'], added['series']['trigger']) if on]}
        assert m['protected_before'] == {n: sha(ROOT/n) for n in PROTECTED}
        m['protected_after'] = {n: sha(ROOT/n) for n in PROTECTED}
        # 成功比较后才交付生成产物；生产三份 CSV 从未被写入。
        shutil.copy2(phase_dir/'data_alt.json', ROOT/'data_alt.json')
        shutil.copy2(phase_dir/'dashboard_alt.html', ROOT/'dashboard_alt.html')
        m['outputs_after'] = {n: sha(ROOT/n) for n in ('data_alt.json', 'dashboard_alt.html')}
    manifest.write_text(json.dumps(m, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(m.get('comparison', m[args.phase]), ensure_ascii=False))

if __name__ == '__main__':
    main()
