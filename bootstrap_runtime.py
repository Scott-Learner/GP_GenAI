"""Create a private venv and COPY the existing PyTorch distribution closure.

The source Python environment is read-only; no shared site-packages, hard links,
or package updates in the source environment. This avoids multi-GB re-downloads.
Run with the host's original Python, not the new venv's Python.
"""
import importlib.metadata as md,json,pathlib,shutil,subprocess,sys,venv
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import canonicalize_name
PROJECT=pathlib.Path(__file__).resolve().parent
RUNTIME=pathlib.Path('/tmp/dflash-minimal-runtime')
ENV=RUNTIME/'venv'
if not (ENV/'pyvenv.cfg').exists():venv.EnvBuilder(with_pip=True).create(ENV)
SITE=ENV/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages'
assert md.version('torch')=='2.13.0+cu130', 'Host torch build changed; restore the recorded build before copying'
queue=[('torch',set())];selected={};processed=set()
while queue:
    name,extras=queue.pop();key=(canonicalize_name(name),tuple(sorted(extras)))
    if key in processed:continue
    processed.add(key);d=md.distribution(name);selected[canonicalize_name(name)]=d
    for raw in d.requires or []:
        r=Requirement(raw)
        if r.marker and not any(r.marker.evaluate({'extra':e}) for e in ['',*extras]):continue
        dependency=md.distribution(r.name)
        assert not r.specifier or dependency.version in r.specifier,(r.name,dependency.version,str(r.specifier))
        queue.append((r.name,set(r.extras)))
provenance=[]
for name,d in selected.items():
    copied=0;size=0
    for f in d.files or []:
        if '..' in f.parts or f.is_absolute():continue
        source=pathlib.Path(d.locate_file(f));dest=SITE/f
        if not source.is_file():continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest);copied+=1;size+=source.stat().st_size
    info={'distribution':name,'version':d.version,'source_metadata':str(d._path),'copied_files':copied,'copied_bytes':size}
    provenance.append(info);print(json.dumps(info),flush=True)
(PROJECT/'runtime_copy_provenance.json').write_text(json.dumps(provenance,indent=2))
print('ISOLATED TORCH COPY COMPLETE',flush=True)
