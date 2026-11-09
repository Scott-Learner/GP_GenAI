"""Fetch locked official model files; preserve a single verified local copy."""
import concurrent.futures,json,pathlib,urllib.request,os
from ranged_download import fetch_many
PROJECT=pathlib.Path(__file__).resolve().parent
BASE=pathlib.Path('/tmp/dflash-minimal-runtime/models')
REVISIONS=json.loads((PROJECT/'model_revisions.json').read_text())

def main():
    files=[]
    for model,meta in REVISIONS.items():
        with urllib.request.urlopen(f'https://huggingface.co/api/models/{model}/revision/{meta["revision"]}?blobs=true',timeout=60) as r:data=json.load(r)
        folder='target' if model.startswith('Qwen/') else 'draft'
        for f in data['siblings']:
            name=f['rfilename']
            if '/' in name or not name.endswith(('.json','.safetensors','.txt','.model','.jinja')):continue
            files.append({'model':model,'revision':meta['revision'],'file':name,'size':f['size'],'sha256':f.get('lfs',{}).get('sha256'),'dst':str(BASE/folder/name),'url':f'https://huggingface.co/{model}/resolve/{meta["revision"]}/{name}'})
    if os.environ.get('DFLASH_DOWNLOAD_SOURCE')=='modelscope':
        for item in files:
            item['official_url']=item['url']
            item['url']=f"https://modelscope.cn/models/{item['model']}/resolve/master/{item['file']}"
    (PROJECT/'download_manifest.json').write_text(json.dumps(files,indent=2))
    print('Pinned model bytes',sum(f['size'] for f in files),flush=True)
    fetch_many(files,workers=32)
    print('DOWNLOAD COMPLETE',flush=True)
if __name__=='__main__':main()
