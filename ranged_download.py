"""Bounded concurrent HTTP Range download, resumable, final SHA256 verified."""
import concurrent.futures,hashlib,json,pathlib,shutil,time,urllib.request
CHUNK=32*1024*1024

def fetch_range(task):
    url,path,start,end=task
    path=pathlib.Path(path)
    size=end-start+1
    for attempt in range(5):
        offset=path.stat().st_size if path.exists() else 0
        if offset==size:return
        try:
            request=urllib.request.Request(url,headers={'Range':f'bytes={start+offset}-{end}'})
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({})) if 'modelscope.cn/' in url else urllib.request.build_opener()
            with opener.open(request,timeout=90) as r:
                content_range=r.headers.get('Content-Range','')
                assert r.status==206 and content_range.startswith(f'bytes {start+offset}-'),(r.status,content_range,start+offset)
                with path.open('ab') as f:
                    remaining=size-offset
                    while remaining:
                        b=r.read(min(1024*1024,remaining))
                        if not b:break
                        f.write(b);remaining-=len(b)
            if path.stat().st_size!=size:raise IOError('incomplete range')
            return
        except Exception as e:
            print('range retry',path.name,attempt+1,repr(e),flush=True)
            if attempt==4:raise

def fetch_many(files,workers=32):
    tasks=[];plans=[]
    for item in files:
        dst=pathlib.Path(item['dst']);dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists() and dst.stat().st_size==item['size']:continue
        partial=dst.with_suffix(dst.suffix+'.partial')
        parts=dst.with_suffix(dst.suffix+'.parts');parts.mkdir(exist_ok=True)
        planfile=parts/'plan.json'
        if planfile.exists():plan=json.loads(planfile.read_text())
        else:
            offset=partial.stat().st_size if partial.exists() else 0
            plan={'prefix':offset,'ranges':[(i,min(i+CHUNK,item['size'])-1) for i in range(offset,item['size'],CHUNK)]}
            planfile.write_text(json.dumps(plan))
        for a,b in plan['ranges']:tasks.append((item['url'],str(parts/f'{a}-{b}'),a,b))
        plans.append((item,partial,parts,plan))
    start=time.perf_counter()
    print('range tasks',len(tasks),'workers',workers,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for i,_ in enumerate(pool.map(fetch_range,tasks),1):
            if i%10==0:print('finished ranges',i,'/',len(tasks),'seconds',round(time.perf_counter()-start,1),flush=True)
    for item,partial,parts,plan in plans:
        with partial.open('ab') as out:
            assert out.tell()==plan['prefix']
            for a,b in plan['ranges']:
                with (parts/f'{a}-{b}').open('rb') as src:shutil.copyfileobj(src,out,8*1024*1024)
        assert partial.stat().st_size==item['size']
        h=hashlib.sha256()
        with partial.open('rb') as f:
            while b:=f.read(8*1024*1024):h.update(b)
        assert not item.get('sha256') or h.hexdigest()==item['sha256'],('SHA mismatch',item['dst'],h.hexdigest())
        partial.rename(item['dst']);shutil.rmtree(parts)
        print('verified',item['dst'],h.hexdigest(),flush=True)
