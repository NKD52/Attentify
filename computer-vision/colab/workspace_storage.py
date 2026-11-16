"""Private asset loading from persistent folders or explicit ZIPs; no network calls."""
import json,shutil,uuid,zipfile
from pathlib import Path
from core import digest,assemble

REGISTER=Path(__file__).with_name('model-register.json')
PARTS=['scb-refined.part00','scb-refined.part01','scb-refined.part02']

def ensure_head_assets(source,destination):
    """Head weights are separate from the existing immutable action pack."""
    register=json.loads(REGISTER.read_text());spec=register['head-6drepnet-v1']
    source=Path(source).expanduser();destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    target=destination/spec['filename']
    if target.is_file() and digest(target)==spec['sha256']:return target
    if source.is_file() and source.suffix.lower()=='.pth':
        if digest(source)!=spec['sha256']:raise ValueError('Head checkpoint checksum mismatch')
        temp=destination/(spec['filename']+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with source.open('rb') as inp,temp.open('wb') as out:shutil.copyfileobj(inp,out)
            temp.replace(target)
        finally:temp.unlink(missing_ok=True)
        return target
    if source.is_file():
        with zipfile.ZipFile(source) as archive:
            def member(name,optional=False):
                matches=[n for n in archive.namelist() if n==name or n.endswith('/assets/'+name)]
                if not matches and optional:return None
                if len(matches)!=1:raise FileNotFoundError('Head asset missing/ambiguous in ZIP: '+name+'; set HEAD_MODEL_SOURCE to its separate folder or disable head pose')
                if archive.getinfo(matches[0]).file_size>160*1024*1024:raise ValueError('Unexpected head asset size')
                return matches[0]
            direct_member=member(spec['filename'],True)
            entries=[{'name':spec['filename'],'sha256':spec['sha256']}] if direct_member else spec['parts']
            for part in entries:
                name=part['name']
                if Path(name).name!=name:raise ValueError('Unexpected head asset path')
                dest=destination/name
                if dest.is_file() and digest(dest)==part['sha256']:continue
                item=member(name);temp=destination/(name+'.'+uuid.uuid4().hex+'.tmp')
                try:
                    with archive.open(item) as inp,temp.open('wb') as out:shutil.copyfileobj(inp,out)
                    if digest(temp)!=part['sha256']:raise ValueError('Head asset checksum mismatch: '+name)
                    temp.replace(dest)
                finally:temp.unlink(missing_ok=True)
        return target if direct_member else assemble(destination,spec,target)
    if not source.is_dir():raise FileNotFoundError('Head model source is missing')
    direct=source/spec['filename']
    names=[{'name':spec['filename'],'sha256':spec['sha256']}] if direct.is_file() else spec['parts']
    for part in names:
        name=part['name']
        if Path(name).name!=name:raise ValueError('Unexpected head asset path')
        path=source/name;dest=destination/name
        if dest.is_file() and digest(dest)==part['sha256']:continue
        if not path.is_file():raise FileNotFoundError('Head asset missing from model folder: '+name)
        temp=destination/(name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with path.open('rb') as inp,temp.open('wb') as out:shutil.copyfileobj(inp,out)
            if digest(temp)!=part['sha256']:raise ValueError('Head asset checksum mismatch: '+name)
            temp.replace(dest)
        finally:temp.unlink(missing_ok=True)
    return target if direct.is_file() else assemble(destination,spec,target)

def resolve_pack(source,destination,register):
    """A persistent folder may contain one ZIP or verified transport segments."""
    spec=register.get('asset_pack')
    if not source.is_dir() or not spec:return source
    name=spec['filename']
    if Path(name).name!=name:raise ValueError('Invalid pack name')
    if (source/name).is_file():return source/name
    if not spec.get('parts') or not (source/spec['parts'][0]['name']).is_file():return source
    target=destination/name
    if target.exists() and digest(target)==spec['sha256']:return target
    temp=destination/(name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        with temp.open('wb') as out:
            for part in spec['parts']:
                partname=part['name']
                if Path(partname).name!=partname:raise ValueError('Invalid pack part name')
                path=source/partname
                if not path.is_file():raise FileNotFoundError('Missing private model part: '+partname)
                if digest(path)!=part['sha256']:raise ValueError('Model pack part checksum mismatch: '+partname)
                with path.open('rb') as inp:shutil.copyfileobj(inp,out)
        if digest(temp)!=spec['sha256']:raise ValueError('Assembled model pack checksum mismatch')
        temp.replace(target)
    finally:temp.unlink(missing_ok=True)
    return target

def ensure_assets(source,destination,feature_mode='combined',register=None):
    if feature_mode not in ('faces','actions','combined'):raise ValueError('Choose faces, actions or combined')
    register=register if register is not None else json.loads(REGISTER.read_text())
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    source=Path(source).expanduser()
    wanted=[]
    if feature_mode!='actions':wanted.append('yunet-fp32')
    if feature_mode!='faces':wanted.append('scb-refined-onnx-v1')
    if all((destination/register[k]['filename']).is_file() and digest(destination/register[k]['filename'])==register[k]['sha256'] for k in wanted):return destination
    if not source.exists():raise FileNotFoundError('Model source not found. Set MODEL_SOURCE to a persistent folder or model ZIP')
    source=resolve_pack(source,destination,register)
    archive=zipfile.ZipFile(source) if source.is_file() else None
    try:
        def member(name,optional=False):
            if archive is not None:
                names=[n for n in archive.namelist() if n==name or n.endswith('/assets/'+name)]
                if not names and optional:return None
                if len(names)!=1:raise ValueError('Expected exactly one model asset named '+name)
                info=archive.getinfo(names[0])
                if info.file_size>160*1024*1024:raise ValueError('Unexpected asset size: '+name)
                return names[0]
            paths=[p for p in (source/name,source/'assets'/name) if p.is_file()]
            if not paths and optional:return None
            if len(paths)!=1:raise ValueError('Expected exactly one model asset named '+name)
            return paths[0]

        def copy_verified(name,expected):
            target=destination/name
            if target.is_file() and digest(target)==expected:return
            item=member(name);temp=destination/(name+'.'+uuid.uuid4().hex+'.tmp')
            try:
                with (archive.open(item) if archive else Path(item).open('rb')) as inp,temp.open('wb') as out:
                    shutil.copyfileobj(inp,out)
                if digest(temp)!=expected:raise ValueError('Model checksum mismatch: '+name)
                temp.replace(target)
            finally:temp.unlink(missing_ok=True)

        for key in wanted:
            spec=register[key];name=spec['filename']
            if (destination/name).is_file() and digest(destination/name)==spec['sha256']:continue
            if member(name,optional=True) is not None:
                copy_verified(name,spec['sha256']);continue
            if key!='scb-refined-onnx-v1':raise FileNotFoundError('Missing '+name)
            item=member('model.json')
            with (archive.open(item) if archive else Path(item).open('rb')) as inp:manifest=json.load(inp)
            if manifest['sha256']!=spec['sha256'] or [p['name'] for p in manifest['parts']]!=PARTS:raise ValueError('Unexpected split-model manifest')
            for p in manifest['parts']:copy_verified(p['name'],p['sha256'])
            assemble(destination,manifest,destination/name)
        return destination
    finally:
        if archive is not None:archive.close()
