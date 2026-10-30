"""Resumable CPU Dask generator backfill; immutable EOS outputs and WORK proofs.

No simulation or particle selections. Each task uses worker-local scratch;
completion requires two stable full SHA256 readbacks. Run the controller in the
global environment and CMSSW/GFAL in their separate, pinned subprocesses.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback
import uuid


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def digest(path, algorithm='sha256'):
    import zlib
    h = hashlib.sha256() if algorithm == 'sha256' else 1
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            if algorithm == 'sha256':
                h.update(block)
            else:
                h = zlib.adler32(block, h)
    return h.hexdigest() if algorithm == 'sha256' else '%08x' % (h & 0xffffffff)


def validate(path, expected=None, source=None, source_info=None):
    import awkward as ak
    import numpy as np
    import uproot
    from uproot.source.file import MemmapSource
    count = 0
    packed = pruned = 0
    identities = set()
    with uproot.open(path, handler=MemmapSource) as f:
        tree = f['gen/events']
        n = tree.num_entries
        if f['Events'].num_entries != n or (expected is not None and n != expected):
            raise ValueError('Native/view/expected counts differ')
        branches = set(f['Events'].keys(recursive=False))
        if source:
            with uproot.open(source, handler=MemmapSource) as original:
                # Include GEN process products, gen particles/jets/MET, flavour
                # associations and event vertex/time products, not reco rho.
                required = {b for b in original['Events'].keys(recursive=False)
                            if ('gen' in b.lower() and 'fixedGridRho' not in b)
                            or 'PileupSummary' in b or 'LHE' in b}
                if required - branches:
                    raise ValueError('Generator branches dropped: ' + repr(required - branches))
                if expected is None and original['Events'].num_entries != n:
                    raise ValueError('Incomplete source export')
        if source_info:
            if set(source_info['required_branches']) - branches:
                raise ValueError('Remote source generator products dropped')
            if expected is None and source_info['events'] != n:
                raise ValueError('Incomplete remote source export')
        for batch in tree.iterate(step_size=1000, library='ak'):
            keys = list(zip(ak.to_list(batch.run), ak.to_list(batch.lumi), ak.to_list(batch.event)))
            if len(set(keys)) != len(keys) or identities.intersection(keys):
                raise ValueError('Duplicate event ID')
            identities.update(keys)
            if ak.to_list(batch.entry) != list(range(count, count + len(batch))):
                raise ValueError('Source entry order differs')
            for prefix in ['Packed_', 'Pruned_']:
                sizes = ak.num(batch[prefix + 'pdgId'])
                if prefix == 'Packed_': packed += int(ak.sum(sizes))
                else: pruned += int(ak.sum(sizes))
                momentum = ['pt','eta','phi'] if prefix == 'Packed_' else ['px','py','pz','energy']
                for field in momentum + ['mass', 'x_mm', 'y_mm', 'z_mm']:
                    a = batch[prefix + field]
                    if not bool(ak.all(ak.num(a) == sizes)) or not np.all(np.isfinite(ak.to_numpy(ak.flatten(a)))):
                        raise ValueError('Invalid particle fields')
                relations = ['mother', 'lastPruned'] if prefix == 'Packed_' else ['mother', 'daughter']
                for relation in relations:
                    p = prefix + relation
                    offsets = batch[p + 'Offset']
                    states = batch[p + 'State']
                    indices = batch[p + 'Index']
                    if not bool(ak.all(ak.num(offsets) == sizes + 1)):
                        raise ValueError('Graph offsets differ')
                    if not bool(ak.all(offsets[:, 0] == 0)) or not bool(ak.all(offsets[:, -1] == ak.num(indices))):
                        raise ValueError('Graph offsets not closed')
                    if not bool(ak.all(ak.flatten(offsets[:, 1:] - offsets[:, :-1]) >= 0)):
                        raise ValueError('Non-monotonic graph offsets')
                    if not bool(ak.all((states == 0) | (states == 2))):
                        raise ValueError('Unavailable graph references')
                    target_n = ak.num(batch.Pruned_pdgId)
                    if not bool(ak.all((states == 0) | ((indices >= 0) & (indices < target_n)))):
                        raise ValueError('Graph reference outside pruned collection')
            count += len(batch)
        if count != n: raise ValueError('Incomplete basket read')
    return dict(events=count, packed=packed, pruned=pruned, all_view_baskets_read=True,
                required_native_products_present=True, unavailable_links=0)


def runtime(config, directory):
    prefix = json.loads(Path(config['output'], 'runtime-prefix.json').read_text())
    prefix[prefix.index('-w') + 1] = str(directory)
    return [('HEPTOKENS_CMSSW_WORK=' + str(directory)) if v.startswith('HEPTOKENS_CMSSW_WORK=') else v for v in prefix]


def remote_source_info(source):
    """Verify server-reported catalog size/checksum; do not claim a full SHA read.

    ROOT baskets are fetched on demand. The full source Adler32 is reported by
    the authoritative server, whereas output hashes are freshly measured.
    """
    from urllib.parse import urlsplit
    import re
    import uproot
    url = source['uri']
    split = urlsplit(url)
    host = split.scheme + '://' + split.netloc
    path = '/' + split.path.lstrip('/')
    xrdfs = '/work/pixi/global/.pixi/envs/default/bin/xrdfs'
    stat = subprocess.run([xrdfs,host,'stat',path],check=True,capture_output=True,text=True,timeout=120).stdout
    checksum = subprocess.run([xrdfs,host,'query','checksum',path],check=True,capture_output=True,text=True,timeout=120).stdout
    size = re.search(r'\bSize:\s*(\d+)',stat)
    if size is None or int(size.group(1)) != int(source['bytes']):
        raise ValueError('Remote size differs from source catalog')
    if checksum.strip().split()[-1].lower().zfill(8) != source['checksum'].split(':')[-1].lower().zfill(8):
        raise ValueError('Remote catalog Adler32 differs')
    with uproot.open(url,timeout=120) as f:
        required = [b for b in f['Events'].keys(recursive=False)
                    if ('gen' in b.lower() and 'fixedGridRho' not in b) or 'PileupSummary' in b or 'LHE' in b]
        info = dict(events=f['Events'].num_entries,root_uuid=str(f.file.uuid),required_branches=required)
    return dict(info,stat=stat,checksum=checksum,verification='server-reported-size-and-Adler32-match-frozen-catalog; selected ROOT baskets read',full_source_sha256_measured=False)


def reserve_remote(config, key, size, sha256):
    import fcntl
    ledger = Path(config['output'], 'remote-ledger.json')
    with open(str(ledger) + '.lock', 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(ledger.read_text()) if ledger.exists() else {}
        item = dict(bytes=size, sha256=sha256)
        if key in data and data[key] != item:
            raise ValueError('Conflicting immutable remote reservation')
        data[key] = item
        if sum(x['bytes'] for x in data.values()) > config['remote_cap_bytes']:
            raise ValueError('Remote storage budget exhausted; output retained')
        atomic(ledger, data)


def compare_independent_audit(output, original_audit, backfill_audit):
    import gzip
    import math
    import numpy as np
    import uproot
    from uproot.source.file import MemmapSource
    original = json.loads(gzip.open(original_audit, 'rt').read())
    restored = json.loads(gzip.open(backfill_audit, 'rt').read())
    if original['matched'] != restored['matched'] or restored['matched'] != 10:
        raise ValueError('Independent audit event counts differ')
    by_key = {tuple(row['key']):row for row in original['events']}
    with uproot.open(output, handler=MemmapSource) as f:
        arrays = f['gen/events'].arrays(library='ak')
    view_keys = {(int(row.run),int(row.lumi),int(row.event)):row for row in arrays}
    particle_count = 0
    for row in restored['events']:
        before = by_key[tuple(row['key'])]
        for key in ['key','weight','packed','graph']:
            if before[key] != row[key]:
                raise ValueError('Native generator preservation failed: '+key)
        view = view_keys[tuple(row['key'])]
        if float(view.weight) != before['weight']: raise ValueError('View weight differs')
        for collection, prefix in [('packed','Packed_'),('graph','Pruned_')]:
            nodes = before[collection]
            if len(nodes) != len(view[prefix+'pdgId']): raise ValueError('View particles missing')
            for i,node in enumerate(nodes):
                get = lambda name: view[prefix+name][i]
                if int(get('pdgId')) != node['pdg_id'] or int(get('status')) != node['status']:
                    raise ValueError('View particle identity differs')
                # reco::ParticleState::charge() is integer division qx3_/3,
                # so its value cannot recover the fractional charge of quarks.
                q3 = int(get('threeCharge'))
                observed_charge = q3/3 if collection == 'packed' else int(q3/3)
                if observed_charge != node['charge']: raise ValueError('Charge differs')
                flags = sum(int(value)<<j for j,value in enumerate(node['flags'][name] for name in [
                    'isPrompt','isDecayedLeptonHadron','isTauDecayProduct','isPromptTauDecayProduct',
                    'isDirectTauDecayProduct','isDirectPromptTauDecayProduct','isDirectHadronDecayProduct',
                    'isHardProcess','fromHardProcess','isHardProcessTauDecayProduct',
                    'isDirectHardProcessTauDecayProduct','fromHardProcessBeforeFSR','isFirstCopy','isLastCopy','isLastCopyBeforeFSR']))
                if int(get('statusFlags')) != flags: raise ValueError('Status flags differ')
                values = node['values']
                xyzm = [float(get(name)) for name in ['mass','x_mm','y_mm','z_mm']]
                if xyzm != values[5:9]: raise ValueError('Mass/vertex differs')
                if collection == 'packed':
                    pt,eta,phi,mass = [float(get(name)) for name in ['pt','eta','phi','mass']]
                    px,py,pz = pt*math.cos(phi),pt*math.sin(phi),pt*math.sinh(eta)
                    energy = math.sqrt(px*px+py*py+pz*pz+mass*mass)
                    # Polar->Cartesian conversion incurs floating-point roundoff.
                    np.testing.assert_allclose([px,py,pz,energy],values[1:5],rtol=2e-12,atol=1e-10)
                    relations = [('mother',[node['mother']]),('lastPruned',[node['last_pruned']])]
                else:
                    if [float(get(n)) for n in ['px','py','pz','energy']] != values[1:5]:
                        raise ValueError('Pruned Cartesian four-vector differs')
                    relations = [('mother',node['mothers']),('daughter',node['daughters'])]
                for name,refs in relations:
                    offsets=view[prefix+name+'Offset']
                    start,end=int(offsets[i]),int(offsets[i+1])
                    if end-start != len(refs): raise ValueError('Graph degree differs')
                    for j,ref in enumerate(refs,start):
                        state=int(view[prefix+name+'State'][j]); index=int(view[prefix+name+'Index'][j])
                        if ref is None:
                            if state!=0 or index!=-1:raise ValueError('Null reference differs')
                        elif state!=2 or index!=ref['index'] or [int(view[prefix+name+'Process'][j]),int(view[prefix+name+'Product'][j])]!=ref['product_id']:
                            raise ValueError('Graph reference differs')
                particle_count += 1
    return dict(events=10,particles=particle_count,native_audited_fields_exact=True,
                view_flags_vertices_graph_exact=True,packed_p4_roundoff_rtol=2e-12,packed_p4_roundoff_atol_GeV=1e-10)


def run_task(config, source, limit=-1, cached=None, publish=True):
    import fcntl
    lane = config.get('pilot_lane','pilot') if limit > 0 else 'sources'
    shared = Path(config['output'],lane,'%05d'%int(source['task']))
    shared.mkdir(parents=True,exist_ok=True)
    if (shared/'complete.json').exists():
        return checked_receipt(shared/'complete.json',source)
    with open(shared/'task.lock','a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            # A scheduler can reschedule a task before its previous process
            # exits. This is a lease wait, not a failed source or lost work.
            return dict(status='source-busy',task=int(source['task']))
        from generator_scratch import admit
        with admit(config) as admitted:
            if not admitted:return dict(status='scratch-busy',task=int(source['task']))
            return _run_task(config,source,limit,cached,publish)


def checked_receipt(path, source):
    item=json.loads(Path(path).read_text())
    if item['source']!=source or item['status']!='complete':
        raise ValueError('Completed source identity/status differs')
    proof=item['remote']
    if proof['status']!='two-stable-full-readbacks-passed':
        raise ValueError('Completed source lacks durable proof')
    for readback in ['first','second']:
        if proof[readback]['sha256']!=item['sha256'] or proof[readback]['bytes']!=item['bytes'] or proof[readback]['readback']!='passed':
            raise ValueError('Completed source proof differs')
    return item


def attempt_url(config, lane, tag, sha256):
    # ROOT embeds per-write UUIDs/timestamps. Regenerated physics-equivalent
    # files need not be byte-identical: never overwrite an earlier attempt.
    return config['remote'].rstrip('/')+'/'+lane+'/'+tag+'/attempts/'+sha256+'/generator.root'


def recover_source(config, source, shared, scratch, command):
    """Adopt proven output or restore a validated orphan before regenerating."""
    for candidate in sorted(shared.glob('attempt-*/validated.json')):
        old=json.loads(candidate.read_text())
        if old.get('source')!=source:raise ValueError('Recovery source identity differs')
        proof_path=candidate.parent/'remote-proof.json'
        if proof_path.exists():
            old.update(remote=json.loads(proof_path.read_text()),status='complete',finished=time.time())
            temporary=scratch/'adoption.json';atomic(temporary,old)
            checked_receipt(temporary,source)
            atomic(shared/'complete.json',old)
            return old
        # Legacy attempts used a single fixed URL. Prefer the matching ledger
        # payload; other regenerated hashes at that URL cannot be adopted.
        url=old.get('remote_url')
        if not url:
            ledger=Path(config['output'],'remote-ledger.json')
            prior=json.loads(ledger.read_text()).get('sources/'+shared.name) if ledger.exists() else None
            if not prior or prior['sha256']!=old['sha256']:continue
            url=config['remote'].rstrip('/')+'/sources/'+shared.name+'/generator.root'
        destination=scratch/('restore-'+candidate.parent.name+'.root')
        local=None
        if old.get('scratch'):
            for path in Path(old['scratch']).glob('generator*.root'):
                if path.stat().st_size==old['bytes'] and digest(path)==old['sha256']:
                    local=path;destination=path;break
        request=scratch/'restore-request.json'
        atomic(request,dict(action='restore',source=str(destination),url=url,
               bytes=old['bytes'],sha256=old['sha256'],proof=str(scratch/'restore-proof.json'),
               limits=str(Path(config['output'],'io-limits.json')),slots=str(Path(config['output'],'io-slots'))))
        try:
            if local is None:
                command(['bash',config.get('transfer_script',str(Path(config['root'],'purdue-setup/gen-backfill-transfer.sh'))),str(request)],'restore-'+candidate.parent.name,3600)
        except subprocess.CalledProcessError:
            # An absent/partial orphan is retained. A fresh attempt gets its
            # own URL; it must not replace or relabel these bytes as verified.
            atomic(scratch/('unusable-'+candidate.parent.name+'.json'),dict(candidate=str(candidate),url=url,status='not-restored'))
            if destination.exists():destination.unlink()  # failed scratch readback only
            continue
        info=old.get('input_verification')
        checks=validate(destination,expected=old['validation']['events'],source_info=info if isinstance(info,dict) else None)
        request_data=json.loads(request.read_text());request_data.update(action='publish',proof=str(proof_path),checkpoint=str(candidate.parent/'transfer-checkpoint.json'))
        atomic(request,request_data)
        command(['bash',config.get('transfer_script',str(Path(config['root'],'purdue-setup/gen-backfill-transfer.sh'))),str(request)],'recover-verify-'+candidate.parent.name,7200)
        old.update(remote=json.loads(proof_path.read_text()),validation=checks,status='complete',finished=time.time(),recovered_from=str(candidate))
        atomic(shared/'complete.json',old)
        if local is not None and local.parent.parent==Path('/tmp') and local.parent.name.startswith('gen-backfill-'+shared.name+'-'):
            shutil.rmtree(local.parent)
        return old
    return None


def _run_task(config, source, limit=-1, cached=None, publish=True):
    began = time.time()
    task = int(source['task'])
    tag = '%05d' % task
    lane = config.get('pilot_lane', 'pilot') if limit > 0 else 'sources'
    shared = Path(config['output'], lane, tag)
    shared.mkdir(parents=True, exist_ok=True)
    complete = shared / 'complete.json'
    if complete.exists(): return checked_receipt(complete,source)
    attempt = shared / ('attempt-' + uuid.uuid4().hex)
    attempt.mkdir()
    scratch = Path(tempfile.mkdtemp(prefix='gen-backfill-' + tag + '-'))
    root = Path(config['root'])
    tools = Path(config.get('tools_dir',str(root/'heptokens/tools')))
    os.environ['HEPTOKENS_BACKFILL_TOOLS']=str(tools)
    state = dict(task=task, started=began, scratch=str(scratch), source=source, limit=limit)
    atomic(attempt / 'state.json', state)
    def command(args, name, timeout=3600):
        atomic(attempt / (name + '-command.json'), args)
        with open(attempt / (name + '.log'), 'w') as log:
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    try:
        for name, expected_hash in config['software'].items():
            if digest(root / name) != expected_hash:
                raise ValueError('Campaign software changed: ' + name)
        if publish:
            recovered=recover_source(config,source,shared,scratch,command)
            if recovered is not None:
                shutil.rmtree(scratch)
                return recovered
        streaming = not cached and config.get('input_mode') == 'stream'
        source_info = None
        inp = Path(cached) if cached else scratch / 'input.root'
        if streaming:
            from replay_io_control import controlled_slot
            errors=[]
            for trial in range(3):
                try:
                    with controlled_slot(Path(config['output'],'io-slots/metadata'),str(Path(config['output'],'io-limits.json')),'max_metadata',32):
                        source_info=remote_source_info(source)
                    break
                except (subprocess.SubprocessError,OSError,TimeoutError) as error:
                    errors.append(repr(error))
                    if trial==2:raise
                    time.sleep(2*(trial+1))
            atomic(attempt/'remote-input-check.json',source_info)
            state['input_url'] = source['uri']
        elif not cached:
            # CERN Open Data is the frozen manifest authority and does not
            # require a Rucio rule. Site PFNs may be supplied after replica audit.
            urls = config.get('replicas', {}).get(str(task), []) + [source['uri']]
            errors = []
            for url in urls:
                try:
                    if inp.exists(): inp.unlink()  # only this task's incomplete scratch
                    command(['/work/pixi/global/.pixi/envs/default/bin/python', str(tools / 'download_generator_source.py'),
                             str(Path(config['output'], 'io-limits.json')), str(Path(config['output'], 'io-slots/download')),
                             url, str(inp)], 'download', 3600)
                    if inp.stat().st_size != int(source['bytes']) or digest(inp, 'adler32') != source['checksum'].split(':')[-1]:
                        raise ValueError('Source size/Adler32 differs from frozen manifest')
                    state['input_url'] = url
                    break
                except Exception as error: errors.append(repr(error))
            else: raise RuntimeError('All input routes failed: ' + repr(errors))
        elif inp.stat().st_size != int(source['bytes']) or digest(inp, 'adler32') != source['checksum'].split(':')[-1]:
            raise ValueError('Cached source differs from manifest')
        state['input_sha256'] = None if streaming else digest(inp)
        state['input_verification'] = source_info if streaming else 'fresh full local size, Adler32 and SHA256'
        state['download_finished'] = time.time()
        prefix = runtime(config, scratch)
        skim = prefix + ['cmsRun', str(tools / 'skim_generator_products.py'),
                         'inputFiles=' + (source['uri'] if streaming else 'file:' + str(inp)), 'outputFile=' + str(scratch / 'generator.root'),
                         'maxEvents=' + str(limit)]
        if streaming:
            atomic(attempt/'skim-runtime-command.json',skim)
            skim = ['/work/pixi/global/.pixi/envs/default/bin/python',str(tools/'run_generator_stream.py'),
                    str(Path(config['output'],'io-limits.json')),str(Path(config['output'],'io-slots/stream')),
                    str(attempt/'skim-runtime-command.json')]
        command(skim, 'skim')
        output = next(scratch.glob('generator*.root'))
        metadata = dict(schema='heptokens-generator-backfill-v1', source=source,
                        source_sha256=state['input_sha256'], tier='MINIAODSIM',
                        input_verification=state['input_verification'],
                        selections='none; all retained particles and native generator products',
                        units={'momentum':'GeV','position':'mm'}, particle_time='not supplied by packed/pruned accessors',
                        limits='MiniAOD pruning and quantization cannot be inverted',
                        software=config['software'])
        atomic(scratch / 'schema.json', metadata)
        command(prefix + ['python', str(tools / 'export_generator_view.py'), '--library', config['library'],
                         '--input', str(output), '--summary', str(attempt / 'native-summary.json'),
                         '--metadata', str(scratch / 'schema.json')], 'view')
        state['export_finished'] = time.time()
        checks = validate(output, limit if limit > 0 else None, None if streaming else inp, source_info)
        if limit > 0 and config.get('independent_pilot_audit') and (root/'purdue-setup/results/gen-ancestry-recovery-20261029'/('miniaod-'+tag+'.json.gz')).exists():
            audit_root = root / 'purdue-setup/results/gen-ancestry-recovery-20261029'
            audit = attempt / 'native-reopened.json.gz'
            command(prefix + ['python', str(audit_root/'audit_gen_ancestry.py'), '--input',str(output),
                              '--reference',str(root/'purdue-setup/results/fullsim-delphes-validation-100k-20261021'/tag/'input-reference.json'),
                              '--limit','10','--output',str(audit)], 'independent-audit')
            comparison = compare_independent_audit(output, audit_root/('miniaod-'+tag+'.json.gz'), audit)
            atomic(attempt/'independent-comparison.json', comparison)
            checks['independent_comparison'] = comparison
        state.update(validation=checks, bytes=output.stat().st_size, sha256=digest(output),
                     validated=time.time())
        if output.stat().st_size > config.get('max_output_bytes', 4000000000):
            raise ValueError('Source output exceeds admitted size')
        if publish:
            state['remote_url']=attempt_url(config,lane,tag,state['sha256'])
        atomic(attempt / 'validated.json', state)
        if publish:
            reserve_remote(config, lane + '/' + tag+'/attempts/'+state['sha256'], state['bytes'], state['sha256'])
            url = state['remote_url']
            request = dict(source=str(output), url=url, proof=str(attempt / 'remote-proof.json'),
                           checkpoint=str(attempt/'transfer-checkpoint.json'),
                           limits=str(Path(config['output'], 'io-limits.json')), slots=str(Path(config['output'], 'io-slots')))
            atomic(scratch / 'transfer.json', request)
            command(['bash',config.get('transfer_script',str(root/'purdue-setup/gen-backfill-transfer.sh')), str(scratch / 'transfer.json')], 'transfer', 7200)
            state['remote'] = json.loads((attempt / 'remote-proof.json').read_text())
        else:
            state['retained_output'] = str(output)
        state.update(status='complete' if publish else 'validated-local', finished=time.time(), seconds=time.time()-began)
        atomic(complete, state)
        if publish: shutil.rmtree(scratch)
        return state
    except BaseException:
        state.update(status='failed', error=traceback.format_exc(), finished=time.time())
        atomic(attempt / 'failure.json', state)
        # Keep every validated payload. Failed pre-validation intermediates
        # are reproducible from the frozen source and must not fill a node.
        if 'validation' not in state:
            atomic(attempt/'unvalidated-scratch-cleanup.json',dict(reason='source failed before full validation',
                   source=source,scratch=str(scratch),files={p.name:p.stat().st_size for p in scratch.iterdir() if p.is_file()}))
            shutil.rmtree(scratch)
        raise


async def set_slots(dask_worker):
    await dask_worker.set_resources(generator_source=1)


def slots_plugin():
    from distributed import WorkerPlugin
    class GeneratorSlots(WorkerPlugin):
        async def setup(self, worker):
            await worker.set_resources(generator_source=1)
    return GeneratorSlots()


def main():
    from dask_gateway import Gateway
    from distributed import Client, as_completed
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--tasks', nargs='*', type=int)
    p.add_argument('--limit', type=int, default=-1)
    p.add_argument('--cached', action='store_true')
    p.add_argument('--no-publish', action='store_true')
    p.add_argument('--inflight', type=int, default=4)
    p.add_argument('--defer-tasks', nargs='*', type=int, default=[])
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.limit<0:
        raise SystemExit('Full production uses watch_generator_backfill.py --config <campaign-recovery.json>; this entrypoint is reserved for bounded validation pilots.')
    manifest = json.loads(Path(config['manifest']).read_text())
    sources = [s for s in manifest['files'] if args.tasks is None or int(s['task']) in args.tasks]
    if args.limit < 0 and not Path(config['output'], 'PREFLIGHT-PASSED.json').exists():
        raise RuntimeError('Representative scientific and durable-storage preflight required')
    cluster = Gateway().connect(config['cluster'], shutdown_on_close=False)
    client = Client(cluster)
    client.register_plugin(slots_plugin(), name='generator-source-slots')
    lane = config.get('pilot_lane', 'pilot') if args.limit > 0 else 'sources'
    pending = [s for s in sources if not Path(config['output'], lane, '%05d'%int(s['task']), 'complete.json').exists()]
    completed = len(sources) - len(pending)
    queue = iter(s for s in pending if int(s['task']) not in args.defer_tasks)
    running = {}
    events = 0
    failures = []
    started = time.time()
    ac = as_completed()
    production = args.limit < 0
    ready = {}
    if production:
        for receipt in Path(config['output'],'sources').glob('*/complete.json'):
            item = json.loads(receipt.read_text())
            ready[str(item['task'])] = dict(task=item['task'],source=item['source'],events=item['validation']['events'],
                url=item['remote']['second']['url'],bytes=item['bytes'],sha256=item['sha256'],receipt=str(receipt))
    def save_ready():
        if not production: return
        atomic(Path(config['output'],'ready-sources.json'),dict(schema='heptokens-generator-backfill-index-v1',
               entire_campaign_complete=False,sources=ready,updated=time.time(),manifest=config['manifest'],
               manifest_sha256=digest(config['manifest']),events=sum(x['events'] for x in ready.values())))
    def submit():
        try: source = next(queue)
        except StopIteration: return False
        task = int(source['task'])
        cached = str(Path(config['root'], 'purdue-setup/results/fullsim-delphes-validation-20261021/input-cache', '%05d.root'%task)) if args.cached else None
        if cached and not Path(cached).exists(): cached = None
        future = client.submit(run_task, config, source, args.limit, cached, not args.no_publish,
                               pure=False, resources={'generator_source':1}, retries=0)
        running[future] = source
        ac.add(future)
        return True
    for _ in range(args.inflight):
        if not submit(): break
    while running:
        future = next(ac)
        source = running.pop(future)
        try:
            result = future.result()
            completed += 1
            events += result['validation']['events']
            if production:
                ready[str(result['task'])] = dict(task=result['task'],source=result['source'],events=result['validation']['events'],
                    url=result['remote']['second']['url'],bytes=result['bytes'],sha256=result['sha256'],
                    receipt=str(Path(config['output'],'sources','%05d'%result['task'],'complete.json')))
                save_ready()
            print(json.dumps(dict(task=source['task'],status=result['status'],events=result['validation']['events'],seconds=result['seconds'])), flush=True)
        except Exception:
            failures.append(dict(task=source['task'],error=traceback.format_exc()))
        atomic(Path(config['output'], lane + '-status.json'), dict(started=started,updated=time.time(),completed=completed,
               total=len(sources),events_this_run=events,failures=failures,running=len(running)))
        if failures and (not config.get('continue_after_failure', False) or len(failures)>=config.get('max_failures',20)):
            # Preflight stops admission on any failure. Production may continue
            # independent sources while recording failures for explicit repair.
            continue
        submit()
    client.close()
    if failures: raise RuntimeError('Backfill failures recorded; sources retained for inspection')
    if production:
        # Include complete sources owned by an earlier, non-overlapping pilot
        # controller while this controller processed the rest of the campaign.
        for receipt in Path(config['output'],'sources').glob('*/complete.json'):
            item=json.loads(receipt.read_text())
            ready[str(item['task'])]=dict(task=item['task'],source=item['source'],events=item['validation']['events'],
                url=item['remote']['second']['url'],bytes=item['bytes'],sha256=item['sha256'],receipt=str(receipt))
    save_ready()
    if production and len(ready) == len(manifest['files']):
        counts = {str(record['record']):sum(x['events'] for x in ready.values() if str(x['source']['record'])==str(record['record']))
                  for record in manifest['records']}
        if sum(x['events'] for x in ready.values()) != manifest['total_events']:
            raise ValueError('Campaign event total differs from frozen catalog')
        if any(counts[str(record['record'])] != record['events'] for record in manifest['records']):
            raise ValueError('Per-bin event total differs from frozen catalog')
        atomic(Path(config['output'],'READY.json'),dict(schema='heptokens-generator-backfill-ready-v1',
               entire_campaign_complete=True,sources=len(ready),events=manifest['total_events'],per_record=counts,
               manifest_sha256=digest(config['manifest']),index_sha256=digest(Path(config['output'],'ready-sources.json')),
               software=config['software'],finished=time.time()))


if __name__ == '__main__': main()
