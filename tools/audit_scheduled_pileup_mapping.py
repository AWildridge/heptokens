"""Independent full-slot check of serialized PU sharing against the SQL ledger.

Uses only Python's standard library; does not import the assembler or NumPy.
The final ledger contains immutable physical-event hashes for every ordinal.
Repeated ordinals must have byte-identical payloads in every scheduled bank.
"""
import argparse
import array
import ast
import hashlib
import json
import mmap
from pathlib import Path
import sqlite3
import struct
import sys
import zipfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def indices(path):
    with zipfile.ZipFile(path) as archive:
        payload=archive.read('indices.npy')
    if payload[:6]!=b'\x93NUMPY' or payload[6:8]!=b'\x01\x00':
        raise ValueError('Unsupported NPY format; inspect before extending parser')
    length=struct.unpack_from('<H',payload,8)[0]
    header=ast.literal_eval(payload[10:10+length].decode('latin1'))
    if header['descr']!='<i4' or header['fortran_order'] or sys.byteorder!='little':
        raise ValueError('Unexpected schedule dtype/order/platform')
    values=array.array('i')
    values.frombytes(payload[10+length:])
    if values.itemsize!=4 or header['shape']!=(len(values),):
        raise ValueError('Schedule shape differs')
    return values


def audit_group(group,schedule,ledger,output):
    index=json.loads(schedule.read_text())
    with sqlite3.connect('file:'+str(ledger.resolve())+'?mode=ro',uri=True) as db:
        count,maximum=db.execute('select count(*),max(idx) from interactions').fetchone()
        if count!=maximum+1:
            raise ValueError('Physical identity ordinals are not contiguous')
        expected=[row[0] for row in db.execute('select payload_sha256 from interactions order by idx')]
    records=[];total=0
    folders=sorted(group.glob('sequence-*'))
    if len(folders)!=5:
        raise ValueError('Require one complete five-chunk archive group')
    for folder in folders:
        sequence=int(folder.name.split('-')[1])
        chunk=index['chunks'][sequence]
        plan=Path(chunk['schedule'])
        if sha(plan)!=chunk['schedule_sha256']:
            raise ValueError('Schedule SHA changed')
        order=indices(plan)
        bank=folder/'bank.pileup'
        proof=json.loads((folder/'bank.pileup.json').read_text())
        actual_sha=sha(bank)
        if actual_sha!=proof['sha256'] or proof['schedule_sha256']!=chunk['schedule_sha256']:
            raise ValueError('Bank or schedule identity differs')
        mapping=hashlib.sha256()
        with bank.open('rb') as stream,mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as data:
            events=struct.unpack_from('>q',data,len(data)-8)[0]
            if events!=len(order) or events!=chunk['draws']:
                raise ValueError('Scheduled slot count differs')
            index_start=len(data)-8*(events+1)
            position=0
            for slot,identity in enumerate(order):
                start=struct.unpack_from('>q',data,index_start+8*slot)[0]
                particles=struct.unpack_from('>i',data,start)[0]
                stop=start+4+36*particles
                if start!=position or particles<0 or not start<stop<=index_start:
                    raise ValueError('Native event boundary/index differs')
                actual=hashlib.sha256(data[start:stop]).digest()
                if not 0<=identity<len(expected) or actual!=expected[identity]:
                    raise ValueError('Particle payload does not match scheduled physical identity')
                mapping.update(struct.pack('>q',identity)+actual)
                position=stop
            if position!=index_start or mapping.hexdigest()!=proof['mapping_sha256']:
                raise ValueError('Independent mapping digest differs')
        total+=events
        records.append(dict(sequence=sequence,slots=events,sha256=actual_sha,mapping_sha256=mapping.hexdigest(),schedule_sha256=chunk['schedule_sha256']))
    result=dict(status='every-slot-matches-scheduled-physical-payload',hard_events=5000,
                interaction_slots=total,chunks=records,ledger=str(ledger.resolve()),
                scope='Every complete native particle-event record SHA checked against the scheduled ordinal in the independent SQL ledger; full record/index boundaries checked; no assembler code imported')
    output.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['group','schedule','ledger','output']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    result=audit_group(a.group,a.schedule,a.ledger,a.output)
    print(json.dumps({k:result[k] for k in ['status','hard_events','interaction_slots']}))


if __name__=='__main__':
    main()
