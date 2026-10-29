"""Wait for validated pilot arms, produce comparisons and preserve evidence."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from replay_scratch_budget import durable_json
from commit_replay_gfal import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','pilot','baseline','prior-default','metadata','manifest','schedule']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--destination',required=True)
    p.add_argument('--deadline-epoch',type=float,required=True)
    a=p.parse_args()
    site=a.root/'purdue-setup';tools=a.root/'heptokens/tools'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(a.pilot/'scratch'))
    def state(phase,**extra):
        durable_json(a.pilot/'finish-status.json',dict(phase=phase,updated_epoch=time.time(),**extra))
    with (a.pilot/'finish.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            while True:
                status=json.loads((a.pilot/'status.json').read_text())
                if status['phase']=='simulation-complete-analysis-pending':
                    break
                if time.time()>a.deadline_epoch:
                    raise RuntimeError('Overnight deadline reached; inspect retained campaign status')
                state('waiting-for-native-pilot',campaign_phase=status['phase'])
                time.sleep(30)
            state('validating-and-plotting')
            command=['/work/pixi/global/.pixi/envs/default/bin/python',str(tools/'plot_matched_pileup_campaign.py')]
            for key in ['pilot','baseline','prior-default','metadata','manifest','schedule']:
                command+=['--'+key,str(getattr(a,key.replace('-','_')))]
            started=time.time()
            with (a.pilot/'plot-command.log').open('w') as stream:
                result=subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=3600)
            durable_json(a.pilot/'plot-command.json',dict(command=command,returncode=result.returncode,seconds=time.time()-started))
            if result.returncode:
                raise RuntimeError('Plot/independent occupancy validation failed')
            plot=json.loads((a.pilot/'plots/summary.json').read_text())
            if plot['status']!='completed' or plot['new_native_simulated_events']!=200000:
                raise ValueError('Incomplete comparison release')
            provenance=a.pilot/'provenance';provenance.mkdir(exist_ok=True)
            schedule_dir=provenance/'matched-schedule'
            if not schedule_dir.exists():
                shutil.copytree(a.schedule,schedule_dir)
            evidence=a.schedule.parent
            for name in ['additional-gen-selection.json','rucio-rules-current.csv','full-campaign-occupancy.json']:
                shutil.copy2(evidence/name,provenance/name)
            shutil.copy2(site/'results/pileup-rucio-20261021/verification/verified-input.json',provenance/'original-gen-verified-input.json')
            for name in ['cms-audit.json','library-manifest.json','source-identities.json']:
                shutil.copy2(site/'results/pileup-libraries-equal-size-20261027'/name,provenance/('original-bank-'+name))
            shutil.copy2(a.manifest,provenance/'hard-source-manifest.json')
            durable_json(provenance/'hard-reference-identities.json',[
                dict(task=task,reference_sha256=digest(a.baseline/('%05d'%task)/'input-reference.json','sha256'),
                     fullsim_sha256=digest(a.baseline/('%05d'%task)/'fullsim.json','sha256'))
                for task in [84,1373,2785,4151]])
            for subdirectory in ['native-export-validation','streaming-preflight']:
                target=provenance/subdirectory;target.mkdir(exist_ok=True)
                for path in (evidence/subdirectory).glob('*.json'):
                    shutil.copy2(path,target/path.name)
            fork=site/'software/delphes-28658365-ordered-pileup-v1'
            shutil.copy2(fork/'ORDERED-PILEUP-PROVENANCE.json',provenance/'ORDERED-PILEUP-PROVENANCE.json')
            for relative in ['modules/PileUpMerger.cc','modules/PileUpMerger.h','cards/delphes_card_CMS.tcl']:
                target=provenance/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(fork/relative,target)
            software={name:dict(path=str(fork/name),sha256=digest(fork/name,'sha256')) for name in ['DelphesHepMC2','libDelphes.so']}
            durable_json(provenance/'runtime-identities.json',software)
            durable_json(provenance/'cms-export-runtime.json',dict(
                cmssw='CMSSW_10_6_30',scram_arch='slc7_amd64_gcc700',
                image_root='/cvmfs/unpacked.cern.ch/.flat/2d/2dd3974a818c6092c053c9d7531873ebe766636cb461f3a5d90e0373cfc1211c',
                launcher_sha256=digest(site/'cmssw-exec.sh','sha256'),
                proot_sha256=digest(site/'software/proot-5.4.1','sha256'),
                note='Read existing GEN EDM; no regeneration of CMS collisions or detector response'))
            shutil.copy2(site/'cmssw-exec.sh',provenance/'cmssw-exec.sh')
            generic=site/'results/pileup-libraries-equal-size-20261027'
            shutil.copy2(generic/'generic.cmnd',provenance/'generic.cmnd')
            shutil.copy2(generic/'compile.json',provenance/'generic-generator-compile.json')
            durable_json(provenance/'generic-generator-identity.json',dict(
                pythia='8.240',binary=str(generic/'generate-generic-pileup'),
                sha256=digest(generic/'generate-generic-pileup','sha256'),
                source_sha256=digest(tools/'generate_generic_pileup.cc','sha256')))
            tool_copy=provenance/'tools';tool_copy.mkdir(exist_ok=True)
            for name in ['run_matched_pileup_campaign.py','export_cms_pileup_native.py','export_cms_pileup_native.h',
                         'archive_pileup_pilot.py','run_delphes_pileup_pilot.py','audit_delphes_pileup_library.py',
                         'generate_generic_pileup.cc','build_delphes_pileup_library.py','export_pileup_pilot_inputs.py',
                         'plot_matched_pileup_campaign.py','plot_pileup_pilot.py','archive_pileup_evidence.py','finish_matched_pileup_campaign.py']:
                shutil.copy2(tools/name,tool_copy/name)
            release=dict(status='complete-automated-validation',new_native_simulated_events=200000,
                         events_per_library=100000,unique_interactions_per_library=2222684,
                         interaction_slots_per_library=2236408,repeated_slots_per_library=13724,
                         repeat_slot_fraction=13724/2236408,expected_archives=40,
                         plots='plots/matched-pileup-comparison.pdf',metrics=plot['metrics'],
                         limitations=plot['limits'],software=software,completed_epoch=time.time())
            durable_json(a.pilot/'release-summary.json',release)
            text='Completed matched100k pileup pilot\n\n200000 new native detector events:100000 per particle library.\nEach library supplies2222684 distinct particle interactions in2236408 scheduled slots;13724 reused slots (0.6136626%), preserving the exact observed FullSim in-time sharing graph and N per hard event.\nAll40 ROOT+particle-bank archives passed native checks, member hashes and two stable full EOS readbacks.\n\nCross-section-weighted mean PF counts (statistical SEM only):\n'
            for view in ['FullSim','default-zero','matched-generic','matched-cms']:
                text+=view+': '+', '.join('%s %.3f +/- %.3f'%(key,plot['metrics'][view][key]['cross_section_weighted_mean'],plot['metrics'][view][key]['statistical_sem']) for key in ['pf','charged_pf','neutral_pf'])+'\n'
            text+='\nRead plots/matched-pileup-comparison.pdf and plots/METHODOLOGY.txt.\nThe larger bank and controlled reuse fix the identity-diversity mismatch; they do not reproduce CMS detector premixing/OOT or establish detector/ML closure. PF object definitions differ; errors exclude modelling and finite-library correlations. Earlier-versus-new panels also change N distribution and generic recipe.\n'
            (a.pilot/'SUMMARY.txt').write_text(text)
            state('archiving-evidence-and-plots')
            transfer=[str(site/'transfer-pilot/.pixi/envs/default/bin/python'),str(tools/'archive_pileup_evidence.py'),
                      '--directory',str(a.pilot),'--destination',a.destination.rstrip('/')+'/evidence-and-plots.tar']
            command=['bash','-c','source "$1" || exit; shift; exec "$@"','matched-release',str(site/'transfer-pilot/env.sh'),*transfer]
            started=time.time()
            with (a.pilot/'evidence-archive-command.log').open('w') as stream:
                result=subprocess.run(command,env=dict(env,HEPTOKENS_GFAL_BACKEND='context'),stdout=stream,stderr=subprocess.STDOUT,timeout=3600)
            durable_json(a.pilot/'evidence-archive-command.json',dict(command=command,returncode=result.returncode,seconds=time.time()-started))
            if result.returncode:
                raise RuntimeError('Final evidence staging failed; local release retained')
            state('complete-automated-validation',release='release-summary.json',plots=release['plots'],visual_review='pending agent inspection')
            durable_json(a.pilot/'status.json',dict(phase='complete-automated-validation',updated_epoch=time.time(),release='release-summary.json',new_native_simulated_events=200000,eos_evidence='evidence-archive-verified.json',visual_review='pending agent inspection'))
            print(text,flush=True)
        except BaseException as error:
            state('failed-output-preserved',error=str(error),traceback=traceback.format_exc())
            raise


if __name__=='__main__':
    main()
