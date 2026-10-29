"""Record final integrity aggregation after explicit human/agent visual review."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--grayscale', type=Path, required=True)
    p.add_argument('--visual-review-passed', action='store_true', required=True)
    a = p.parse_args()
    root = a.pilot.resolve()
    manifest = read(root / 'RELEASE-MANIFEST.json')['members_sha256']
    for name, expected in manifest.items():
        assert sha(root / name) == expected, name
    evidence = read(root / 'evidence-archive-verified.json')
    assert evidence['status'] == 'stable-and-read-back-twice'
    assert evidence['members_sha256'] == manifest
    assert evidence['first']['sha256'] == evidence['second']['sha256']
    plots = read(root / 'plots/summary.json')
    assert plots['status'] == 'completed' and plots['pages'] == 9
    assert all(s['marker'] is None for s in plots['styles'].values())
    proofs, hard_events, slots = [], 0, 0
    for arm in ['generic', 'cms']:
        complete = read(root / arm / 'complete.json')
        for key, value in dict(hard_events=100000, interaction_slots=2236408,
                               distinct_physical_interactions=2222684,
                               repeat_slots=13724, expected_uses_mismatches=0).items():
            assert complete[key] == value, (arm, key)
        groups = sorted((root / arm / 'groups').glob('group-*'))
        assert len(groups) == 20
        for group in groups:
            archive = read(group / 'archive-verified.json')
            sharing = read(group / 'scheduled-sharing-audit.json')
            assert archive['status'] == 'stable-and-read-back-twice'
            assert archive['first']['sha256'] == archive['second']['sha256']
            assert sharing['status'] == 'every-slot-matches-scheduled-physical-payload'
            hard_events += sharing['hard_events']
            slots += sharing['interaction_slots']
            proofs.append(str(group.relative_to(root)))
    assert hard_events == 200000 and slots == 4472816
    frozen = read(root / 'frozen-production-integrity.json')
    assert frozen['files_checked'] == frozen['matching'] == 71 and not frozen['mismatches']
    gray = read(a.grayscale / 'review.json')
    assert sha(a.grayscale / 'grayscale-comparison.png') == gray['image_sha256']
    assert sha(root / 'plots/histograms.json') == gray['source_histograms_sha256']
    review = dict(status='complete', reviewed_utc=datetime.now(timezone.utc).isoformat(),
                  integrity_method='SHA256 of every frozen release member; aggregate all archive and independent full-slot sharing proofs; exact completion totals',
                  method_sha256=sha(Path(__file__)), immutable_release_members=len(manifest),
                  release_manifest_sha256=sha(root / 'RELEASE-MANIFEST.json'),
                  archive_proofs=proofs, validated_archive_count=len(proofs),
                  independently_checked_hard_events=hard_events,
                  independently_checked_interaction_slots=slots,
                  evidence_archive_sha256=evidence['first']['sha256'],
                  evidence_archive_url=evidence['first']['url'],
                  plot_pages=9, visual_review='passed by agent inspection',
                  inspected_images=['plots/stitched-multiplicities.png',
                                    'plots/multiplicity-vs-recorded-pileup.png',
                                    'plots/earlier-versus-matched-cms.png',
                                    str(a.grayscale / 'grayscale-comparison.png')],
                  visual_findings='Distinct color/dashes, no curve markers, consistent main/ratio/legend styles. Grayscale remains distinguishable. Exact N curves intentionally overlap and are annotated. Ratios outside displayed range are clipped; flow bins are explicit.',
                  grayscale_image_sha256=gray['image_sha256'],
                  production_hashes_matching=71,
                  limits=plots['limits'])
    (root / 'final-review.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps({k:review[k] for k in ['status','immutable_release_members','validated_archive_count','independently_checked_hard_events','independently_checked_interaction_slots','plot_pages']}))


if __name__ == '__main__':
    main()
