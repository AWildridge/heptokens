import argparse
import json
import numpy as np
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from heptokens.data.omnilearned import OmniLearnedModule

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
args = p.parse_args()
summary = {}
for name in ['aspen', 'jetclass', 'jetclass2']:
    with initialize_config_module(config_module='heptokens.conf', version_base=None):
        config = compose(config_name='train', overrides=[f'datamodule={name}', f'datamodule.root={args.root}', 'datamodule.num_workers=0', 'datamodule.batch_size=32'])
        module = instantiate(config.datamodule)
    module.setup()
    summary[name] = {}
    for split, ds, loader in [('train', module.train_set, module.train_dataloader()), ('val', module.valid_set, module.val_dataloader()), ('test', module.test_set, module.test_dataloader())]:
        batch = next(iter(loader))
        assert batch['mask'].dtype.is_floating_point is False
        assert batch['mask'].any()
        labels = []
        for i in range(len(ds)):
            item = ds[i]
            assert np.isfinite(item['csts'].numpy()).all()
            if 'labels' in item: labels.append(item['labels'].item())
        summary[name][split] = {'rows': len(ds), 'batch_shape': list(batch['csts'].shape), 'valid_particles': int(batch['mask'].sum()), 'labels_present': sorted(set(labels)), 'global_features': batch['jets'].shape[-1]}
        ds.close()
print(json.dumps(summary, indent=2))
