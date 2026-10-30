"""The exporter benchmark must retain all gameplay and supervision differences."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from profile_training_exporter import semantic_digest


class DigestTests(unittest.TestCase):
    def test_only_identity_and_measured_placement_seconds_are_normalized(self):
        original=[dict(record='header',engine_exe_crc=1,engine_git_sha='old',engine_git_dirty=1,engine_ini_crc=123),
                  dict(record='sample',observation=dict(build_placements=[dict(seconds=1.2,mask_hex='01')],
                       known_nonowned_entities=[dict(id=10,visible=False)]),
                       actions=[dict(build_result=dict(accepted=False))])]
        with tempfile.TemporaryDirectory() as name:
            path=Path(name)/'test.jsonl'
            def hashed(rows):
                path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                return semantic_digest(path)
            baseline=hashed(original)
            diagnostic=copy.deepcopy(original)
            diagnostic[0].update(engine_exe_crc=2,engine_git_sha='new',engine_git_dirty=0)
            diagnostic[1]['observation']['build_placements'][0]['seconds']=99
            self.assertEqual(hashed(diagnostic),baseline)
            for section,field,value in [('header','engine_ini_crc',124),('placement','mask_hex','02'),
                                        ('sighting','visible',True),('outcome','accepted',True)]:
                changed=copy.deepcopy(original)
                target={'header':changed[0], 'placement':changed[1]['observation']['build_placements'][0],
                        'sighting':changed[1]['observation']['known_nonowned_entities'][0],
                        'outcome':changed[1]['actions'][0]['build_result']}[section]
                target[field]=value
                self.assertNotEqual(hashed(changed),baseline)


if __name__=='__main__':unittest.main()
