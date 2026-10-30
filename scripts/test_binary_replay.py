import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib

from binary_replay import BLOCK, MAGIC, Frame, ObjectView, DeferredObservation, load_block, materialize, records
from replay_training import audit
from test_replay_training import fixture


def write_fixture(path, rows):
    """Independent test encoder; production bytes are written by the engine."""
    strings, schemas = [], []
    with Path(path).open('wb') as stream:
        stream.write(MAGIC)
        for row in rows:
            before_strings, before_schemas = len(strings), len(schemas)
            tags, values = [], []
            def intern(text):
                if text not in strings:
                    strings.append(text)
                return strings.index(text)
            def encode(value, index):
                tag, bits = 0, 0
                if value is None:
                    pass
                elif isinstance(value, bool):
                    tag = 2 if value else 1
                elif isinstance(value, int):
                    tag, bits = (3 if value < 0 else 4), value & ((1 << 64)-1)
                elif isinstance(value, float):
                    tag, bits = 5, struct.unpack('<Q', struct.pack('<d', value))[0]
                elif isinstance(value, str):
                    tag, bits = 6, intern(value)
                else:
                    children = list(value.values()) if isinstance(value, dict) else value
                    layout = [intern(key) for key in value] if isinstance(value, dict) else None
                    if layout is not None and layout not in schemas:
                        schemas.append(layout)
                    tag = 8 if layout is not None else 7
                    first = len(tags)
                    bits = first | ((schemas.index(layout) if layout is not None else len(value)) << 32)
                    tags.extend([0]*len(children)); values.extend([0]*len(children))
                    for offset, child in enumerate(children):
                        encode(child, first+offset)
                tags[index], values[index] = tag, bits
            if 'observation' in row:
                tags.append(0); values.append(0)
                encode(row['observation'], 0)
            envelope = dict(record={k:v for k,v in row.items() if k != 'observation'},
                strings=strings[before_strings:], schemas=schemas[before_schemas:])
            metadata = json.dumps(envelope, ensure_ascii=False).encode()
            raw = bytes(tags)+struct.pack('<'+'Q'*len(values), *values)
            body = zlib.compress(raw, 1) if raw else b''
            flags = int(len(body) < len(raw))
            if not flags:
                body = raw
            stream.write(BLOCK.pack(b'BLK1',len(metadata),len(tags),len(raw),len(body),
                zlib.crc32(raw,zlib.crc32(metadata)),flags))
            stream.write(metadata); stream.write(body)


class BinaryReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)/'test.training.bin'

    def test_complete_record_roundtrip_and_audit(self):
        rows = fixture()
        write_fixture(self.path, rows)
        self.assertEqual(list(records(self.path)), rows)
        reference = self.path.with_suffix('.jsonl')
        reference.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        for level in ('integrity', 'full'):
            self.assertEqual(audit(self.path, level=level), audit(reference, level=level))

    def test_lazy_ids_do_not_decode_unrequested_task_data(self):
        rows = fixture()
        entity = rows[2]['observation']['known_nonowned_entities'][0]
        entity['unrequested'] = dict(large=list(range(10000)))
        write_fixture(self.path, rows)
        sample = list(records(self.path,lazy=True))[2]
        self.assertIsInstance(sample['observation'], ObjectView)
        calls = []
        original = Frame.value
        def decode(frame,node):
            calls.append(node)
            return original(frame,node)
        with patch.object(Frame,'value',decode):
            self.assertEqual([e['id'] for e in sample['observation']['known_nonowned_entities']],[10])
        self.assertEqual(len(calls),1)
        self.assertEqual(materialize(sample['observation']),rows[2]['observation'])

    def test_shared_schemas_optional_fields_strings_and_exact_numbers(self):
        rows = fixture()
        extra = dict(none=None, yes=True, no=False, signed=-(1 << 63), unsigned=(1 << 64)-1,
                     negative_zero=-0.0, precise=1.2345678901234567, text='你好 / café',
                     empty_list=[], empty_dict={}, mixed=[1,2.5,None,dict(value='repeat')])
        rows[2]['observation']['extra'] = extra
        rows[3]['observation']['different'] = dict(text='repeat', position=[1.0,2.0,3.0])
        write_fixture(self.path,rows)
        actual=list(records(self.path))
        self.assertEqual(actual,rows)
        self.assertEqual(struct.pack('<d',actual[2]['observation']['extra']['negative_zero']),struct.pack('<d',-0.0))

    def test_vector_ids_and_types_equal_materialized_values(self):
        rows=fixture()
        entity=copy.deepcopy(rows[2]['observation']['known_nonowned_entities'][0])
        entity['type']='ChinaTank'; entity['visible']=False;entity['id']=20
        rows[2]['observation']['known_nonowned_entities'][0]['type']='ChinaDozer'
        rows[2]['observation']['known_nonowned_entities'].append(entity)
        write_fixture(self.path,rows)
        sample=list(records(self.path,lazy=True))[2]
        table=sample['observation']['known_nonowned_entities']
        with patch.object(Frame,'value',side_effect=AssertionError('ID/type lookup decoded an entity')):
            self.assertEqual(table.ids(),{10,20})
            self.assertEqual(table.ids(visible_only=True),{10})
            self.assertEqual(table.field_values('type'),['ChinaDozer','ChinaTank'])

    def test_changed_hidden_sighting_and_missing_build_result_rejected(self):
        rows=fixture()
        rows[3]['observation']['known_nonowned_entities'][0]['position'][0]+=1
        write_fixture(self.path,rows)
        with self.assertRaisesRegex(ValueError,'Hidden sighting changed'):
            audit(self.path,level='full')
        rows=fixture()
        rows[2]['actions'].append(dict(name='MSG_DOZER_CONSTRUCT',arguments=[]))
        write_fixture(self.path,rows)
        with self.assertRaisesRegex(ValueError,'Missing engine construction outcome'):
            audit(self.path)

    def test_player_filter_applies_dictionary_updates_from_other_players(self):
        rows=fixture()
        other=copy.deepcopy(rows[2]); other['player_index']=2
        other['observation']['unique']={'other_player_only':'new string'}
        rows.insert(2,other)
        rows[3]['observation']['shared']={'other_player_only':'new string'}
        write_fixture(self.path,rows)
        actual=list(records(self.path,player=3))
        self.assertEqual(actual,[row for row in rows if row.get('record')!='sample' or row['player_index']==3])

    def test_all_float_slots_checked_including_untouched_fields(self):
        rows=fixture()
        rows[2]['observation']['unused']={'value':float('inf')}
        write_fixture(self.path,rows)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            list(records(self.path,lazy=True,check_finite=True))

    def test_verified_views_defer_bodies_but_full_audit_never_does(self):
        rows=fixture();write_fixture(self.path,rows)
        with patch('binary_replay.load_block',wraps=load_block) as loading:
            result=list(records(self.path,lazy=True,validated=True))
            self.assertIsInstance(result[2]['observation'],DeferredObservation)
            self.assertFalse(any(call.args[2] for call in loading.call_args_list))
            self.assertEqual(materialize(result[2]['observation']),rows[2]['observation'])
            self.assertEqual(sum(bool(call.args[2]) for call in loading.call_args_list),1)
        rows[3]['observation']['unused']=float('inf');write_fixture(self.path,rows)
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            list(records(self.path,lazy=True,validated=True,check_finite=True))

    def test_bulk_numeric_arrays_preserve_types_and_float_bits(self):
        rows=fixture()
        rows[2]['observation']['columns']=dict(
            signed=[-(1 << 63),-1,0,1]*8,unsigned=[(1 << 64)-1,0,1]*8,
            floats=[-0.0,0.0,1.2345678901234567]*8,mixed=[1,2.5]*16)
        write_fixture(self.path,rows)
        actual=list(records(self.path))[2]['observation']['columns']
        self.assertEqual(actual,rows[2]['observation']['columns'])
        self.assertTrue(all(type(value) is float for value in actual['floats']))
        self.assertEqual(struct.pack('<d',actual['floats'][0]),struct.pack('<d',-0.0))

    def test_truncation_signature_and_checksum_are_rejected(self):
        write_fixture(self.path,fixture())
        original=self.path.read_bytes()
        for raw in (b'badmagic'+original[8:], original[:-1], original[:12], original[:30]):
            self.path.write_bytes(raw)
            with self.assertRaises(ValueError):
                list(records(self.path))
        raw=bytearray(original); raw[-1]^=1
        self.path.write_bytes(raw)
        with self.assertRaises(ValueError):
            list(records(self.path))

    def test_malformed_typed_ranges_and_unknown_tags_are_rejected(self):
        for tags, values in (([8],[1 << 32]), ([8,7],[1, (1 << 32)|1]), ([8,9],[1,0])):
            raw=bytes(tags)+struct.pack('<'+'Q'*len(values),*values)
            with self.assertRaises(ValueError):
                Frame(raw,len(tags),[],[{'field':0}])


if __name__=='__main__':
    unittest.main()
