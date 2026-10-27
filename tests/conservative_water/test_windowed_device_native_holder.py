"""Constructor-schema regression; Root alone runs these tests.

The genuine Native dataclass is constructed with opaque field tokens. No fake
Native module/type, bed capability, model, planner, array or GPU is constructed.
Physical validation stays with the unchanged real Native planners.
"""
import dataclasses
import unittest

from maple.core.types.voxel import WindowedVoxelColumnState
from maple_syrup.sediment_bed import BedIntegrationError
from maple_syrup import windowed_shared_bed_device as adapter


def _column():
    return WindowedVoxelColumnState(
        window_kg=object(), base=object(), deep_by_class_kg=object(),
        deep_total_kg=object(), deep_room_kg=object(), n_voxels=6,
        deep_store=object(), generation=0, manifest_digest='a'*64)


class NativeColumnHolderTests(unittest.TestCase):
    def test_native_constructor_uses_class_defaults_without_own_cache_keys(self):
        column = _column()
        before = vars(column).copy()
        self.assertEqual(len(before), 10)
        self.assertNotIn('window_cache', before)
        self.assertNotIn('height_packet', before)
        self.assertIsNone(column.window_cache)
        self.assertIsNone(column.height_packet)
        # Preserve the exact global holder rule for every other DTO.
        with self.assertRaises(BedIntegrationError):
            adapter._holder(column, WindowedVoxelColumnState)
        self.assertIs(adapter._windowed_column_holder(column), column)
        self.assertEqual(vars(column), before)

    def test_native_replace_drops_derived_keys_without_changing_inputs(self):
        column = _column()
        for name in ('window_cache', 'height_packet'):
            object.__setattr__(column, name, object())
        before = vars(column).copy()
        replaced = dataclasses.replace(column)
        self.assertNotIn('window_cache', vars(replaced))
        self.assertNotIn('height_packet', vars(replaced))
        self.assertIs(adapter._windowed_column_holder(replaced), replaced)
        self.assertEqual(vars(column), before)

    def test_declared_derived_own_keys_remain_allowed(self):
        column = _column()
        object.__setattr__(column, 'height_packet', None)
        self.assertIs(adapter._windowed_column_holder(column), column)
        object.__setattr__(column, 'window_cache', None)
        self.assertIs(adapter._windowed_column_holder(column), column)

    def test_every_init_field_is_required_as_an_own_key(self):
        for field in dataclasses.fields(WindowedVoxelColumnState):
            if field.init:
                with self.subTest(field=field.name):
                    column = _column()
                    object.__delattr__(column, field.name)
                    with self.assertRaises(BedIntegrationError):
                        adapter._windowed_column_holder(column)

    def test_extra_key_and_subclass_refuse(self):
        column = _column()
        object.__setattr__(column, 'foreign', None)
        with self.assertRaises(BedIntegrationError):
            adapter._windowed_column_holder(column)
        class ForeignColumn(WindowedVoxelColumnState):
            pass
        foreign = ForeignColumn(**vars(_column()))
        with self.assertRaises(BedIntegrationError):
            adapter._windowed_column_holder(foreign)

    def test_foreign_dictionary_keys_refuse_before_hash_or_comparison(self):
        calls = []
        class ForeignKey:
            def __hash__(self):
                calls.append('hash')
                return 1
            def __eq__(self, other):
                calls.append('equality')
                return False
        column = _column()
        vars(column)[ForeignKey()] = None
        calls.clear()
        with self.assertRaises(BedIntegrationError):
            adapter._windowed_column_holder(column)
        self.assertEqual(calls, [])

    def test_changed_field_init_default_factory_and_annotation_refuse(self):
        field = WindowedVoxelColumnState.__dataclass_fields__['window_cache']
        column = _column()
        calls = []
        def foreign_factory():
            calls.append('factory')
            raise AssertionError('a schema check must not call a factory')
        for name, invalid in (('init', True), ('default', object()),
                              ('default_factory', foreign_factory), ('type', object())):
            original = getattr(field, name)
            try:
                setattr(field, name, invalid)
                with self.assertRaises(BedIntegrationError):
                    adapter._windowed_column_holder(column)
            finally:
                setattr(field, name, original)
        self.assertEqual(calls, [])

    def test_foreign_field_definition_refuses_before_attribute_hook(self):
        column = _column()
        schema = WindowedVoxelColumnState.__dataclass_fields__
        original = schema['window_cache']
        calls = []
        class ForeignField:
            def __getattribute__(self, name):
                calls.append(name)
                raise AssertionError('foreign metadata hook')
        try:
            schema['window_cache'] = ForeignField()
            with self.assertRaises(BedIntegrationError):
                adapter._windowed_column_holder(column)
        finally:
            schema['window_cache'] = original
        self.assertEqual(calls, [])

    def test_changed_class_defaults_refuse_without_evaluating_descriptors(self):
        column = _column()
        calls = []
        def foreign_property(self):
            calls.append('property')
            raise AssertionError('must not evaluate a class default descriptor')
        for name in ('window_cache', 'height_packet', 'pending_deep', 'n_voxels'):
            namespace = vars(WindowedVoxelColumnState)
            present, original = name in namespace, namespace.get(name)
            try:
                setattr(WindowedVoxelColumnState, name, property(foreign_property))
                with self.assertRaises(BedIntegrationError):
                    adapter._windowed_column_holder(column)
            finally:
                if present:
                    setattr(WindowedVoxelColumnState, name, original)
                else:
                    delattr(WindowedVoxelColumnState, name)
        self.assertEqual(calls, [])

    def test_changed_native_attribute_hook_refuses_without_calling_it(self):
        column = _column()
        calls = []
        def foreign_getattribute(self, name):
            calls.append(name)
            raise AssertionError('must not invoke a replaced Native hook')
        try:
            WindowedVoxelColumnState.__getattribute__ = foreign_getattribute
            with self.assertRaises(BedIntegrationError):
                adapter._windowed_column_holder(column)
        finally:
            del WindowedVoxelColumnState.__getattribute__
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
