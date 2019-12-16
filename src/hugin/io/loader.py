# -*- coding: utf-8 -*-
__license__ = \
    """Copyright 2019 West University of Timisoara
    
       Licensed under the Apache License, Version 2.0 (the "License");
       you may not use this file except in compliance with the License.
       You may obtain a copy of the License at
    
           http://www.apache.org/licenses/LICENSE-2.0
    
       Unless required by applicable law or agreed to in writing, software
       distributed under the License is distributed on an "AS IS" BASIS,
       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
       See the License for the specific language governing permissions and
       limitations under the License.
    """

import atexit
import logging
import threading
from collections import OrderedDict
from queue import Queue
from urllib.parse import urlparse

import backoff
import math
import numpy as np
import os
import rasterio
from hashlib import sha224
from keras.utils import to_categorical
from rasterio.io import DatasetReader
from rasterio.windows import Window
from tempfile import NamedTemporaryFile, mkdtemp

from hugin.tools.IOUtils import IOUtils

log = logging.getLogger(__name__)


class NullFormatConverter(object):
    def __init__(self):
        pass

    def __call__(self, entry):
        return entry


class CategoricalConverter(object):
    def __init__(self, num_classes=2):
        self._num_classes = num_classes

    def __call__(self, entry):
        # entry = entry.reshape(entry.shape + (1, ))
        cat = to_categorical(entry, self._num_classes)
        return cat


class BinaryCategoricalConverter(CategoricalConverter):
    """
    Converter used for representing Urband3D Ground Truth / GTI
    """

    def __init__(self, do_categorical=True):
        CategoricalConverter.__init__(self, 2)
        self.do_categorical = do_categorical

    def __call__(self, entry):
        entry = entry > 0
        if self.do_categorical:
            return CategoricalConverter.__call__(self, entry)
        return entry


class MultiClassToBinaryCategoricalConverter(BinaryCategoricalConverter):
    def __init__(self, class_label, do_categorical=True):
        BinaryCategoricalConverter.__init__(self, do_categorical)
        self.class_label = class_label

    def __call__(self, entry):
        entry = entry.copy()
        entry[entry != self.class_label] = 0
        return BinaryCategoricalConverter.__call__(self, entry)


class ColorMapperConverter(object):
    def __init__(self, color_map):
        self._color_map = color_map
        raise NotImplementedError()

    def __call__(self):
        pass


def adapt_shape_and_stride(scene, base_scene, shape, stride):
    x_geo_orig, y_geo_orig = base_scene.xy(shape[0], shape[1], offset='ul')

    computed_shape = scene.index(x_geo_orig, y_geo_orig)
    computed_stride, _ = scene.index(*base_scene.xy(stride, stride, offset='ul'))

    # We should check if scene and base_scene are the same and avoid the computation
    return computed_shape, computed_stride


class DatasetLoader(object):
    def __init__(self, datasets, loop=False, rasterio_env={}, _cache_data=False, _delete_temporary_cache=True):
        self._datasets = datasets
        self.rasterio_env = rasterio_env
        self._curent_position = 0
        self.loop = loop
        self._cache_data = _cache_data
        if self._cache_data:
            self._temp_dir = mkdtemp("cache", "hugin")

        def cleanup_dir(temp_dir):
            IOUtils.delete_recursively(temp_dir)

        if self._cache_data and _delete_temporary_cache:
            atexit.register(cleanup_dir, self._temp_dir)

    @property
    def loop(self):
        return self._loop

    @loop.setter
    def loop(self, val):
        self._loop = val

    @property
    def datasets(self):
        return self._datasets

    @datasets.setter
    def datasets(self, val):
        self._datasets = val

    def __len__(self):
        return len(self._datasets)

    def __iter__(self):
        return self

    def reset(self):
        self._curent_position = 0

    def next(self):
        return self.__next__(self)

    def __next__(self):
        length = len(self)
        if length == 0:
            raise StopIteration()
        if self._curent_position == length:
            if self._loop:
                self.reset()
            else:
                raise StopIteration()

        entry = self._datasets[self._curent_position]
        env = getattr(self, 'rasterio_env', {})
        self._curent_position += 1
        entry_name, entry_components = entry
        new_components = {}
        cache_data = self._cache_data
        use_tensorflow_io = False
        for component_name, component_path in entry_components.items():
            if isinstance(component_path, DatasetReader):
                component_path = component_path.name
            local_component_path = component_path
            url_components = urlparse(component_path)
            if not url_components.scheme:
                cache_data = False
                if url_components.path.startswith('/vsigs/'):
                    cache_data = True  # We should check if we run inside GCP ML Engine
                    use_tensorflow_io = True
                    component_path = url_components.path[6:]
                    component_path = "gs:/" + component_path
            else:
                if url_components.scheme == 'file':
                    local_component_path = url_components.path
                    use_tensorflow_io = False
                    cache_data = False

            with rasterio.Env(**env):
                if use_tensorflow_io:
                    real_path = component_path
                    data = IOUtils.open_file(real_path, "rb").read()
                    if cache_data:
                        hash = sha224(component_path.encode("utf8")).hexdigest()
                        hash_part = "/".join(list(hash)[:3])
                        dataset_path = os.path.join(self._temp_dir, hash_part)
                        if not IOUtils.file_exists(dataset_path):
                            IOUtils.recursive_create_dir(dataset_path)
                        dataset_path = os.path.join(dataset_path, os.path.basename(component_path))
                        if not IOUtils.file_exists(dataset_path):
                            f = IOUtils.open_file(dataset_path, "wb")
                            f.write(data)
                            f.close()
                        component_src = rasterio.open(dataset_path)
                    else:
                        with NamedTemporaryFile() as tmpfile:
                            tmpfile.write(data)
                            tmpfile.flush()
                            component_src = rasterio.open(tmpfile.name)
                else:
                    component_src = rasterio.open(local_component_path)
                new_components[component_name] = component_src
        return (entry_name, new_components)


class TileGenerator(object):
    def __init__(self, scene, shape=None, mapping=(), stride=None, swap_axes=False, normalize=False):
        """
        @shape: specify the shape of the window/tile. None means the window covers the whole image
        @stride: the stride used for moving the windows
        @mapping: how to map bands to the dataset data
        """
        self._scene = scene
        self._shape = shape
        self._mapping = mapping
        self._stride = stride
        self.swap_axes = swap_axes
        self._normalize = normalize
        self._count = 0
        if self._stride is None and self._shape is not None:
            self._stride = self._shape[0]

    def __iter__(self):
        return self.generate_tiles_for_dataset()

    def __len__(self):
        if not self._shape:
            return 0
        input_mapping = self._mapping
        mapped_scene = augment_mapping_with_datasets(self._scene, input_mapping)

        # pick the bigest image
        max_component_area = 0
        max_component = None
        for entry in mapped_scene:
            backing_store = entry["backing_store"]
            area = backing_store.height * backing_store.width
            if area > max_component_area:
                max_component_area = area
                max_component = backing_store

        scene_width = max_component.width
        scene_height = max_component.height

        tile_width, tile_height = self._shape
        if tile_width != self._stride:
            num_horiz = math.ceil((scene_width - tile_width) / float(self._stride) + 2)
        else:
            num_horiz = math.ceil((scene_width - tile_width) / float(self._stride) + 1)
        if tile_height != self._stride:
            num_vert = math.ceil((scene_height - tile_height) / float(self._stride) + 2)
        else:
            num_vert = math.ceil((scene_height - tile_height) / float(self._stride) + 1)

        return int(num_horiz * num_vert)

    @backoff.on_exception(backoff.expo, OSError, max_time=120)
    def read_window(self, dset, band, window):
        data = dset.read(band, window=window)

        self._count += 1
        return data

    def _generate_tiles_for_mapping(self, dataset, mapping, target_shape, target_stride):
        if not mapping: return

        window_width, window_height = target_shape

        augmented_mapping = augment_mapping_with_datasets(dataset, mapping)

        image_height, image_width = augmented_mapping[0]["backing_store"].shape

        tile_width, tile_height = target_shape
        stride = target_stride
        if tile_width != stride:
            xtiles = math.ceil((image_width - tile_width) / float(stride) + 2)
        else:
            xtiles = math.ceil((image_width - tile_width) / float(stride) + 1)

        if tile_height != stride:
            ytiles = math.ceil((image_height - tile_height) / float(stride) + 2)
        else:
            ytiles = math.ceil((image_height - tile_height) / float(stride) + 1)

        ytile = 0

        data = []
        while ytile < ytiles:
            y_start = ytile * stride
            y_end = y_start + window_height
            if y_end > image_height:
                y_start = image_height - window_height

            ytile += 1
            xtile = 0
            while xtile < xtiles:
                x_start = xtile * stride
                x_end = x_start + window_width
                if x_end > image_width:
                    x_start = image_width - window_width

                xtile += 1

                window = Window(x_start, y_start, window_width, window_height)
                for map_entry in augmented_mapping:
                    backing_store = map_entry["backing_store"]
                    channel = map_entry["channel"]
                    normalization_value = map_entry.get("normalize", None)
                    transform_expression = map_entry.get("transform", None)
                    preprocessing_callbacks = map_entry.get("preprocessing", [])

                    band = self.read_window(backing_store, channel, window)
                    if normalization_value is not None:
                        band = band / normalization_value
                    if transform_expression is not None:
                        raise NotImplementedError("Snuggs expressions are currently not implemented")
                    for callback in preprocessing_callbacks:
                        band = callback(band)
                    data.append(band)
                img_data = np.array(data)

                if self.swap_axes:
                    img_data = np.swapaxes(np.swapaxes(img_data, 0, 1), 1, 2)
                data.clear()
                yield img_data

    def generate_tiles_for_dataset(self):
        input_mapping, output_mapping = self._mapping
        if output_mapping is None:
            output_mapping = {}
        output_generators = {}
        input_generators = {}
        primary_mapping = [v for k, v in input_mapping.items() if v.get("primary", False)][0]
        primary_shape = primary_mapping['window_shape']
        primary_stride = primary_mapping['stride']
        primary_channels = primary_mapping['channels']
        primary_base_scene = self._scene[primary_channels[0][0]]

        for mapping_name, mapping in input_mapping.items():
            base_channel = mapping['channels'][0][0]
            target_shape, target_stride = adapt_shape_and_stride(self._scene[base_channel], primary_base_scene,
                                                                 primary_shape, primary_stride)
            input_generators[mapping_name] = self._generate_tiles_for_mapping(self._scene, mapping, target_shape,
                                                                              target_stride)

        for mapping_name, mapping in output_mapping.items():
            base_channel = mapping['channels'][0][0]
            target_shape, target_stride = adapt_shape_and_stride(self._scene[base_channel], primary_base_scene,
                                                                 primary_shape, primary_stride)
            output_generators[mapping_name] = self._generate_tiles_for_mapping(self._scene, mapping, target_shape,
                                                                               target_stride)

        while True:
            inputs = {}
            outputs = {}
            for mapping_name, generator in input_generators.items():
                inputs[mapping_name] = next(generator)
            for mapping_name, generator in output_generators.items():
                outputs[mapping_name] = next(generator)
            yield (inputs, outputs)


def augment_mapping_with_datasets(dataset, mapping):
    augmented_mapping = []
    if 'channels' in mapping:
        mapping = mapping['channels']
    for entry in mapping:
        if isinstance(entry, dict):
            new_entry = entry.copy()
            if "preprocessing" not in new_entry:
                new_entry["preprocessing"] = []
        elif isinstance(entry, list) or isinstance(entry, tuple):
            new_entry = OrderedDict({
                "type": entry[0],
                "channel": entry[1],
                "normalize": 1.0,
                "preprocessing": []
            })
            if len(entry) > 2:
                new_entry["normalize"] = entry[2]
        else:
            raise NotImplementedError("Unsupported format for mapping")

        new_entry['backing_store'] = dataset[new_entry["type"]]
        augmented_mapping.append(new_entry)

    return augmented_mapping


def make_categorical(y, num_classes=None):
    from keras.utils import to_categorical
    cat = to_categorical(y, num_classes)
    return cat


def make_categorical2(entry, num_classes=None):
    input_shape = entry.shape
    if input_shape and input_shape[0] == 1:
        input_shape = tuple(input_shape[1:])
    flaten = entry.ravel()
    flaten[:] = flaten[:] > 0  # make it binary
    if not num_classes:
        num_classes = np.max(flaten) + 1

    categorical = np.zeros((flaten.shape[0], num_classes))
    categorical[np.arange(flaten.shape[0]), flaten] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class DataGenerator(object):
    def __init__(self,
                 datasets,
                 batch_size,
                 input_mapping,
                 output_mapping,
                 loop=True,
                 format_converter=NullFormatConverter(),
                 swap_axes=False,
                 postprocessing_callbacks=[],
                 optimise_huge_datasets=True,
                 default_window_size=None,
                 default_stride_size=None):

        if default_window_size is not None:
            self.primary_window_shape = default_window_size

        if default_stride_size is not None:
            self.primary_stride = default_stride_size
        else:
            self.primary_stride = self.primary_window_shape[0]

        if type(input_mapping) is list or type(input_mapping) is tuple:
            input_mapping = self._convert_input_mapping(input_mapping)
        if type(output_mapping) is list or type(output_mapping) is tuple:
            output_mapping = self._convert_output_mapping(output_mapping)

        self._datasets = datasets

        primary_mapping = [input_mapping[m] for m in input_mapping if input_mapping[m].get('primary', False)]
        if len(primary_mapping) > 1:
            raise TypeError("More then one primary mappings")
        elif not primary_mapping:
            raise TypeError("No primary mapping")
        primary_mapping = primary_mapping[0]
        self._primary_mapping = primary_mapping

        self.primary_window_shape = primary_mapping['window_shape']
        self.primary_stride = primary_mapping['stride']
        datasets = self._datasets.datasets
        primary_mapping_type_id = self._primary_mapping['channels'][0][0]
        if datasets:
            self.primary_scene = datasets[0][1][primary_mapping_type_id]
        else:  # No scenes available
            self.primary_scene = None
        log.info("Primary sliding window: %s stride: %s", self.primary_window_shape, self.primary_stride)
        self._mapping = (input_mapping, output_mapping)
        self._swap_axes = swap_axes
        self._postprocessing_callbacks = postprocessing_callbacks
        self._num_tiles = None
        self._optimise_huge_datasets = optimise_huge_datasets
        self._format_converter = format_converter
        if batch_size is None:
            self._batch_size = len(self)
        else:
            self._batch_size = batch_size

        if loop:
            self.__output_generator_object = self._looping_output_generator()
        else:
            self.__output_generator_object = self._output_generator()

    def _convert_mapping(self, endpoint, mapping, primary):
        new_mapping = {}
        mapping_name = endpoint + "_1"
        new_mapping[mapping_name] = {
            'primary': primary,
            'window_shape': self.primary_window_shape,
            'stride': self.primary_stride,
            'channels': mapping
        }
        return new_mapping

    def _convert_input_mapping(self, mapping, primary=True):
        return self._convert_mapping("input", mapping, primary)

    def _convert_output_mapping(self, mapping, primary=False):
        return self._convert_mapping("output", mapping, primary)

    def next(self):
        return self.__next__(self)

    def __len__(self):
        """This is a huge resource hog! Avoid it!"""
        if self._num_tiles is not None:
            return self._num_tiles
        self._num_tiles = 0
        dataset_loader = self._datasets

        input_stride = self._primary_mapping['stride']
        input_window_shape = self._primary_mapping['window_shape']
        input_channels = self._primary_mapping['channels']
        for scene_id, scene_data in dataset_loader:
            tile_generator = TileGenerator(scene_data,
                                           input_window_shape,
                                           stride=input_stride,
                                           mapping=input_channels,
                                           swap_axes=self._swap_axes)
            self._num_tiles += len(tile_generator)
            if self._optimise_huge_datasets:
                self._num_tiles = self._num_tiles * len(dataset_loader)
                break
        dataset_loader.reset()
        return self._num_tiles

    def __next__(self):
        return next(self.__output_generator_object)

    def __iter__(self):
        return self

    def _looping_output_generator(self):
        while True:
            for data in self._output_generator():
                yield data

    def _flaten_simple_input(self, inp):
        if len(inp.keys()) != 1:
            return inp
        return list(inp.values())[0]

    def _output_generator(self):
        dataset_loader = self._datasets
        count = 0

        input_data = {}
        output_data = {}

        scene_count = 0
        callbacks = self._postprocessing_callbacks
        for scene in dataset_loader:
            scene_count += 1
            scene_id, scene_data = scene
            tile_generator = TileGenerator(scene_data,
                                           self.primary_window_shape,
                                           stride=self.primary_stride,
                                           mapping=self._mapping,
                                           swap_axes=self._swap_axes)

            for entry in tile_generator:
                count += 1
                input_patches, output_patches = entry

                for callback in callbacks:
                    input_patches, output_patches = callback(input_patches, output_patches)

                for in_patch_name, in_patch_value in input_patches.items():
                    if in_patch_name not in input_data:
                        input_data[in_patch_name] = []
                    input_data[in_patch_name].append(in_patch_value)

                for out_patch_name, out_patch_value in output_patches.items():
                    if out_patch_name not in output_data:
                        output_data[out_patch_name] = []
                    output_data[out_patch_name].append(self._format_converter(out_patch_value))

                if count == self._batch_size:
                    in_arrays = {k: np.array(v) for k, v in input_data.items()}
                    out_arrays = {k: np.array(v) for k, v in output_data.items()}
                    in_arrays = self._flaten_simple_input(in_arrays)
                    out_arrays = self._flaten_simple_input(out_arrays)

                    yield (in_arrays, out_arrays)
                    count = 0
                    for v in input_data.values():
                        v.clear()
                    for v in output_data.values():
                        v.clear()
        if count > 0:
            in_arrays = {k: np.array(v) for k, v in input_data.items()}
            out_arrays = {k: np.array(v) for k, v in output_data.items()}
            in_arrays = self._flaten_simple_input(in_arrays)
            out_arrays = self._flaten_simple_input(out_arrays)

            yield (in_arrays, out_arrays)
            for v in output_data.values():
                v.clear()
            for v in input_data.values():
                v.clear()


class ThreadedDataGenerator(threading.Thread):
    def __init__(self, data_generator, queue_size=4):
        self._queue_size = queue_size
        self._data_generator = data_generator
        self._q = Queue(maxsize=self._queue_size)
        self._len = len(self._data_generator)
        self._data_generator_flow = self._flow_data()
        threading.Thread.__init__(self)
        self.setName("ThreadedDataGenerator")
        self.setDaemon(True)
        self.start()

    def run(self):
        for d in self._data_generator:
            self._q.put(d)
        self._q.put(None)

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._data_generator_flow)

    def _flow_data(self):
        while True:
            d = self._q.get()
            if d is None:
                break
            yield d


if __name__ == "__main__":
    pass
