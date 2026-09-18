import ctypes
import ctypes.util
import gzip
import math
import struct
from pathlib import Path
import numpy as np
from thrift.Thrift import TType
from thrift.protocol import TCompactProtocol
from thrift.transport import TTransport
BOOLEAN = 0
INT32 = 1
INT64 = 2
INT96 = 3
FLOAT = 4
DOUBLE = 5
BYTE_ARRAY = 6
FIXED_LEN_BYTE_ARRAY = 7
PLAIN = 0
PLAIN_DICTIONARY = 2
RLE = 3
RLE_DICTIONARY = 8
BYTE_STREAM_SPLIT = 9
DATA_PAGE = 0
INDEX_PAGE = 1
DICTIONARY_PAGE = 2
DATA_PAGE_V2 = 3
UNCOMPRESSED = 0
SNAPPY = 1
GZIP = 2
BROTLI = 4
LZ4 = 5
ZSTD = 6
LZ4_RAW = 7

def _read_thrift_value(proto, ttype):
    if ttype == TType.BOOL:
        return proto.readBool()
    if ttype == TType.BYTE:
        return proto.readByte()
    if ttype == TType.I16:
        return proto.readI16()
    if ttype == TType.I32:
        return proto.readI32()
    if ttype == TType.I64:
        return proto.readI64()
    if ttype == TType.DOUBLE:
        return proto.readDouble()
    if ttype == TType.STRING:
        return proto.readBinary()
    if ttype == TType.STRUCT:
        return _read_thrift_struct(proto)
    if ttype == TType.LIST:
        etype, n = proto.readListBegin()
        out = [_read_thrift_value(proto, etype) for _ in range(n)]
        proto.readListEnd()
        return out
    if ttype == TType.SET:
        etype, n = proto.readSetBegin()
        out = [_read_thrift_value(proto, etype) for _ in range(n)]
        proto.readSetEnd()
        return out
    if ttype == TType.MAP:
        ktype, vtype, n = proto.readMapBegin()
        out = [(_read_thrift_value(proto, ktype), _read_thrift_value(proto, vtype)) for _ in range(n)]
        proto.readMapEnd()
        return out
    raise NotImplementedError(f'Unsupported Thrift type {ttype}')

def _read_thrift_struct(proto):
    proto.readStructBegin()
    out = {}
    while True:
        _, ttype, fid = proto.readFieldBegin()
        if ttype == TType.STOP:
            break
        out[fid] = _read_thrift_value(proto, ttype)
        proto.readFieldEnd()
    proto.readStructEnd()
    return out

def _parse_thrift(data):
    trans = TTransport.TMemoryBuffer(data)
    proto = TCompactProtocol.TCompactProtocol(trans)
    value = _read_thrift_struct(proto)
    return (value, trans._buffer.tell())

class _Snappy:

    def __init__(self):
        libname = ctypes.util.find_library('snappy') or 'libsnappy.so.1'
        self.lib = ctypes.CDLL(libname)
        self.lib.snappy_uncompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.snappy_uncompress.restype = ctypes.c_int

    def decompress(self, data, expected_size):
        src = ctypes.create_string_buffer(data)
        dst = ctypes.create_string_buffer(expected_size)
        out_len = ctypes.c_size_t(expected_size)
        rc = self.lib.snappy_uncompress(src, len(data), dst, ctypes.byref(out_len))
        if rc != 0:
            raise RuntimeError(f'snappy_uncompress failed with code {rc}')
        return dst.raw[:out_len.value]
_SNAPPY = None

def _decompress(codec, data, expected_size):
    global _SNAPPY
    if codec == UNCOMPRESSED:
        return data
    if codec == SNAPPY:
        if _SNAPPY is None:
            _SNAPPY = _Snappy()
        return _SNAPPY.decompress(data, expected_size)
    if codec == GZIP:
        return gzip.decompress(data)
    if codec == ZSTD:
        libname = ctypes.util.find_library('zstd') or 'libzstd.so.1'
        lib = ctypes.CDLL(libname)
        lib.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
        lib.ZSTD_decompress.restype = ctypes.c_size_t
        src = ctypes.create_string_buffer(data)
        dst = ctypes.create_string_buffer(expected_size)
        n = int(lib.ZSTD_decompress(dst, expected_size, src, len(data)))
        lib.ZSTD_isError.argtypes = [ctypes.c_size_t]
        lib.ZSTD_isError.restype = ctypes.c_uint
        if lib.ZSTD_isError(n):
            raise RuntimeError('ZSTD decompression failed')
        return dst.raw[:n]
    if codec in (LZ4, LZ4_RAW):
        import lz4.block
        return lz4.block.decompress(data, uncompressed_size=expected_size)
    if codec == BROTLI:
        import brotli
        return brotli.decompress(data)
    raise NotImplementedError(f'Unsupported Parquet codec {codec}')

def _read_uvarint(data, pos):
    result = 0
    shift = 0
    while True:
        b = int(data[pos])
        pos += 1
        result |= (b & 127) << shift
        if b < 128:
            return (result, pos)
        shift += 7
        if shift > 63:
            raise ValueError('varint too long')

def _decode_rle_bitpacked(data, bit_width, count):
    if count == 0:
        return (np.empty(0, dtype=np.uint64), 0)
    if bit_width == 0:
        return (np.zeros(count, dtype=np.uint64), 0)
    mv = memoryview(data)
    out = np.empty(count, dtype=np.uint64)
    out_pos = 0
    pos = 0
    byte_width = (bit_width + 7) // 8
    while out_pos < count:
        header, pos = _read_uvarint(mv, pos)
        if header & 1 == 0:
            run_len = header >> 1
            if pos + byte_width > len(mv):
                raise EOFError('truncated RLE run')
            value = int.from_bytes(mv[pos:pos + byte_width], 'little')
            pos += byte_width
            take = min(run_len, count - out_pos)
            out[out_pos:out_pos + take] = value
            out_pos += take
        else:
            groups = header >> 1
            run_len = groups * 8
            nbytes = groups * bit_width
            if pos + nbytes > len(mv):
                raise EOFError('truncated bit-packed run')
            packed = mv[pos:pos + nbytes]
            pos += nbytes
            take = min(run_len, count - out_pos)
            bit_pos = 0
            mask = (1 << bit_width) - 1
            for k in range(take):
                byte_idx = bit_pos >> 3
                shift = bit_pos & 7
                need = (shift + bit_width + 7) // 8
                chunk = int.from_bytes(packed[byte_idx:byte_idx + need], 'little')
                out[out_pos + k] = chunk >> shift & mask
                bit_pos += bit_width
            out_pos += take
    return (out, pos)

def _decode_plain(data, physical_type, count, type_length=None):
    mv = memoryview(data)
    if physical_type == BOOLEAN:
        nbytes = (count + 7) // 8
        raw = np.frombuffer(mv[:nbytes], dtype=np.uint8)
        vals = np.unpackbits(raw, bitorder='little')[:count].astype(bool)
        return (vals, nbytes)
    if physical_type == INT32:
        n = count * 4
        return (np.frombuffer(mv[:n], dtype='<i4').copy(), n)
    if physical_type == INT64:
        n = count * 8
        return (np.frombuffer(mv[:n], dtype='<i8').copy(), n)
    if physical_type == FLOAT:
        n = count * 4
        return (np.frombuffer(mv[:n], dtype='<f4').copy(), n)
    if physical_type == DOUBLE:
        n = count * 8
        return (np.frombuffer(mv[:n], dtype='<f8').copy(), n)
    if physical_type == INT96:
        n = count * 12
        return ([bytes(mv[i:i + 12]) for i in range(0, n, 12)], n)
    if physical_type == BYTE_ARRAY:
        out = []
        pos = 0
        for _ in range(count):
            if pos + 4 > len(mv):
                raise EOFError('truncated byte-array length')
            n = struct.unpack_from('<I', mv, pos)[0]
            pos += 4
            out.append(bytes(mv[pos:pos + n]))
            pos += n
        return (out, pos)
    if physical_type == FIXED_LEN_BYTE_ARRAY:
        if type_length is None:
            raise ValueError('FIXED_LEN_BYTE_ARRAY requires type_length')
        n = count * type_length
        return ([bytes(mv[i:i + type_length]) for i in range(0, n, type_length)], n)
    raise NotImplementedError(f'Unsupported physical type {physical_type}')

def _decode_byte_stream_split(data, physical_type, count):
    if physical_type == FLOAT:
        width, dtype = (4, np.dtype('<f4'))
    elif physical_type == DOUBLE:
        width, dtype = (8, np.dtype('<f8'))
    else:
        raise NotImplementedError('BYTE_STREAM_SPLIT only implemented for float/double')
    raw = np.frombuffer(data, dtype=np.uint8, count=count * width).reshape(width, count).T.copy()
    return raw.reshape(-1).view(dtype)

class ColumnInfo:

    def __init__(self, name, physical_type, repetition_type, type_length, logical_type, converted_type):
        self.name = name
        self.physical_type = physical_type
        self.repetition_type = repetition_type
        self.type_length = type_length
        self.logical_type = logical_type
        self.converted_type = converted_type

class ParquetFile:

    def __init__(self, path):
        self.path = Path(path)
        self.metadata = self._read_metadata()
        self.num_rows = int(self.metadata[3])
        self.row_groups = self.metadata[4]
        self.created_by = self.metadata.get(6, b'')
        self.columns = {}
        for elem in self.metadata[2][1:]:
            name_raw = elem[4]
            name = name_raw.decode('utf-8') if isinstance(name_raw, bytes) else str(name_raw)
            self.columns[name] = ColumnInfo(name=name, physical_type=int(elem[1]), repetition_type=int(elem.get(3, 0)), type_length=int(elem[2]) if 2 in elem else None, logical_type=elem.get(10), converted_type=int(elem[6]) if 6 in elem else None)

    def _read_metadata(self):
        with self.path.open('rb') as f:
            if f.read(4) != b'PAR1':
                raise ValueError('not a Parquet file')
            f.seek(-8, 2)
            tail = f.read(8)
            if tail[4:] != b'PAR1':
                raise ValueError('invalid Parquet footer')
            n = struct.unpack('<I', tail[:4])[0]
            f.seek(-8 - n, 2)
            data = f.read(n)
        meta, consumed = _parse_thrift(data)
        if consumed != len(data):
            if len(data) - consumed > 0:
                raise ValueError(f'metadata parser left {len(data) - consumed} bytes')
        return meta

    @property
    def schema_names(self):
        return list(self.columns)

    def _read_page_header(self, f, offset):
        f.seek(offset)
        data = f.read(65536)
        header, n = _parse_thrift(data)
        return (header, n)

    def _column_chunk_meta(self, row_group, name):
        for chunk in row_group[1]:
            meta = chunk[3]
            path = meta[3]
            leaf = path[-1]
            leaf_name = leaf.decode('utf-8') if isinstance(leaf, bytes) else str(leaf)
            if leaf_name == name:
                return meta
        raise KeyError(name)

    def read_column(self, name, row_groups=None, decode_utf8=True):
        if name not in self.columns:
            raise KeyError(f'Unknown column {name!r}; available: {self.schema_names}')
        info = self.columns[name]
        groups = list(range(len(self.row_groups))) if row_groups is None else list(row_groups)
        arrays = []
        with self.path.open('rb') as f:
            for rg_idx in groups:
                arrays.append(self._read_column_row_group(f, self.row_groups[rg_idx], info, decode_utf8))
        if not arrays:
            return np.empty(0)
        return np.concatenate(arrays)

    def _read_column_row_group(self, f, row_group, info, decode_utf8):
        meta = self._column_chunk_meta(row_group, info.name)
        codec = int(meta[4])
        total_values = int(meta[5])
        data_offset = int(meta[9])
        dict_offset = int(meta[11]) if 11 in meta else None
        offset = min(data_offset, dict_offset) if dict_offset is not None else data_offset
        dictionary = None
        chunks = []
        values_seen = 0
        max_def_level = 1 if info.repetition_type == 1 else 0
        def_bit_width = 0 if max_def_level == 0 else math.ceil(math.log2(max_def_level + 1))
        while values_seen < total_values:
            header, header_len = self._read_page_header(f, offset)
            page_type = int(header[1])
            uncompressed_size = int(header[2])
            compressed_size = int(header[3])
            f.seek(offset + header_len)
            compressed = f.read(compressed_size)
            if len(compressed) != compressed_size:
                raise EOFError(f'truncated page at {offset}')
            offset += header_len + compressed_size
            if page_type == DICTIONARY_PAGE:
                page = _decompress(codec, compressed, uncompressed_size)
                dhead = header[7]
                n_dict = int(dhead[1])
                encoding = int(dhead[2])
                if encoding not in (PLAIN, PLAIN_DICTIONARY):
                    raise NotImplementedError(f'dictionary encoding {encoding}')
                dictionary, _ = _decode_plain(page, info.physical_type, n_dict, info.type_length)
                continue
            if page_type == INDEX_PAGE:
                continue
            if page_type == DATA_PAGE:
                page = _decompress(codec, compressed, uncompressed_size)
                dhead = header[5]
                n_values = int(dhead[1])
                encoding = int(dhead[2])
                pos = 0
                if max_def_level > 0:
                    level_nbytes = struct.unpack_from('<I', page, pos)[0]
                    pos += 4
                    def_levels, _ = _decode_rle_bitpacked(memoryview(page)[pos:pos + level_nbytes], def_bit_width, n_values)
                    pos += level_nbytes
                else:
                    def_levels = np.zeros(n_values, dtype=np.uint64)
                present = def_levels == max_def_level if max_def_level > 0 else np.ones(n_values, dtype=bool)
                n_present = int(present.sum())
                values_data = memoryview(page)[pos:]
            elif page_type == DATA_PAGE_V2:
                dhead = header[8]
                n_values = int(dhead[1])
                encoding = int(dhead[4])
                def_len = int(dhead[5])
                rep_len = int(dhead[6])
                is_compressed = bool(dhead.get(7, True))
                rep_bytes = compressed[:rep_len]
                def_bytes = compressed[rep_len:rep_len + def_len]
                value_bytes = compressed[rep_len + def_len:]
                expected_value_size = uncompressed_size - rep_len - def_len
                if is_compressed:
                    value_bytes = _decompress(codec, value_bytes, expected_value_size)
                if max_def_level > 0:
                    def_levels, _ = _decode_rle_bitpacked(def_bytes, def_bit_width, n_values)
                else:
                    def_levels = np.zeros(n_values, dtype=np.uint64)
                present = def_levels == max_def_level if max_def_level > 0 else np.ones(n_values, dtype=bool)
                n_present = int(present.sum())
                values_data = memoryview(value_bytes)
            else:
                raise NotImplementedError(f'page type {page_type}')
            if encoding in (RLE_DICTIONARY, PLAIN_DICTIONARY):
                if dictionary is None:
                    raise ValueError('dictionary data page without dictionary')
                bit_width = int(values_data[0]) if n_present else 0
                indices, _ = _decode_rle_bitpacked(values_data[1:], bit_width, n_present)
                if isinstance(dictionary, np.ndarray):
                    present_values = dictionary[indices.astype(np.intp)]
                else:
                    present_values = np.array([dictionary[int(i)] for i in indices], dtype=object)
            elif encoding == PLAIN:
                present_values, _ = _decode_plain(values_data, info.physical_type, n_present, info.type_length)
                if not isinstance(present_values, np.ndarray):
                    present_values = np.array(present_values, dtype=object)
            elif encoding == BYTE_STREAM_SPLIT:
                present_values = _decode_byte_stream_split(values_data, info.physical_type, n_present)
            else:
                raise NotImplementedError(f'data encoding {encoding} for {info.name}')
            if n_present == n_values:
                page_values = np.asarray(present_values)
            elif info.physical_type in (FLOAT, DOUBLE):
                page_values = np.full(n_values, np.nan, dtype=np.float64)
                page_values[present] = np.asarray(present_values, dtype=np.float64)
            else:
                page_values = np.empty(n_values, dtype=object)
                page_values[:] = None
                page_values[present] = np.asarray(present_values, dtype=object)
            chunks.append(page_values)
            values_seen += n_values
        if values_seen != total_values:
            raise ValueError(f'column {info.name}: decoded {values_seen}, expected {total_values}')
        out = np.concatenate(chunks) if chunks else np.empty(0)
        if decode_utf8 and info.physical_type in (BYTE_ARRAY, FIXED_LEN_BYTE_ARRAY):

            def dec(v):
                if v is None:
                    return None
                if isinstance(v, (bytes, bytearray, memoryview)):
                    return bytes(v).decode('utf-8', errors='replace')
                return v
            out = np.array([dec(v) for v in out], dtype=object)
        return out

    def read(self, columns, row_groups=None):
        import pandas as pd
        data = {name: self.read_column(name, row_groups=row_groups) for name in columns}
        return pd.DataFrame(data)
