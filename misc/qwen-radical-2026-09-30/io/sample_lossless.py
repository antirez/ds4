import ctypes, hashlib, json, math, os, statistics, struct, time, zlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
MODEL=Path('gguf/Qwen3.8-Flash-Next-Q4.gguf')
lib=ctypes.CDLL('/usr/lib/libcompression.dylib')
for name in ('compression_encode_buffer','compression_decode_buffer'):
 f=getattr(lib,name);f.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_int];f.restype=ctypes.c_size_t
sizes={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def scalar(f,fmt): return struct.unpack(fmt,f.read(struct.calcsize(fmt)))[0]
def string(f):return f.read(scalar(f,'<Q')).decode()
def skip(f,t):
 if t in sizes:f.seek(sizes[t],1)
 elif t==8:f.seek(scalar(f,'<Q'),1)
 elif t==9:
  it,n=scalar(f,'<I'),scalar(f,'<Q')
  if it in sizes:f.seek(n*sizes[it],1)
  else:
   for _ in range(n):skip(f,it)
 else:raise ValueError(t)
def read_header(f):
 assert f.read(4)==b'GGUF';ver=scalar(f,'<I');nt=scalar(f,'<Q');nk=scalar(f,'<Q');align=32
 for _ in range(nk):
  key=string(f);t=scalar(f,'<I')
  if key=='general.alignment':assert t==4;align=scalar(f,'<I')
  else:skip(f,t)
 tensors={}
 for _ in range(nt):
  name=string(f);nd=scalar(f,'<I');dims=[scalar(f,'<Q') for _ in range(nd)];kind=scalar(f,'<I');offset=scalar(f,'<Q');tensors[name]=(dims,kind,offset)
 return (f.tell()+align-1)//align*align,tensors

def transform(data,bsize):
 # Independent 1024-block tiles preserve small bounded decompression units.
 return b''.join(b''.join(data[start+c:start+1024*bsize:bsize] for c in range(bsize)) for start in range(0,len(data),1024*bsize))
def inverse(data,bsize):
 out=bytearray(len(data))
 for start in range(0,len(data),1024*bsize):
  n=min(1024,(len(data)-start)//bsize)
  for c in range(bsize):out[start+c:start+n*bsize:bsize]=data[start+c*n:start+(c+1)*n]
 return bytes(out)
def native_encode(data,algo):
 src=ctypes.create_string_buffer(data);dst=ctypes.create_string_buffer(len(data)*2+1024)
 n=lib.compression_encode_buffer(dst,len(dst),src,len(data),None,algo)
 if not n:raise RuntimeError('encode failed')
 return dst.raw[:n]
def native_decode_time(packed,data,algo):
 src=ctypes.create_string_buffer(packed);dst=ctypes.create_string_buffer(len(data));times=[]
 for _ in range(7):
  t=time.perf_counter_ns();n=lib.compression_decode_buffer(dst,len(data),src,len(packed),None,algo);times.append(time.perf_counter_ns()-t)
  assert n==len(data)
 assert dst.raw==data
 return statistics.median(times)/1e6
rows=[];total=0
with MODEL.open('rb') as f:
 start,tensors=read_header(f)
 for layer in (0,9,23,47):
  for expert in (0,511):
   for part in ('gate','up','down'):
    name=f'blk.{layer}.ffn_{part}_exps.weight';dims,kind,off=tensors[name]
    assert dims==([2560,640,512] if part!='down' else [640,2560,512]),(name,dims)
    bsize,belems=(144,256) if kind==12 else (17,32)
    assert kind in (12,39)
    nbytes=math.prod(dims[:2])//belems*bsize;f.seek(start+off+expert*nbytes);raw=f.read(nbytes);assert len(raw)==nbytes;total+=nbytes
    variants={'raw':raw,'byte_shuffle_1024':transform(raw,bsize)}
    assert inverse(variants['byte_shuffle_1024'],bsize)==raw
    for variant,data in variants.items():
     for codec,algo in [('lz4',0x100),('lzfse',0x801),('zlib1',None)]:
      t=time.perf_counter_ns();packed=zlib.compress(data,1) if algo is None else native_encode(data,algo);enc_ms=(time.perf_counter_ns()-t)/1e6
      if algo is None:
       times=[]
       for _ in range(7):
        t=time.perf_counter_ns();unpacked=zlib.decompress(packed);times.append(time.perf_counter_ns()-t);assert unpacked==data
       dec_ms=statistics.median(times)/1e6
      else:dec_ms=native_decode_time(packed,data,algo)
      rows.append(dict(layer=layer,expert=expert,part=part,kind=kind,variant=variant,codec=codec,bytes=nbytes,compressed_bytes=len(packed),encode_ms=enc_ms,decode_ms=dec_ms,raw_sha256=hashlib.sha256(raw).hexdigest()))
summary=[]
for variant in ('raw','byte_shuffle_1024'):
 for codec in ('lz4','lzfse','zlib1'):
  selected=[r for r in rows if r['variant']==variant and r['codec']==codec];n=sum(r['bytes'] for r in selected);c=sum(r['compressed_bytes'] for r in selected);ms=sum(r['decode_ms'] for r in selected)
  summary.append(dict(variant=variant,codec=codec,ratio=c/n,saving_percent=100*(1-c/n),decode_mib_per_s=n/1048576/(ms/1000),decode_ms_per_expert=ms/8,decode_ms=ms,bytes=n,compressed_bytes=c))
result=dict(model=str(MODEL),model_bytes=MODEL.stat().st_size,payload_read_bytes=total,header_end=start,native_codec='macOS libcompression',notes=['All decompress outputs and inverse transforms checked byte-exact.','Decoder times are memory-resident median of 7 calls; raw data has no disk timing.','Byte-shuffle inverse cost excluded: this is an optimistic bound for transformed variants.','Eight experts across layers0,9,23,47 IDs0,511; not representative proof for entire model.'],summary=summary,rows=rows)
(ROOT/'lossless-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(payload_read_bytes=total,header_end=start,summary=summary),indent=2))
