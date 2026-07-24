#!/usr/bin/env python3
# expand_args.py — append extra explicit `global_buffer` kernel arguments to an AMDGPU
# code object's NT_AMDGPU_METADATA note, so the HIP runtime will COPY a matching extra
# pointer (appended to hipLaunchKernel's args[] by the preload) into the kernarg segment.
#
# Why: HIP fills the kernarg strictly from the `.args` descriptors — it copies args[i] to
# the i-th explicit arg's `.offset`. An appended args[] entry is IGNORED unless a matching
# descriptor exists. Dyninst bumps only the binary KD and does NOT touch the msgpack note
# (that would require an msgpack codec in dyninst), so this offline tool adds the
# descriptor + grows `.kernarg_segment_size`. Mirrors the KR-13 `expand_args` split:
# offline metadata growth here + runtime args[] extension in the preload.
#
# The new arg is placed at the OLD kernarg_segment_size (8-aligned) and the segment grown
# by 8 per arg — which is exactly where the dyninst emitter reads the per-wave base
# (*(kernarg + getKernargSize())) for the instrumented case.
#
# Usage: expand_args.py <co> [--kernel NAME] [--count N] [--objcopy PATH]
import sys, os, struct, subprocess, tempfile, copy
import msgpack

OBJCOPY_DEFAULT = '/opt/rocm-7.0.2/lib/llvm/bin/llvm-objcopy'

def _align(x, a=8):
    return (x + a - 1) & ~(a - 1)

def rewrite_note(note_bytes, target, count):
    """Append `count` global_buffer args to matching kernels; grow kernarg_segment_size."""
    name_sz, desc_sz, ntype = struct.unpack_from('<III', note_bytes, 0)
    name = note_bytes[12:12 + name_sz]
    off = 12 + name_sz
    while off % 4:
        off += 1
    root = msgpack.unpackb(note_bytes[off:off + desc_sz], strict_map_key=False, raw=False)

    touched = 0
    for k in root.get('amdhsa.kernels', []):
        kn = k.get('.name')
        if target and kn != target:
            continue
        args = k.get('.args', [])
        # clone an existing global_buffer descriptor so every field (address_space, etc.)
        # is exactly what this toolchain emits; only .offset (and .name) differ.
        proto = next((a for a in args if a.get('.value_kind') == 'global_buffer'), None)
        if proto is None:
            print(f"  {kn}: no global_buffer arg to clone; skipping")
            continue
        ksize = int(k.get('.kernarg_segment_size', 0))
        for i in range(count):
            newa = copy.deepcopy(proto)
            newoff = _align(ksize)
            newa['.offset'] = newoff
            newa['.size'] = 8
            newa['.name'] = f'__pw_extra{i}'
            args.append(newa)
            ksize = newoff + 8
            print(f"  {kn}: + global_buffer __pw_extra{i} @ offset {newoff}")
        k['.args'] = args
        old = k.get('.kernarg_segment_size')
        k['.kernarg_segment_size'] = ksize
        print(f"  {kn}: .kernarg_segment_size {old} -> {ksize}  (.args {len(args)-count} -> {len(args)})")
        touched += 1
    if touched == 0:
        print("  WARNING: no kernels modified")

    new_desc = msgpack.packb(root, use_bin_type=True)
    out = bytearray()
    out += struct.pack('<III', name_sz, len(new_desc), ntype)
    out += name
    while len(out) % 4:
        out.append(0)
    out += new_desc
    while len(out) % 4:
        out.append(0)
    return bytes(out)

# PT_NOTE repoint — identical to sync_note_from_kd.py (kept local to avoid a cross-import).
def repoint_pt_note(path):
    with open(path, 'r+b') as f:
        d = bytearray(f.read())
        e_shoff = struct.unpack_from('<Q', d, 0x28)[0]
        e_phoff = struct.unpack_from('<Q', d, 0x20)[0]
        e_phentsize = struct.unpack_from('<H', d, 0x36)[0]
        e_phnum = struct.unpack_from('<H', d, 0x38)[0]
        e_shentsize = struct.unpack_from('<H', d, 0x3a)[0]
        e_shnum = struct.unpack_from('<H', d, 0x3c)[0]
        e_shstrndx = struct.unpack_from('<H', d, 0x3e)[0]
        shstr_off = struct.unpack_from('<Q', d, e_shoff + e_shstrndx * e_shentsize + 24)[0]
        note_off = note_size = None
        for i in range(e_shnum):
            b = e_shoff + i * e_shentsize
            nameoff = struct.unpack_from('<I', d, b + 0)[0]
            end = d.index(b'\0', shstr_off + nameoff)
            if d[shstr_off + nameoff:end] == b'.note':
                note_off = struct.unpack_from('<Q', d, b + 24)[0]
                note_size = struct.unpack_from('<Q', d, b + 32)[0]
                break
        if note_off is None:
            print("  WARN: no .note section after add; PT_NOTE not repointed"); return
        for i in range(e_phnum):
            pb = e_phoff + i * e_phentsize
            if struct.unpack_from('<I', d, pb)[0] == 4:  # PT_NOTE
                struct.pack_into('<Q', d, pb + 8, note_off)
                struct.pack_into('<Q', d, pb + 16, note_off)
                struct.pack_into('<Q', d, pb + 24, note_off)
                struct.pack_into('<Q', d, pb + 32, note_size)
                struct.pack_into('<Q', d, pb + 40, note_size)
                print(f"  PT_NOTE -> offset={note_off:#x} size={note_size:#x}")
                break
        f.seek(0); f.write(d)

def main():
    pos = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not pos:
        print(__doc__); sys.exit(2)
    co = pos[0]
    target = None
    count = 1
    objcopy = OBJCOPY_DEFAULT
    a = sys.argv[1:]
    for i, x in enumerate(a):
        if x == '--kernel': target = a[i + 1]
        elif x == '--count': count = int(a[i + 1])
        elif x == '--objcopy': objcopy = a[i + 1]

    print(f"[expand-args] {co}: appending {count} global_buffer arg(s)"
          + (f" to {target}" if target else " to all kernels"))
    with tempfile.TemporaryDirectory() as td:
        note = os.path.join(td, 'note.bin')
        subprocess.check_call([objcopy, f'--dump-section=.note={note}', co])
        new = rewrite_note(open(note, 'rb').read(), target, count)
        newf = os.path.join(td, 'note.new')
        open(newf, 'wb').write(new)
        subprocess.check_call([objcopy, '--remove-section=.note', co])
        subprocess.check_call([objcopy, f'--add-section=.note={newf}', co])
        repoint_pt_note(co)
    print(f"[expand-args] {co}: done")

if __name__ == '__main__':
    main()
