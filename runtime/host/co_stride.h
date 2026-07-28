// co_stride.h — read an absolute symbol's value from an AMDGPU code-object ELF file.
//
// The per-wave STRIDE (bytes each wavefront occupies in the launch-time PerWaveBuf) is
// baked into the instrumented co as the absolute symbol __dyninst_pw_stride (st_value =
// STRIDE), so the artifact is self-describing: the host reads it straight from the co's
// symbol table instead of relying on an out-of-band env var. Minimal ELF64 walker (the
// co is a little-endian ET_DYN on x86 hosts, so raw reads are fine).
#ifndef CO_STRIDE_H
#define CO_STRIDE_H
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

static inline uint32_t co_read_symbol_u32(const char* path, const char* name, uint32_t dflt) {
    if (!path) return dflt;
    FILE* f = fopen(path, "rb");
    if (!f) return dflt;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 64) { fclose(f); return dflt; }
    std::vector<uint8_t> buf((size_t)sz);
    size_t rd = fread(buf.data(), 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) return dflt;
    const uint8_t* b = buf.data();
    const uint64_t fsz = (uint64_t)sz;
    if (memcmp(b, "\x7f""ELF", 4) != 0 || b[4] != 2 /*ELFCLASS64*/) return dflt;
    uint64_t shoff     = *(const uint64_t*)(b + 0x28);   // e_shoff
    uint16_t shentsize = *(const uint16_t*)(b + 0x3a);   // e_shentsize
    uint16_t shnum     = *(const uint16_t*)(b + 0x3c);   // e_shnum
    if (shentsize < 0x40 || shoff == 0 || shoff + (uint64_t)shnum * shentsize > fsz) return dflt;
    for (uint16_t i = 0; i < shnum; i++) {
        const uint8_t* sh = b + shoff + (uint64_t)i * shentsize;
        uint32_t sht = *(const uint32_t*)(sh + 0x04);                 // sh_type
        if (sht != 2 /*SHT_SYMTAB*/ && sht != 11 /*SHT_DYNSYM*/) continue;
        uint64_t symoff = *(const uint64_t*)(sh + 0x18);              // sh_offset
        uint64_t symsz  = *(const uint64_t*)(sh + 0x20);              // sh_size
        uint32_t link   = *(const uint32_t*)(sh + 0x28);              // sh_link -> strtab
        uint64_t syment = *(const uint64_t*)(sh + 0x38);              // sh_entsize
        if (syment < 24 || link >= shnum || symoff + symsz > fsz) continue;
        const uint8_t* strh = b + shoff + (uint64_t)link * shentsize;
        uint64_t stroff = *(const uint64_t*)(strh + 0x18);
        uint64_t strsz  = *(const uint64_t*)(strh + 0x20);
        if (stroff + strsz > fsz) continue;
        for (uint64_t o = 0; o + syment <= symsz; o += syment) {
            const uint8_t* sym = b + symoff + o;
            uint32_t st_name = *(const uint32_t*)(sym + 0x00);        // Elf64_Sym.st_name
            if (st_name == 0 || st_name >= strsz) continue;
            const char* nm = (const char*)(b + stroff + st_name);
            if (strcmp(nm, name) == 0)
                return (uint32_t)*(const uint64_t*)(sym + 0x08);      // Elf64_Sym.st_value
        }
    }
    return dflt;
}

// ---------------------------------------------------------------------------
// Read a kernarg .offset from the co's AMDGPU metadata (msgpack) note, so the host locates
// the COV5 hidden implicit args (hidden_block_count_x, hidden_group_size_x, ...) from the
// mutatee ITSELF instead of a hardcoded per-kernel constant. Self-describing, like
// __dyninst_pw_stride above. Targeted scan (no full msgpack parser): the compiler emits each
// arg map's keys alphabetically (.address_space, .offset, .size, .value_kind), so the
// `.offset` value is the msgpack int right after the `\xa7.offset` key, and the nearest
// `.offset` key BEFORE a given value_kind string belongs to that arg. Scans SHT_NOTE payloads
// only. Assumes the relevant kernel is the sole one in the co (true for launcher mutatees).

// Decode a msgpack unsigned int at p (positive fixint / uint8 / uint16 / uint32, big-endian);
// returns 0xffffffff on an unhandled type or overrun.
static inline uint32_t co_mp_uint(const uint8_t* p, const uint8_t* end) {
    if (p >= end) return 0xffffffffu;
    uint8_t t = *p;
    if (t < 0x80)                return t;                                            // positive fixint
    if (t == 0xcc && p + 1 < end) return p[1];                                        // uint8
    if (t == 0xcd && p + 2 < end) return ((uint32_t)p[1] << 8) | p[2];                // uint16 BE
    if (t == 0xce && p + 4 < end) return ((uint32_t)p[1] << 24) | ((uint32_t)p[2] << 16)
                                       | ((uint32_t)p[3] << 8) | p[4];                // uint32 BE
    return 0xffffffffu;
}
// Dependency-free substring search over [hay, hay+hlen); returns pointer or nullptr.
static inline const uint8_t* co_memfind(const uint8_t* hay, size_t hlen,
                                        const void* needle, size_t nlen) {
    if (nlen == 0 || hlen < nlen) return nullptr;
    for (size_t i = 0; i + nlen <= hlen; i++)
        if (memcmp(hay + i, needle, nlen) == 0) return hay + i;
    return nullptr;
}

static inline uint32_t co_read_kernarg_offset(const char* path, const char* kind, uint32_t dflt) {
    if (!path || !kind) return dflt;
    FILE* f = fopen(path, "rb");
    if (!f) return dflt;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 64) { fclose(f); return dflt; }
    std::vector<uint8_t> buf((size_t)sz);
    size_t rd = fread(buf.data(), 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) return dflt;
    const uint8_t* b = buf.data();
    const uint64_t fsz = (uint64_t)sz;
    if (memcmp(b, "\x7f""ELF", 4) != 0 || b[4] != 2) return dflt;
    uint64_t shoff     = *(const uint64_t*)(b + 0x28);
    uint16_t shentsize = *(const uint16_t*)(b + 0x3a);
    uint16_t shnum     = *(const uint16_t*)(b + 0x3c);
    if (shentsize < 0x40 || shoff == 0 || shoff + (uint64_t)shnum * shentsize > fsz) return dflt;
    const size_t klen = strlen(kind);
    static const uint8_t OFFKEY[8] = { 0xa7, '.', 'o', 'f', 'f', 's', 'e', 't' };  // msgpack str7 ".offset"
    for (uint16_t i = 0; i < shnum; i++) {
        const uint8_t* sh = b + shoff + (uint64_t)i * shentsize;
        if (*(const uint32_t*)(sh + 0x04) != 7 /*SHT_NOTE*/) continue;
        uint64_t noff = *(const uint64_t*)(sh + 0x18);
        uint64_t nsz  = *(const uint64_t*)(sh + 0x20);
        if (noff + nsz > fsz || nsz < klen) continue;
        const uint8_t* ns = b + noff;
        const uint8_t* P  = co_memfind(ns, (size_t)nsz, kind, klen);   // the value_kind string
        if (!P) continue;
        const uint8_t* Q = nullptr;                                    // nearest ".offset" key before it
        for (const uint8_t* s = ns; s + 8 <= P; s++)
            if (memcmp(s, OFFKEY, 8) == 0) Q = s;
        if (!Q) continue;
        uint32_t v = co_mp_uint(Q + 8, ns + nsz);                      // the int right after the key
        if (v != 0xffffffffu) return v;
    }
    return dflt;
}

// Count the EXPLICIT kernel arguments (i.e. the .args before the first hidden_* arg) of the
// first kernel in the co, from the AMDGPU metadata note — so the host can derive where the
// preload appends the per-wave buffer (self-describing, no PW_NARGS env). Scans SHT_NOTE for
// each `\xab.value_kind` key in order and reads the value_kind fixstr; explicit args precede
// the hidden_* block, so the count up to the first "hidden_" is the explicit-arg count.
// Assumes a single relevant kernel in the co (true for the launcher's/preload's mutatee).
static inline int co_read_explicit_arg_count(const char* path, int dflt) {
    if (!path) return dflt;
    FILE* f = fopen(path, "rb");
    if (!f) return dflt;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 64) { fclose(f); return dflt; }
    std::vector<uint8_t> buf((size_t)sz);
    size_t rd = fread(buf.data(), 1, (size_t)sz, f);
    fclose(f);
    if (rd != (size_t)sz) return dflt;
    const uint8_t* b = buf.data();
    const uint64_t fsz = (uint64_t)sz;
    if (memcmp(b, "\x7f""ELF", 4) != 0 || b[4] != 2) return dflt;
    uint64_t shoff     = *(const uint64_t*)(b + 0x28);
    uint16_t shentsize = *(const uint16_t*)(b + 0x3a);
    uint16_t shnum     = *(const uint16_t*)(b + 0x3c);
    if (shentsize < 0x40 || shoff == 0 || shoff + (uint64_t)shnum * shentsize > fsz) return dflt;
    static const uint8_t VKKEY[12] = { 0xab, '.', 'v', 'a', 'l', 'u', 'e', '_', 'k', 'i', 'n', 'd' };
    for (uint16_t i = 0; i < shnum; i++) {
        const uint8_t* sh = b + shoff + (uint64_t)i * shentsize;
        if (*(const uint32_t*)(sh + 0x04) != 7 /*SHT_NOTE*/) continue;
        uint64_t noff = *(const uint64_t*)(sh + 0x18);
        uint64_t nsz  = *(const uint64_t*)(sh + 0x20);
        if (noff + nsz > fsz || nsz < 12) continue;
        const uint8_t* ns = b + noff; const uint8_t* ne = ns + nsz;
        int count = 0; bool found = false;
        for (const uint8_t* p = ns; p + 12 <= ne; ) {
            if (memcmp(p, VKKEY, 12) != 0) { p++; continue; }
            found = true;
            const uint8_t* v = p + 12;                       // the value_kind string (fixstr / str8)
            uint32_t len; const char* str;
            if (v < ne && *v >= 0xa0 && *v <= 0xbf)      { len = *v & 0x1f; str = (const char*)(v + 1); }
            else if (v + 1 < ne && *v == 0xd9)           { len = v[1];      str = (const char*)(v + 2); }
            else { p += 12; continue; }
            if ((const uint8_t*)str + len > ne) break;
            if (len >= 7 && memcmp(str, "hidden_", 7) == 0) return count;   // first hidden -> done
            count++;
            p = (const uint8_t*)str + len;                   // past this value_kind, keep scanning
        }
        if (found) return count;                             // no hidden args (unexpected) -> explicit count
    }
    return dflt;
}
#endif // CO_STRIDE_H
