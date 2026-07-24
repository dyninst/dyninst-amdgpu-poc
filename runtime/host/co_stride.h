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
#endif // CO_STRIDE_H
