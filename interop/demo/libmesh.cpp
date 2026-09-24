// A demonstration library: a struct of arrays exported through md_abi.h.
// Exercises every field kind the ABI supports.
//
// Build: make        (produces libmesh.so)
#include "md_abi.h"

#include <new>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------- Bounds
// A nested struct, held by value in the parent.
struct Bounds {
  double lo[3] = {0, 0, 0};          // plain C array member
  double hi[3] = {0, 0, 0};
  md::sarray<double, 3> centre{};    // mdxarray static array
};

static const md_field bounds_fields_[] = {
  MD_CARRAY(Bounds, lo),
  MD_CARRAY(Bounds, hi),
  MD_ARRAY (Bounds, centre),
};
static const md_type bounds_type_ = MD_TYPE(Bounds, bounds_fields_);
extern "C" const md_type* bounds_type() { return &bounds_type_; }

// ---------------------------------------------------------------- Patch
// A sub-object held in a list whose length varies at runtime.
struct Patch {
  int32_t     id = 0;
  std::string name;
  md::array<int, 1> faces;
};

static const md_field patch_fields_[] = {
  MD_SCALAR(Patch, id, md_kind_i32),
  MD_STRING(Patch, name),
  MD_ARRAY (Patch, faces),
};
static const md_type patch_type_ = MD_TYPE(Patch, patch_fields_);
extern "C" const md_type* patch_type() { return &patch_type_; }

// ---------------------------------------------------------------- Mesh
struct Mesh {
  int32_t     dim = 0, np = 0, nt = 0;
  int64_t     nglobal = 0;               // 64-bit scalar
  double      h = 0.0;
  bool        curved = false;
  std::string name;                      // string
  int32_t     nbc = 0;                   // runtime length of bctags
  std::string bctags[8];                 // list of strings, nbc of them
  int32_t     npatch = 0;                // runtime length of patches
  Patch       patches[8];                // list of nested structs
  Bounds      bbox;                      // nested struct, by value
  md::array<double, 2> p;                // (np, dim)
  md::array<int, 2>    t;                // (nv, nt)
};

static const md_field mesh_fields_[] = {
  MD_SCALAR     (Mesh, dim,     md_kind_i32),
  MD_SCALAR     (Mesh, np,      md_kind_i32),
  MD_SCALAR     (Mesh, nt,      md_kind_i32),
  MD_SCALAR     (Mesh, nglobal, md_kind_i64),
  MD_SCALAR     (Mesh, h,       md_kind_f64),
  MD_SCALAR     (Mesh, curved,  md_kind_bool),
  MD_STRING     (Mesh, name),
  MD_STRING_LIST(Mesh, bctags,  self->nbc),        // runtime length
  MD_STRUCT_LIST(Mesh, patches, patch_type, self->npatch),
  MD_STRUCT     (Mesh, bbox,    bounds_type),
  MD_ARRAY      (Mesh, p),
  MD_ARRAY      (Mesh, t),
};
static const md_type mesh_type_ = MD_TYPE(Mesh, mesh_fields_);

MD_ABI_EXPORT()

extern "C" {

const md_type* mesh_type()  { return &mesh_type_; }
Mesh*          mesh_new()   { return new (std::nothrow) Mesh(); }
void           mesh_delete(Mesh* m) { delete m; }

void mesh_build(Mesh* m, int np, int nt, int dim) {
  md::abi::guard([&] {
    if (np <= 0 || nt <= 0 || dim <= 0)
      throw std::invalid_argument("mesh_build: np, nt and dim must be positive");

    m->dim = dim; m->np = np; m->nt = nt;
    m->nglobal = static_cast<int64_t>(np) * 1000000;
    m->h = 1.0 / (np > 1 ? np - 1 : 1);
    m->curved = true;
    m->name = "unit-square";

    const char* tags[4] = {"wall", "inlet", "outlet", "symmetry"};
    m->nbc = 4;
    for (int i = 0; i < m->nbc; ++i) m->bctags[i] = tags[i];

    m->npatch = 2;
    for (int i = 0; i < m->npatch; ++i) {
      m->patches[i].id = 100 + i;
      m->patches[i].name = std::string("patch") + std::to_string(i);
      m->patches[i].faces.resize(3);
      for (int j = 0; j < 3; ++j) m->patches[i].faces[j] = 10 * i + j;
    }

    for (int d = 0; d < 3; ++d) {
      m->bbox.lo[d] = -1.0 - d;
      m->bbox.hi[d] =  1.0 + d;
      m->bbox.centre[d] = 0.5 * d;
    }

    m->p.resize(np, dim);
    m->t.resize(3, nt);
    for (int d = 0; d < dim; ++d)
      for (int i = 0; i < np; ++i) m->p[i, d] = i * m->h + 100.0 * d;
    for (int e = 0; e < nt; ++e)
      for (int v = 0; v < 3; ++v) m->t[v, e] = (e + v) % np;
  });
}

// An ordinary kernel taking a host-owned array through a descriptor.
void mesh_scale_coords(Mesh* m, const md_desc* w) {
  md::abi::guard([&] {
    auto wv = md::abi::from_desc<const double, 1>(*w);
    if (wv.nelem() != m->np)
      throw std::invalid_argument("mesh_scale_coords: weight length must equal np");
    for (int d = 0; d < m->dim; ++d)
      for (int i = 0; i < m->np; ++i) m->p[i, d] *= wv[i];
  });
}

double mesh_total(const Mesh* m) {
  return md::abi::guard([&] { return md::sum(const_cast<Mesh*>(m)->p.cview()); });
}

}  // extern "C"
