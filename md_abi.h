// md_abi.h -- a self-describing C ABI for structs built out of mdxarray arrays.
//
// The problem this solves: a C++ program using mdxarray usually wants to be
// driven from Python or Julia, and the interesting objects are structs holding
// a dozen arrays and some scalars. Mirroring such a struct by hand in the host
// language works until someone adds a field, at which point it silently reads
// the wrong memory.
//
// So nothing here exposes C++ object layout. A library exports a table of
// accessors; the host reads the table at load time and builds attribute access
// from it. Adding, removing or reordering a C++ field needs no host-side change
// and cannot corrupt anything.
//
// Two further reasons layout mirroring is a bad idea specifically for mdxarray:
// std::mdspan's memory layout is not specified by the standard, and a struct
// holding an md::array is not standard-layout, so offsetof on it is not even
// valid.
//
// The host mirrors exactly three structs -- md_desc, md_str and md_field --
// all defined and static_asserted below, and all checked against the library at
// load time via MD_ABI_EXPORT().
//
// Header-only, depends only on md.h, and adds no link dependency.
//
// SPDX-License-Identifier: MIT
#pragma once

#include "md.h"

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <exception>
#include <string>
#include <type_traits>
#include <utility>

#define MD_ABI_VERSION  1
#define MD_ABI_MAX_RANK 8

extern "C" {

// Element types for array fields.
enum {
  md_elem_f64  = 0, md_elem_f32 = 1, md_elem_i32  = 2, md_elem_i64 = 3,
  md_elem_u8   = 4, md_elem_c128 = 5, md_elem_c64 = 6
};

// Field kinds.
enum {
  md_kind_i32 = 0, md_kind_i64  = 1, md_kind_f64   = 2, md_kind_f32   = 3,
  md_kind_bool = 4, md_kind_string = 5, md_kind_array = 6, md_kind_struct = 7
};

enum { md_flag_readonly = 1 };

// A rank-erased array descriptor. Fixed size, but only at the border: the C++
// arrays themselves stay rank-templated and 16-24 bytes.
typedef struct {
  void*   data;
  int32_t rank;
  int32_t elem;
  int32_t n[MD_ABI_MAX_RANK];      // extents, first index fastest (layout_left)
} md_desc;

// A borrowed string. Valid until the owning object is modified.
typedef struct { const char* p; int32_t len; } md_str;

struct md_type;

// One row per exported member.
//
// Every field goes through a generated accessor, so there are no offsets and no
// layout assumptions. `i` selects an element of an indexed field and is ignored
// otherwise. `count` is null for plain fields and otherwise returns the current
// length, so lists may vary in size at runtime.
typedef struct {
  const char* name;
  int32_t     kind;
  int32_t     elem;                        // array fields
  int32_t     rank;                        // array fields
  int32_t     flags;
  int32_t     pad;
  int32_t   (*count)(void* obj);           // indexed fields, else null
  void      (*get)(void* obj, void* out, int32_t i);
  void      (*set)(void* obj, const void* in, int32_t i);
  const struct md_type* (*subtype)(void);  // struct fields
} md_field;

// A struct's description. Libraries export one per exported type.
typedef struct md_type {
  const char*     name;
  int32_t         nfields;
  int32_t         pad;
  const md_field* fields;
} md_type;

}  // extern "C"

static_assert(sizeof(md_desc) == 16 + 4 * MD_ABI_MAX_RANK, "md_desc layout changed");
static_assert(offsetof(md_desc, data) == 0 && offsetof(md_desc, rank) == 8 &&
              offsetof(md_desc, elem) == 12, "md_desc layout changed");
static_assert(sizeof(md_str) == 16, "md_str layout changed");

namespace md::abi {

// ---------------------------------------------------------------------------
// Element and rank traits: md::array, md::view, md::sarray and plain C arrays
// are all exportable, and expose slightly different surfaces.
// ---------------------------------------------------------------------------
template <class T> constexpr int32_t elem_code() {
  using U = std::remove_const_t<T>;
  if constexpr (std::is_same_v<U, double>)             return md_elem_f64;
  else if constexpr (std::is_same_v<U, float>)         return md_elem_f32;
  else if constexpr (std::is_same_v<U, std::int32_t>)  return md_elem_i32;
  else if constexpr (std::is_same_v<U, std::int64_t>)  return md_elem_i64;
  else if constexpr (std::is_same_v<U, long>)          return md_elem_i64;
  else if constexpr (std::is_same_v<U, bool>)          return md_elem_u8;
  else if constexpr (std::is_same_v<U, unsigned char>) return md_elem_u8;
  else static_assert(sizeof(T) == 0, "md_abi: unsupported element type");
}

template <class A> struct elem_of { using type = typename A::element_t; };
template <class T, std::size_t N> struct elem_of<T[N]> { using type = T; };
template <class A> using elem_of_t = typename elem_of<std::remove_cvref_t<A>>::type;

template <class A> constexpr int32_t rank_of() {
  if constexpr (std::is_array_v<std::remove_cvref_t<A>>) return 1;
  else return static_cast<int32_t>(std::remove_cvref_t<A>::rank());
}

template <class A> inline constexpr bool is_view_v = false;
template <class T, class E, class L>
inline constexpr bool is_view_v<md::basic_view<T, E, L>> = true;

// A view field rebinds, so the host keeps ownership and nothing is copied. An
// owning array field copies in. if constexpr only discards inside a template,
// so this cannot live directly in an accessor lambda.
template <class A, class V> void assign_field(A& tgt, V src) {
  if constexpr (is_view_v<A>) tgt = src;
  else                        md::assign(tgt, src);
}

template <class A> void to_desc(md_desc& d, A& a) {
  using E = elem_of_t<A>;
  d.data = const_cast<std::remove_const_t<E>*>(a.data());
  d.rank = rank_of<A>();
  d.elem = elem_code<E>();
  for (int32_t r = 0; r < MD_ABI_MAX_RANK; ++r)
    d.n[r] = r < d.rank ? a.n(static_cast<std::size_t>(r)) : 0;
}

// A plain C array member, e.g. double bbox[3].
template <class T, std::size_t N> void to_desc(md_desc& d, T (&a)[N]) {
  d.data = const_cast<std::remove_const_t<T>*>(&a[0]);
  d.rank = 1;
  d.elem = elem_code<T>();
  d.n[0] = static_cast<int32_t>(N);
  for (int32_t r = 1; r < MD_ABI_MAX_RANK; ++r) d.n[r] = 0;
}

template <class T, std::size_t R>
md::view<T, R> from_desc(const md_desc& d) {
  assert(d.rank == static_cast<int32_t>(R) && "md_abi: rank mismatch");
  return [&]<std::size_t... K>(std::index_sequence<K...>) {
    return md::view<T, R>(static_cast<T*>(d.data), d.n[K]...);
  }(std::make_index_sequence<R>{});
}

// ---------------------------------------------------------------------------
// Error channel.
//
// A C++ exception must not propagate across an extern "C" boundary, so every
// exported entry point runs its body through guard(), which stores the message
// for the host to collect with md_last_error().
// ---------------------------------------------------------------------------
inline std::string& error_slot() {
  static thread_local std::string slot;
  return slot;
}
inline void set_error(const char* msg) { error_slot() = msg ? msg : "unknown error"; }
inline void clear_error()              { error_slot().clear(); }
inline const char* last_error() {
  return error_slot().empty() ? nullptr : error_slot().c_str();
}

template <class F>
std::invoke_result_t<F> guard(F&& f) {
  using R = std::invoke_result_t<F>;
  try {
    clear_error();
    return f();
  } catch (const std::exception& e) {
    set_error(e.what());
  } catch (...) {
    set_error("unknown C++ exception");
  }
  if constexpr (!std::is_void_v<R>) return R{};
}

}  // namespace md::abi

// ---------------------------------------------------------------------------
// Declaring an exported struct: one line per member, written next to the struct
// so it cannot drift from the real definition.
//
// Indexed fields take a COUNT expression evaluated against `self`, a pointer to
// the object, so a list length may be a constant or a runtime value:
//     MD_STRING_LIST(Mesh, bctags, 4)
//     MD_STRING_LIST(Mesh, bctags, self->nbc)
// ---------------------------------------------------------------------------
#define MD_COUNT_FN(S, COUNT)                                                  \
  [](void* o_) -> int32_t {                                                    \
    auto* self = static_cast<S*>(o_); (void)self;                              \
    return static_cast<int32_t>(COUNT); }

#define MD_SCALAR(S, m, K)                                                     \
  { #m, K, 0, 0, 0, 0, nullptr,                                                \
    [](void* o, void* v, int32_t) {                                            \
      *static_cast<decltype(S::m)*>(v) = static_cast<S*>(o)->m; },             \
    [](void* o, const void* v, int32_t) {                                      \
      static_cast<S*>(o)->m = *static_cast<const decltype(S::m)*>(v); },       \
    nullptr }

// md::array, md::view or md::sarray.
#define MD_ARRAY(S, m)                                                         \
  { #m, md_kind_array,                                                         \
    md::abi::elem_code<md::abi::elem_of_t<decltype(S::m)>>(),                  \
    md::abi::rank_of<decltype(S::m)>(), 0, 0, nullptr,                         \
    [](void* o, void* v, int32_t) {                                            \
      md::abi::to_desc(*static_cast<md_desc*>(v), static_cast<S*>(o)->m); },   \
    [](void* o, const void* v, int32_t) {                                      \
      using A = std::remove_cvref_t<decltype(S::m)>;                           \
      md::abi::assign_field(static_cast<S*>(o)->m,                             \
        md::abi::from_desc<typename A::element_t, A::rank()>(                  \
          *static_cast<const md_desc*>(v)));                                   \
    },                                                                         \
    nullptr }

// A plain C array member. Read-only: reassigning one is meaningless, and the
// host already aliases it in place.
#define MD_CARRAY(S, m)                                                        \
  { #m, md_kind_array,                                                         \
    md::abi::elem_code<md::abi::elem_of_t<decltype(S::m)>>(),                  \
    1, md_flag_readonly, 0, nullptr,                                           \
    [](void* o, void* v, int32_t) {                                            \
      md::abi::to_desc(*static_cast<md_desc*>(v), static_cast<S*>(o)->m); },   \
    nullptr, nullptr }

#define MD_STRING(S, m)                                                        \
  { #m, md_kind_string, 0, 0, 0, 0, nullptr,                                   \
    [](void* o, void* v, int32_t) {                                            \
      const auto& s = static_cast<S*>(o)->m;                                   \
      auto* r = static_cast<md_str*>(v);                                       \
      r->p = s.c_str(); r->len = static_cast<int32_t>(s.size()); },            \
    [](void* o, const void* v, int32_t) {                                      \
      const auto* r = static_cast<const md_str*>(v);                           \
      static_cast<S*>(o)->m.assign(r->p, static_cast<std::size_t>(r->len)); }, \
    nullptr }

#define MD_STRING_LIST(S, m, COUNT)                                            \
  { #m, md_kind_string, 0, 0, 0, 0, MD_COUNT_FN(S, COUNT),                     \
    [](void* o, void* v, int32_t i) {                                          \
      const auto& s = static_cast<S*>(o)->m[i];                                \
      auto* r = static_cast<md_str*>(v);                                       \
      r->p = s.c_str(); r->len = static_cast<int32_t>(s.size()); },            \
    [](void* o, const void* v, int32_t i) {                                    \
      const auto* r = static_cast<const md_str*>(v);                           \
      static_cast<S*>(o)->m[i].assign(r->p,                                    \
                                      static_cast<std::size_t>(r->len)); },    \
    nullptr }

// A nested struct held by value.
#define MD_STRUCT(S, m, SubType)                                               \
  { #m, md_kind_struct, 0, 0, 0, 0, nullptr,                                   \
    [](void* o, void* v, int32_t) {                                            \
      *static_cast<void**>(v) = &static_cast<S*>(o)->m; },                     \
    nullptr, []() -> const md_type* { return SubType(); } }

// A nested struct held by pointer; may be null, which reads as nothing/None.
#define MD_STRUCT_PTR(S, m, SubType)                                           \
  { #m, md_kind_struct, 0, 0, 0, 0, nullptr,                                   \
    [](void* o, void* v, int32_t) {                                            \
      *static_cast<void**>(v) = static_cast<S*>(o)->m; },                      \
    nullptr, []() -> const md_type* { return SubType(); } }

// An indexed list of nested structs.
#define MD_STRUCT_LIST(S, m, SubType, COUNT)                                   \
  { #m, md_kind_struct, 0, 0, 0, 0, MD_COUNT_FN(S, COUNT),                     \
    [](void* o, void* v, int32_t i) {                                          \
      *static_cast<void**>(v) = &static_cast<S*>(o)->m[i]; },                  \
    nullptr, []() -> const md_type* { return SubType(); } }

#define MD_TYPE(S, tbl)                                                        \
  { #S, static_cast<int32_t>(sizeof(tbl) / sizeof(md_field)), 0, tbl }

// Every exporting library invokes this once, in one translation unit. It
// defines the entry points the host runtimes check at load time, and the error
// channel, which must have a single definition to be reachable through dlsym.
#define MD_ABI_EXPORT()                                                        \
  extern "C" {                                                                 \
  int32_t     md_abi_version()  { return MD_ABI_VERSION; }                     \
  int32_t     md_abi_max_rank() { return MD_ABI_MAX_RANK; }                    \
  int32_t     md_desc_sizeof()  { return static_cast<int32_t>(sizeof(md_desc)); }  \
  int32_t     md_field_sizeof() { return static_cast<int32_t>(sizeof(md_field)); } \
  const char* md_last_error()   { return md::abi::last_error(); }              \
  void        md_clear_error()  { md::abi::clear_error(); }                    \
  }
