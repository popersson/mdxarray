"""
MDX -- generic Julia runtime for libraries exporting the md_abi field table.

Knows nothing about any particular struct: attribute access is built from the
table the library exports, so a field added or reordered in C++ needs no change
here. The only layouts mirrored are md_desc, md_str and md_field, which md_abi.h
defines and static_asserts.
"""
module MDX

using Libdl
export MDXLib, objtype, check, @checked

const MAX_RANK = 8

struct MDesc
    data::Ptr{Cvoid}; rank::Cint; elem::Cint; n::NTuple{8,Cint}
end
MDesc() = MDesc(C_NULL, 0, 0, ntuple(_ -> Cint(0), 8))

struct MStr; p::Ptr{UInt8}; len::Cint; end
MStr() = MStr(C_NULL, 0)

struct MField
    name::Cstring; kind::Cint; elem::Cint; rank::Cint
    flags::Cint;   pad::Cint
    count::Ptr{Cvoid}          # int32(void*) or NULL for non-indexed fields
    get::Ptr{Cvoid}; set::Ptr{Cvoid}; subtype::Ptr{Cvoid}
end

struct MType; name::Cstring; nfields::Cint; pad::Cint; fields::Ptr{MField}; end

const K_I32, K_I64, K_F64, K_F32, K_BOOL, K_STR, K_ARRAY, K_STRUCT = 0,1,2,3,4,5,6,7
const FLAG_READONLY = 1

nitems(f::MField, p::Ptr{Cvoid}) =
    f.count == C_NULL ? -1 : Int(ccall(f.count, Cint, (Ptr{Cvoid},), p))

jtype(e) = (Float64, Float32, Cint, Int64, UInt8, ComplexF64, ComplexF32)[e+1]

struct MDXLib
    path::String
end

# Checked once per library: the host's mirrored layouts must agree with the
# library's, or every accessor below would read the wrong bytes.
function verify(lib::MDXLib)
    geti(sym) = ccall(dlsym(dlopen(lib.path), sym), Cint, ())
    v = geti(:md_abi_version)
    v == 1 || error("md_abi version $v, this runtime supports 1")
    sz = geti(:md_desc_sizeof)
    sizeof(MDesc) == sz || error("MDesc is $(sizeof(MDesc)) bytes, library says $sz")
    fz = geti(:md_field_sizeof)
    sizeof(MField) == fz || error("MField is $(sizeof(MField)) bytes, library says $fz")
    mr = geti(:md_abi_max_rank)
    mr == MAX_RANK || error("MAX_RANK is $mr in the library, $MAX_RANK here")
    true
end

# layout_left is column-major like Julia, so the extents pass through unchanged.
function wrap(d::MDesc)
    d.data == C_NULL && return nothing
    dims = ntuple(i -> Int(d.n[i]), Int(d.rank))
    unsafe_wrap(Array, Ptr{jtype(d.elem)}(d.data), dims; own=false)
end

function desc(A::DenseArray{T,N}) where {T,N}
    e = T === Float64 ? 0 : T === Float32 ? 1 : T === Cint ? 2 :
        T === Int64 ? 3 : T === UInt8 ? 4 : error("unsupported element type $T")
    MDesc(pointer(A), Cint(N), Cint(e),
          ntuple(i -> i <= N ? Cint(size(A,i)) : Cint(0), 8))
end

readtable(t::Ptr{MType}) = begin
    ty = unsafe_load(t)
    tbl = Dict{Symbol,MField}(); order = Symbol[]
    for i in 1:Int(ty.nfields)
        f = unsafe_load(ty.fields, i)
        s = Symbol(unsafe_string(f.name)); tbl[s] = f; push!(order, s)
    end
    (unsafe_string(ty.name), tbl, order)
end

# A view onto a C++ object: either one the host owns (with a finalizer) or a
# sub-object borrowed from a parent.
mutable struct Obj
    ptr::Ptr{Cvoid}
    tbl::Dict{Symbol,MField}
    order::Vector{Symbol}
    tname::String
    refs::Dict{Symbol,Any}      # keeps host arrays alive
    parent::Any                 # keeps an owning parent alive
end

function _setstr(p, f, i, v)
    b = Vector{UInt8}(codeunits(String(v)))
    GC.@preserve b begin
        r = Ref(MStr(pointer(b), Cint(length(b))))
        ccall(f.set, Cvoid, (Ptr{Cvoid},Ptr{MStr},Cint), p, r, Cint(i))
    end
end

Base.propertynames(o::Obj) = Tuple(getfield(o, :order))
Base.show(io::IO, o::Obj) = print(io, getfield(o, :tname), "(",
    join(string.(getfield(o, :order)), ", "), ")")

const _INTERNAL = (:ptr, :tbl, :order, :tname, :refs, :parent)

function Base.getproperty(o::Obj, s::Symbol)
    s in _INTERNAL && return getfield(o, s)
    f = get(getfield(o, :tbl), s, nothing)
    f === nothing && throw(ArgumentError("no field $s on $(getfield(o,:tname))"))
    p = getfield(o, :ptr)
    k = f.kind
    if k == K_ARRAY
        d = Ref(MDesc()); ccall(f.get, Cvoid, (Ptr{Cvoid},Ptr{MDesc},Cint), p, d, 0)
        return wrap(d[])
    elseif k == K_STRUCT
        nm, tbl, ord = readtable(Ptr{MType}(ccall(f.subtype, Ptr{Cvoid}, ())))
        n = nitems(f, p)
        if n >= 0                                   # a list of sub-objects
            return [_substruct(o, f, nm, tbl, ord, i-1) for i in 1:n]
        end
        return _substruct(o, f, nm, tbl, ord, 0)
    elseif k == K_STR
        n = nitems(f, p)
        n >= 0 && return [_getstr(p, f, i-1) for i in 1:n]
        return _getstr(p, f, 0)
    elseif k == K_I32;  v = Ref{Cint}(0);    ccall(f.get, Cvoid, (Ptr{Cvoid},Ptr{Cint},Cint), p, v, 0); return Int(v[])
    elseif k == K_I64;  v = Ref{Int64}(0);   ccall(f.get, Cvoid, (Ptr{Cvoid},Ptr{Int64},Cint), p, v, 0); return v[]
    elseif k == K_F64;  v = Ref{Cdouble}(0); ccall(f.get, Cvoid, (Ptr{Cvoid},Ptr{Cdouble},Cint), p, v, 0); return v[]
    elseif k == K_F32;  v = Ref{Cfloat}(0);  ccall(f.get, Cvoid, (Ptr{Cvoid},Ptr{Cfloat},Cint), p, v, 0); return v[]
    elseif k == K_BOOL; v = Ref{UInt8}(0);   ccall(f.get, Cvoid, (Ptr{Cvoid},Ptr{UInt8},Cint), p, v, 0); return v[] != 0
    end
    error("unhandled kind $k")
end

# Sub-objects are borrowed: they hold a reference to the owning parent so it
# cannot be finalized while the child is alive.
function _substruct(o::Obj, f::MField, nm, tbl, ord, i)
    c = Ref(Ptr{Cvoid}(C_NULL))
    ccall(f.get, Cvoid, (Ptr{Cvoid},Ptr{Ptr{Cvoid}},Cint), getfield(o, :ptr), c, Cint(i))
    c[] == C_NULL && return nothing
    Obj(c[], tbl, ord, nm, Dict{Symbol,Any}(), o)
end

function _getstr(p, f, i)
    r = Ref(MStr()); ccall(f.get, Cvoid, (Ptr{Cvoid},Ptr{MStr},Cint), p, r, Cint(i))
    r[].p == C_NULL ? "" : unsafe_string(r[].p, Int(r[].len))
end

function Base.setproperty!(o::Obj, s::Symbol, val)
    s in _INTERNAL && return setfield!(o, s, val)
    f = getfield(o, :tbl)[s]
    f.set == C_NULL && error("field $s is read-only")
    p = getfield(o, :ptr); k = f.kind
    if k == K_ARRAY
        getfield(o, :refs)[s] = val
        d = Ref(desc(val)); ccall(f.set, Cvoid, (Ptr{Cvoid},Ptr{MDesc},Cint), p, d, 0)
    elseif k == K_STR
        n = nitems(f, p)
        if n >= 0
            length(val) == n || error("field $s expects $n strings, got $(length(val))")
            for (i, v) in enumerate(val); _setstr(p, f, i-1, v); end
        else
            _setstr(p, f, 0, val)
        end
    elseif k == K_I32;  ccall(f.set, Cvoid, (Ptr{Cvoid},Ptr{Cint},Cint),    p, Ref(Cint(val)), 0)
    elseif k == K_I64;  ccall(f.set, Cvoid, (Ptr{Cvoid},Ptr{Int64},Cint),   p, Ref(Int64(val)), 0)
    elseif k == K_F64;  ccall(f.set, Cvoid, (Ptr{Cvoid},Ptr{Cdouble},Cint), p, Ref(Cdouble(val)), 0)
    elseif k == K_BOOL; ccall(f.set, Cvoid, (Ptr{Cvoid},Ptr{UInt8},Cint),   p, Ref(UInt8(val)), 0)
    else error("cannot assign kind $k")
    end
    val
end

"""
    objtype(lib, prefix)

Build a constructor for the type exported as `<prefix>_type` / `<prefix>_new` /
`<prefix>_delete`. Nothing about the struct is written here.
"""
function objtype(lib::MDXLib, prefix::AbstractString)
    verify(lib)
    path = lib.path
    tsym, nsym, dsym = Symbol(prefix, "_type"), Symbol(prefix, "_new"), Symbol(prefix, "_delete")
    tptr = Ptr{MType}(ccall(dlsym(dlopen(path), tsym), Ptr{Cvoid}, ()))
    nm, tbl, ord = readtable(tptr)
    newf = dlsym(dlopen(path), nsym); delf = dlsym(dlopen(path), dsym)
    function ()
        p = ccall(newf, Ptr{Cvoid}, ())
        p == C_NULL && error("$prefix: allocation failed")
        o = Obj(p, tbl, ord, nm, Dict{Symbol,Any}(), nothing)
        finalizer(o) do x
            if getfield(x, :ptr) != C_NULL
                ccall(delf, Cvoid, (Ptr{Cvoid},), getfield(x, :ptr))
                setfield!(x, :ptr, Ptr{Cvoid}(C_NULL))
            end
        end
        o
    end
end

sym(lib::MDXLib, s::Symbol) = dlsym(dlopen(lib.path), s)

"""
    check(lib)

Raise if the last C entry point stored an error. A C++ exception cannot cross
an extern "C" boundary, so exporting libraries run their bodies through
md::abi::guard() and record the message instead.
"""
function check(lib::MDXLib)
    p = ccall(dlsym(dlopen(lib.path), :md_last_error), Cstring, ())
    p == C_NULL || error("C++ error: " * unsafe_string(p))
    nothing
end

"""
    @checked lib expr

Evaluate `expr`, then raise if the library recorded an error.
"""
macro checked(lib, expr)
    quote
        local v = $(esc(expr))
        check($(esc(lib)))
        v
    end
end

end # module
