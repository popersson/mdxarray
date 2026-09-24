# Drives libmesh.so from Julia. Self-checking: exits nonzero on failure.
include(joinpath(@__DIR__, "..", "julia", "mdx.jl"))
using .MDX

const lib  = MDX.MDXLib(joinpath(@__DIR__, "libmesh.so"))
const Mesh = MDX.objtype(lib, "mesh")      # the only line naming a C++ type

build!(m, np, nt, dim) = MDX.@checked lib ccall(
    MDX.sym(lib, :mesh_build), Cvoid, (Ptr{Cvoid},Cint,Cint,Cint),
    getfield(m, :ptr), np, nt, dim)

function scale_coords!(m, w::Vector{Float64})
    d = Ref(MDX.desc(w))
    MDX.@checked lib ccall(MDX.sym(lib, :mesh_scale_coords), Cvoid,
                           (Ptr{Cvoid},Ptr{MDX.MDesc}), getfield(m, :ptr), d)
end

nfail = 0
function check(label, cond)
    global nfail
    println(cond ? "  ok   $label" : "  FAIL $label")
    cond || (nfail += 1)
end

m = Mesh()
build!(m, 5, 4, 2)

println(m)
println("scalars")
check("dim",     m.dim == 2)
check("np",      m.np == 5)
check("nglobal (int64)", m.nglobal == 5_000_000)
check("h",       m.h ≈ 0.25)
check("curved (bool)",   m.curved === true)

println("strings")
check("name",    m.name == "unit-square")
check("bctags (runtime length)", m.bctags == ["wall","inlet","outlet","symmetry"])

println("nested structs")
check("bbox.lo (C array)",     m.bbox.lo == [-1.0,-2.0,-3.0])
check("bbox.centre (sarray)",  m.bbox.centre == [0.0,0.5,1.0])
check("patches length",        length(m.patches) == 2)
check("patches[2].name",       m.patches[2].name == "patch1")
check("patches[2].faces",      m.patches[2].faces == Int32[10,11,12])

println("arrays, zero copy")
check("p shape",  size(m.p) == (5,2))
check("t eltype", eltype(m.t) == Int32)
m.p[1,1] = -99.0
total = ccall(MDX.sym(lib, :mesh_total), Cdouble, (Ptr{Cvoid},), getfield(m, :ptr))
check("write visible in C++", total ≈ sum(m.p))

println("writes back")
m.name = "renamed"; m.curved = false; m.nglobal = 7
check("name write",    m.name == "renamed")
check("bool write",    m.curved === false)
check("int64 write",   m.nglobal == 7)
before = copy(m.p)
w = [1.0,2.0,3.0,4.0,5.0]
scale_coords!(m, w)
check("host array passed in", m.p == before .* w)

println("guards")
check("read-only C array", try (m.bbox.lo = [1.0,2,3]); false catch; true end)
check("C++ exception surfaces", try (build!(m, -1, 4, 2)); false
                                catch e; occursin("must be positive", string(e)) end)
check("shape mismatch surfaces", try (scale_coords!(m, [1.0,2.0])); false
                                 catch e; occursin("equal np", string(e)) end)

if nfail > 0
    println("\n$nfail check(s) FAILED"); exit(1)
end
println("\nall checks passed")
