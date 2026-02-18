from mpi4py import MPI
import numpy as np

from dolfinx import mesh as dmesh, fem, io
from dolfinx.fem.petsc import LinearProblem
import ufl
from petsc4py import PETSc

def mark_bed_surface(mesh: dmesh.Mesh, bed_id: int = 2, surface_id: int = 1, vertical_dim: int = 1):
    """
    Mark bed (downward-facing boundary) with bed_id and surface
    (upward-facing boundary) with surface_id, using the sign of the
    vertical component of the *outward* normal.

    Assumes:
      - 2D mesh (tdim = 2) embedded in 3D
      - coords[:, 0] = along-glacier
      - coords[:, 1] = vertical (thickness)
      - coords[:, 2] = 0 (unused)
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1
    assert tdim == 2, "This normal-based marker is implemented for 2D meshes."

    topo = mesh.topology
    coords = mesh.geometry.x

    # Ensure connectivity is built
    topo.create_connectivity(fdim, 0)    # facet -> vertex
    topo.create_connectivity(fdim, tdim) # facet -> cell
    topo.create_connectivity(tdim, 0)    # cell  -> vertex

    f_to_v = topo.connectivity(fdim, 0)
    f_to_c = topo.connectivity(fdim, tdim)
    c_to_v = topo.connectivity(tdim, 0)

    # All exterior boundary facets
    boundary_facets = dmesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool)
    )

    values = np.zeros(boundary_facets.shape, dtype=np.int32)
    tol = 1e-8

    for local_i, f in enumerate(boundary_facets):
        # Vertices of this facet (edge)
        facet_vertices = f_to_v.links(f)
        assert facet_vertices.size == 2, "Expecting edges as facets in 2D."

        # Use x,y only (2D plane)
        p0 = coords[facet_vertices[0], :2]
        p1 = coords[facet_vertices[1], :2]
        t = p1 - p0              # tangent vector
        n = np.array([t[1], -t[0]])  # one of the normals to edge

        # Cell that owns this boundary facet
        cells = f_to_c.links(f)
        assert cells.size == 1, "Boundary facet should belong to exactly one cell."
        cell = cells[0]

        cell_vertices = c_to_v.links(cell)
        cell_coords = coords[cell_vertices, :2]
        centroid = np.mean(cell_coords, axis=0)
        mid = 0.5 * (p0 + p1)

        # Vector from cell interior to facet midpoint
        v = mid - centroid

        # Flip normal so that it points outward
        if np.dot(n, v) < 0.0:
            n = -n

        # Vertical component (here coord 1 = vertical)
        n_vert = n[vertical_dim]

        if n_vert < -tol:
            values[local_i] = bed_id
        elif n_vert > tol:
            values[local_i] = surface_id
        # else: side/front boundary → leave as 0

    facet_tags = dmesh.meshtags(mesh, fdim, boundary_facets, values)

    if mesh.comm.rank == 0:
        n_bed = np.count_nonzero(values == bed_id)
        n_surf = np.count_nonzero(values == surface_id)
        n_zero = np.count_nonzero(values == 0)
        print(f"[mark_bed_surface] bed facets (id={bed_id}): {n_bed}")
        print(f"[mark_bed_surface] surface facets (id={surface_id}): {n_surf}")
        print(f"[mark_bed_surface] unmarked: {n_zero}")

    return facet_tags


def directional_map(fun: fem.Function, direction: int,
                    facet_tags: dmesh.MeshTags, from_id: int):
    """
    Extrude function defined on a boundary (id = from_id) into the domain
    in given coordinate direction, by solving du/dx_dir = 0 with Dirichlet data.
    """
    V = fun.function_space
    mesh = V.mesh
    dx = ufl.dx(domain=mesh)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # du/d(direction) = 0
    a = ufl.Dx(u, direction) * v * dx
    L = fem.Constant(mesh, 0.0) * v * dx

    # Dirichlet BC on facets with tag == from_id
    fdim = mesh.topology.dim - 1
    facets = facet_tags.find(from_id)
    dofs = fem.locate_dofs_topological(V, fdim, facets)
    bc = fem.dirichletbc(fun, dofs)

    problem = LinearProblem(
        a, L, bcs=[bc],
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        },
        petsc_options_prefix="dir_map_"  # any non-empty string
    )
    mapped_fun = problem.solve()
    return mapped_fun



def get_h(V: fem.FunctionSpace, facet_tags: dmesh.MeshTags,
          surface_id: int = 1, vertical_dim: int = 1):
    """
    Compute surf = h(x): vertical coordinate (coord=vertical_dim) of the surface,
    extruded downwards along that same direction.
    """
    mesh = V.mesh

    # Function with value = vertical coordinate
    z_fun = fem.Function(V)
    z_fun.interpolate(lambda x: x[vertical_dim])

    # Extrude from surface (tag=surface_id) along vertical_dim
    surf = directional_map(z_fun, direction=vertical_dim,
                           facet_tags=facet_tags, from_id=surface_id)
    return surf


def loadmesh(filename: str) -> dmesh.Mesh:
    """
    Load a mesh from XDMF (dolfinx version).

    Assumes the default grid name "mesh" in the XDMF file; adapt `name=...`
    if needed.
    """
    with io.XDMFFile(MPI.COMM_WORLD, filename, "r") as xdmf:
        msh = xdmf.read_mesh(name="Grid", xpath="/Xdmf/Domain")
    msh.name = "mesh"

    return msh
