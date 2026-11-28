// The C face of the OpenCASCADE calls Kreate makes: a handful of functions
// over an opaque shape, every one catching what OCCT throws and answering
// NULL (or 0) instead, so nothing C++ ever unwinds into Rust.
//
// A face is handed over as numbers (see `loops_face`); everything else is
// shapes in and shapes out. Lengths are metres, as Kreate keeps them.

#include <BRepAdaptor_Curve.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_GTransform.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepFilletAPI_MakeChamfer.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <BRepGProp.hxx>
#include <Bnd_Box.hxx>
#include <BRepLib.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepOffsetAPI_DraftAngle.hxx>
#include <BRepOffsetAPI_MakePipe.hxx>
#include <BRepOffsetAPI_MakePipeShell.hxx>
#include <BRepOffsetAPI_MakeThickSolid.hxx>
#include <BRepOffsetAPI_ThruSections.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <BRepTools.hxx>
#include <BRep_Tool.hxx>
#include <GCPnts_QuasiUniformDeflection.hxx>
#include <GProp_GProps.hxx>
#include <GeomAPI_PointsToBSpline.hxx>
#include <GeomAbs_CurveType.hxx>
#include <GeomAbs_SurfaceType.hxx>
#include <Geom_BSplineCurve.hxx>
#include <HLRAlgo_Projector.hxx>
#include <HLRBRep_Algo.hxx>
#include <HLRBRep_HLRToShape.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <Interface_Static.hxx>
#include <STEPControl_Controller.hxx>
#include <STEPControl_Reader.hxx>
#include <STEPControl_Writer.hxx>
#include <ShapeFix_Face.hxx>
#include <ShapeFix_Shape.hxx>
#include <Standard_Failure.hxx>
#include <TColStd_Array1OfInteger.hxx>
#include <TColStd_Array1OfReal.hxx>
#include <TColgp_Array1OfPnt.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Wire.hxx>
#include <gp_Ax2.hxx>
#include <gp_Circ.hxx>
#include <gp_Elips.hxx>
#include <gp_GTrsf.hxx>
#include <gp_Pln.hxx>
#include <gp_Trsf.hxx>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

struct KShape {
    TopoDS_Shape shape;
};

struct KMesh {
    size_t verts;
    double* positions;  // xyz each
    size_t tris;
    uint32_t* indices;  // three each
    uint32_t* faces;    // the B-rep face each triangle is on, from 0
};

struct KHlrBuffer {
    size_t len;
    double* data;
};

struct KTopoSig {
    size_t count;
    double* data;  // 8 doubles per face (cx,cy,cz,nx,ny,nz,area,kind) or 11 per edge
};

namespace {

KShape* wrap(const TopoDS_Shape& s) {
    if (s.IsNull()) return nullptr;
    return new KShape{s};
}

// One piece of a loop, read from `p`, as an edge in the plane z = 0.
// Returns the edge, or a null one; advances `p` past the piece.
TopoDS_Edge piece(const double*& p, const double* end) {
    auto take = [&](size_t n) -> const double* {
        if (p + n > end) throw Standard_Failure("short");
        const double* at = p;
        p += n;
        return at;
    };
    int kind = (int)*take(1);
    gp_Ax2 flat(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1));
    switch (kind) {
        case 0: {  // line: a, b
            const double* v = take(4);
            return BRepBuilderAPI_MakeEdge(gp_Pnt(v[0], v[1], 0), gp_Pnt(v[2], v[3], 0));
        }
        case 1: {  // arc: centre, radius, start, sweep (signed)
            const double* v = take(5);
            gp_Circ c(gp_Ax2(gp_Pnt(v[0], v[1], 0), gp_Dir(0, 0, 1)), v[2]);
            double a = v[3], s = v[4];
            if (s >= 0) return BRepBuilderAPI_MakeEdge(c, a, a + s);
            TopoDS_Edge e = BRepBuilderAPI_MakeEdge(c, a + s, a);
            e.Reverse();
            return e;
        }
        case 2: {  // circle: centre, radius
            const double* v = take(3);
            return BRepBuilderAPI_MakeEdge(gp_Circ(gp_Ax2(gp_Pnt(v[0], v[1], 0), gp_Dir(0, 0, 1)), v[2]));
        }
        case 3: {  // B-spline: degree, controls, knots, rational; then the numbers
            const double* h = take(4);
            int degree = (int)h[0], n = (int)h[1], k = (int)h[2];
            bool rational = h[3] != 0;
            const double* ctrl = take(2 * (size_t)n);
            const double* knots = take((size_t)k);
            const double* weights = rational ? take((size_t)n) : nullptr;
            TColgp_Array1OfPnt poles(1, n);
            for (int i = 0; i < n; i++) poles.SetValue(i + 1, gp_Pnt(ctrl[2 * i], ctrl[2 * i + 1], 0));
            // The flat knot vector as distinct knots and how often each comes.
            std::vector<double> u;
            std::vector<int> m;
            for (int i = 0; i < k; i++) {
                if (!u.empty() && std::fabs(knots[i] - u.back()) < 1e-12) m.back()++;
                else {
                    u.push_back(knots[i]);
                    m.push_back(1);
                }
            }
            TColStd_Array1OfReal uk(1, (int)u.size());
            TColStd_Array1OfInteger um(1, (int)m.size());
            for (size_t i = 0; i < u.size(); i++) {
                uk.SetValue((int)i + 1, u[i]);
                um.SetValue((int)i + 1, m[i]);
            }
            Handle(Geom_BSplineCurve) c;
            if (rational) {
                TColStd_Array1OfReal w(1, n);
                for (int i = 0; i < n; i++) w.SetValue(i + 1, weights[i]);
                c = new Geom_BSplineCurve(poles, w, uk, um, degree);
            } else {
                c = new Geom_BSplineCurve(poles, uk, um, degree);
            }
            return BRepBuilderAPI_MakeEdge(c);
        }
        case 4: {  // ellipse: centre, major half-axis, minor over major, start, sweep
            const double* v = take(7);
            double major = std::hypot(v[2], v[3]);
            gp_Elips e(gp_Ax2(gp_Pnt(v[0], v[1], 0), gp_Dir(0, 0, 1), gp_Dir(v[2], v[3], 0)), major, major * v[4]);
            double a = v[5], s = v[6];
            if (std::fabs(s) >= 2 * M_PI - 1e-12) return BRepBuilderAPI_MakeEdge(e);
            if (s >= 0) return BRepBuilderAPI_MakeEdge(e, a, a + s);
            TopoDS_Edge edge = BRepBuilderAPI_MakeEdge(e, a + s, a);
            edge.Reverse();
            return edge;
        }
        default:
            throw Standard_Failure("unknown piece");
    }
    (void)flat;
}

TopoDS_Wire wire(const double*& p, const double* end) {
    if (p >= end) throw Standard_Failure("short");
    int pieces = (int)*p++;
    BRepBuilderAPI_MakeWire w;
    for (int i = 0; i < pieces; i++) {
        TopoDS_Edge e = piece(p, end);
        if (e.IsNull()) throw Standard_Failure("edge");
        w.Add(e);
        if (!w.IsDone()) throw Standard_Failure("wire");
    }
    return w.Wire();
}

double edge_pick_tolerance(const TopoDS_Shape& shape) {
    Bnd_Box box;
    BRepBndLib::Add(shape, box);
    if (box.IsVoid()) return 1e-3;
    double x0, y0, z0, x1, y1, z1;
    box.Get(x0, y0, z0, x1, y1, z1);
    double dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
    double diag = std::sqrt(dx * dx + dy * dy + dz * dz);
    return std::max(1e-4, diag * 1e-2);
}

int closest_edge_index(const TopTools_IndexedMapOfShape& edges, const gp_Pnt& target, double max_dist) {
    int best = -1;
    double best_dist = max_dist;
    TopoDS_Shape vtx = BRepBuilderAPI_MakeVertex(target);
    for (int i = 1; i <= edges.Extent(); i++) {
        const TopoDS_Edge& e = TopoDS::Edge(edges(i));
        if (BRep_Tool::Degenerated(e)) continue;
        BRepExtrema_DistShapeShape dist(vtx, e);
        if (dist.IsDone() && dist.NbSolution() > 0) {
            double d = dist.Value();
            if (d <= best_dist) {
                best_dist = d;
                best = i;
            }
        }
    }
    return best;
}

int closest_face_index(const TopTools_IndexedMapOfShape& faces, const gp_Pnt& target) {
    int best = -1;
    double best_dist = std::numeric_limits<double>::max();
    TopoDS_Shape vtx = BRepBuilderAPI_MakeVertex(target);
    for (int i = 1; i <= faces.Extent(); i++) {
        const TopoDS_Face& f = TopoDS::Face(faces(i));
        BRepExtrema_DistShapeShape dist(vtx, f);
        if (dist.IsDone() && dist.NbSolution() > 0) {
            double d = dist.Value();
            if (d < best_dist) {
                best_dist = d;
                best = i;
            }
        }
    }
    return best;
}

gp_Dir face_mid_normal(const TopoDS_Face& f, gp_Pnt& out_center) {
    GProp_GProps props;
    BRepGProp::SurfaceProperties(f, props);
    out_center = props.CentreOfMass();
    BRepAdaptor_Surface surf(f);
    double u = 0.5 * (surf.FirstUParameter() + surf.LastUParameter());
    double v = 0.5 * (surf.FirstVParameter() + surf.LastVParameter());
    gp_Pnt p;
    gp_Vec du, dv;
    surf.D1(u, v, p, du, dv);
    gp_Vec n = du.Crossed(dv);
    if (n.Magnitude() < 1e-12) return gp_Dir(0, 0, 1);
    if (f.Orientation() == TopAbs_REVERSED) n.Reverse();
    return gp_Dir(n);
}

void emit_hlr_compound(const TopoDS_Shape& compound, double flags, std::vector<double>& out) {
    if (compound.IsNull()) return;
    BRepLib::BuildCurves3d(compound);
    for (TopExp_Explorer ex(compound, TopAbs_EDGE); ex.More(); ex.Next()) {
        const TopoDS_Edge& e = TopoDS::Edge(ex.Current());
        if (BRep_Tool::Degenerated(e)) continue;
        BRepAdaptor_Curve c(e);
        double u0 = c.FirstParameter();
        double u1 = c.LastParameter();
        if (!std::isfinite(u0) || !std::isfinite(u1) || std::fabs(u1 - u0) < 1e-12) continue;
        GeomAbs_CurveType ct = c.GetType();
        if (ct == GeomAbs_Line) {
            gp_Pnt a = c.Value(u0);
            gp_Pnt b = c.Value(u1);
            if (a.Distance(b) < 1e-9) continue;
            // kind=0 (line): [0, flags, ax, ay, bx, by]
            out.push_back(0.0);
            out.push_back(flags);
            out.push_back(a.X());
            out.push_back(a.Y());
            out.push_back(b.X());
            out.push_back(b.Y());
        } else if (ct == GeomAbs_Circle) {
            gp_Circ circ = c.Circle();
            gp_Pnt ctr = circ.Location();
            double r = circ.Radius();
            gp_Pnt p0 = c.Value(u0);
            gp_Pnt pm = c.Value(0.5 * (u0 + u1));
            gp_Pnt p1 = c.Value(u1);
            double a0 = std::atan2(p0.Y() - ctr.Y(), p0.X() - ctr.X());
            double am = std::atan2(pm.Y() - ctr.Y(), pm.X() - ctr.X());
            double a1 = std::atan2(p1.Y() - ctr.Y(), p1.X() - ctr.X());
            double sweep = 0.0;
            if (std::fabs(u1 - u0) >= 2.0 * M_PI - 1e-5) {
                sweep = 2.0 * M_PI;
            } else {
                // Determine signed CCW sweep passing through midpoint angle `am`.
                auto wrap_pos = [](double d) {
                    while (d < 0.0) d += 2.0 * M_PI;
                    while (d >= 2.0 * M_PI) d -= 2.0 * M_PI;
                    return d;
                };
                double ccw_mid = wrap_pos(am - a0);
                double ccw_end = wrap_pos(a1 - a0);
                if (ccw_mid <= ccw_end + 1e-6) {
                    sweep = ccw_end;
                } else {
                    sweep = ccw_end - 2.0 * M_PI;
                }
            }
            // kind=1 (arc/circle): [1, flags, cx, cy, r, start, sweep]
            out.push_back(1.0);
            out.push_back(flags);
            out.push_back(ctr.X());
            out.push_back(ctr.Y());
            out.push_back(r);
            out.push_back(a0);
            out.push_back(sweep);
        } else {
            // General curve (ellipse, bspline, projected silhouette): adaptive polyline.
            GCPnts_QuasiUniformDeflection sampler(c, 1e-4, u0, u1);
            std::vector<gp_Pnt> pts;
            if (sampler.IsDone() && sampler.NbPoints() >= 2) {
                for (int i = 1; i <= sampler.NbPoints(); i++) {
                    pts.push_back(c.Value(sampler.Parameter(i)));
                }
            } else {
                const int N = 16;
                for (int i = 0; i <= N; i++) {
                    double t = u0 + (u1 - u0) * (double)i / (double)N;
                    pts.push_back(c.Value(t));
                }
            }
            if (pts.size() >= 2) {
                // kind=2 (polyline): [2, flags, n_pts, x0, y0, x1, y1, ...]
                out.push_back(2.0);
                out.push_back(flags);
                out.push_back((double)pts.size());
                for (const gp_Pnt& pt : pts) {
                    out.push_back(pt.X());
                    out.push_back(pt.Y());
                }
            }
        }
    }
}

}  // namespace

extern "C" {

// A face in the plane z = 0 from its loops: the count of loops, then each
// loop as its count of pieces and its pieces — see `piece` — the first loop
// the outside, the rest holes. NULL when it does not make a face.
KShape* kocc_face(const double* data, size_t len) {
    try {
        const double* p = data;
        const double* end = data + len;
        if (len == 0) return nullptr;
        int loops = (int)*p++;
        if (loops < 1) return nullptr;
        TopoDS_Wire outer = wire(p, end);
        BRepBuilderAPI_MakeFace f(gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), outer, true);
        for (int i = 1; i < loops; i++) f.Add(wire(p, end));
        if (!f.IsDone()) return nullptr;
        // Each hole wound against its outside, whichever way it was drawn.
        ShapeFix_Face fix(f.Face());
        fix.FixOrientation();
        fix.Perform();
        return wrap(fix.Face());
    } catch (...) {
        return nullptr;
    }
}

KShape* kocc_prism(const KShape* face, double dx, double dy, double dz) {
    try {
        BRepPrimAPI_MakePrism m(face->shape, gp_Vec(dx, dy, dz));
        return m.IsDone() ? wrap(m.Shape()) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

KShape* kocc_revolve(const KShape* face, double px, double py, double pz, double dx, double dy, double dz, double angle) {
    try {
        gp_Ax1 axis(gp_Pnt(px, py, pz), gp_Dir(dx, dy, dz));
        BRepPrimAPI_MakeRevol m(face->shape, axis, angle);
        return m.IsDone() ? wrap(m.Shape()) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

// 0 joins, 1 cuts `b` from `a`, 2 keeps what they share.
KShape* kocc_boolean(const KShape* a, const KShape* b, int op) {
    try {
        TopoDS_Shape out;
        switch (op) {
            case 0: {
                BRepAlgoAPI_Fuse f(a->shape, b->shape);
                if (!f.IsDone()) return nullptr;
                f.SimplifyResult();
                out = f.Shape();
                break;
            }
            case 1: {
                BRepAlgoAPI_Cut c(a->shape, b->shape);
                if (!c.IsDone()) return nullptr;
                c.SimplifyResult();
                out = c.Shape();
                break;
            }
            case 2: {
                BRepAlgoAPI_Common c(a->shape, b->shape);
                if (!c.IsDone()) return nullptr;
                out = c.Shape();
                break;
            }
            default:
                return nullptr;
        }
        return wrap(out);
    } catch (...) {
        return nullptr;
    }
}

// Rounds selected edges (or all edges when `n_pts == 0`) with constant `radius`.
KShape* kocc_fillet(const KShape* s, double radius, const double* edge_pts, size_t n_pts) {
    try {
        if (!s || radius <= 0.0) return nullptr;
        BRepFilletAPI_MakeFillet mf(s->shape);
        TopTools_IndexedMapOfShape edges;
        TopExp::MapShapes(s->shape, TopAbs_EDGE, edges);
        if (edges.Extent() == 0) return nullptr;
        if (n_pts == 0 || !edge_pts) {
            for (int i = 1; i <= edges.Extent(); i++) {
                const TopoDS_Edge& e = TopoDS::Edge(edges(i));
                if (!BRep_Tool::Degenerated(e)) mf.Add(radius, e);
            }
        } else {
            double tol = edge_pick_tolerance(s->shape);
            std::vector<int> picked;
            for (size_t i = 0; i < n_pts; i++) {
                gp_Pnt p(edge_pts[3 * i], edge_pts[3 * i + 1], edge_pts[3 * i + 2]);
                int idx = closest_edge_index(edges, p, tol);
                if (idx > 0 && std::find(picked.begin(), picked.end(), idx) == picked.end()) {
                    picked.push_back(idx);
                    mf.Add(radius, TopoDS::Edge(edges(idx)));
                }
            }
            if (picked.empty()) return nullptr;
        }
        mf.Build();
        if (!mf.IsDone()) return nullptr;
        return wrap(mf.Shape());
    } catch (...) {
        return nullptr;
    }
}

// Chamfers selected edges (or all edges when `n_pts == 0`) symmetrically by `dist`.
KShape* kocc_chamfer(const KShape* s, double dist, const double* edge_pts, size_t n_pts) {
    try {
        if (!s || dist <= 0.0) return nullptr;
        BRepFilletAPI_MakeChamfer mc(s->shape);
        TopTools_IndexedMapOfShape edges;
        TopExp::MapShapes(s->shape, TopAbs_EDGE, edges);
        TopTools_IndexedDataMapOfShapeListOfShape edge_faces;
        TopExp::MapShapesAndAncestors(s->shape, TopAbs_EDGE, TopAbs_FACE, edge_faces);
        if (edges.Extent() == 0) return nullptr;
        auto add_edge = [&](int idx) {
            const TopoDS_Edge& e = TopoDS::Edge(edges(idx));
            if (BRep_Tool::Degenerated(e)) return;
            const TopTools_ListOfShape& fl = edge_faces.FindFromKey(e);
            if (!fl.IsEmpty()) {
                mc.Add(dist, dist, e, TopoDS::Face(fl.First()));
            }
        };
        if (n_pts == 0 || !edge_pts) {
            for (int i = 1; i <= edges.Extent(); i++) add_edge(i);
        } else {
            double tol = edge_pick_tolerance(s->shape);
            std::vector<int> picked;
            for (size_t i = 0; i < n_pts; i++) {
                gp_Pnt p(edge_pts[3 * i], edge_pts[3 * i + 1], edge_pts[3 * i + 2]);
                int idx = closest_edge_index(edges, p, tol);
                if (idx > 0 && std::find(picked.begin(), picked.end(), idx) == picked.end()) {
                    picked.push_back(idx);
                    add_edge(idx);
                }
            }
            if (picked.empty()) return nullptr;
        }
        mc.Build();
        if (!mc.IsDone()) return nullptr;
        return wrap(mc.Shape());
    } catch (...) {
        return nullptr;
    }
}

// Hollows a solid inward by `thickness` (> 0). Removes either the faces closest
// to `open_pts[0..3*n_pts]` or, when `n_pts == 0` and `has_dir != 0`, the extreme
// face facing `(dx, dy, dz)`.
KShape* kocc_shell(
    const KShape* s,
    double thickness,
    const double* open_pts,
    size_t n_pts,
    int has_dir,
    double dx,
    double dy,
    double dz
) {
    try {
        if (!s || thickness <= 0.0) return nullptr;
        TopTools_IndexedMapOfShape faces;
        TopExp::MapShapes(s->shape, TopAbs_FACE, faces);
        if (faces.Extent() == 0) return nullptr;
        TopTools_ListOfShape to_remove;
        if (n_pts > 0 && open_pts) {
            std::vector<int> picked;
            for (size_t i = 0; i < n_pts; i++) {
                gp_Pnt p(open_pts[3 * i], open_pts[3 * i + 1], open_pts[3 * i + 2]);
                int idx = closest_face_index(faces, p);
                if (idx > 0 && std::find(picked.begin(), picked.end(), idx) == picked.end()) {
                    picked.push_back(idx);
                    to_remove.Append(faces(idx));
                }
            }
        } else if (has_dir) {
            gp_Vec want(dx, dy, dz);
            if (want.Magnitude() < 1e-12) return nullptr;
            gp_Dir wdir(want);
            double best_proj = -std::numeric_limits<double>::max();
            int best_idx = -1;
            for (int i = 1; i <= faces.Extent(); i++) {
                const TopoDS_Face& f = TopoDS::Face(faces(i));
                gp_Pnt ctr;
                gp_Dir n = face_mid_normal(f, ctr);
                if (n.Dot(wdir) > 0.98) {
                    double proj = ctr.XYZ().Dot(wdir.XYZ());
                    if (proj > best_proj) {
                        best_proj = proj;
                        best_idx = i;
                    }
                }
            }
            if (best_idx > 0) to_remove.Append(faces(best_idx));
        }
        BRepOffsetAPI_MakeThickSolid mk;
        mk.MakeThickSolidByJoin(s->shape, to_remove, -thickness, 1e-4);
        if (!mk.IsDone()) return nullptr;
        return wrap(mk.Shape());
    } catch (...) {
        return nullptr;
    }
}

// Sweeps a planar profile face/wire along a 3D spine polyline (`n_pts >= 2`).
KShape* kocc_sweep(const KShape* profile, const double* spine_pts, size_t n_pts, int closed) {
    try {
        if (!profile || !spine_pts || n_pts < 2) return nullptr;
        BRepBuilderAPI_MakeWire spine_builder;
        for (size_t i = 0; i + 1 < n_pts; i++) {
            gp_Pnt a(spine_pts[3 * i], spine_pts[3 * i + 1], spine_pts[3 * i + 2]);
            gp_Pnt b(spine_pts[3 * (i + 1)], spine_pts[3 * (i + 1) + 1], spine_pts[3 * (i + 1) + 2]);
            if (a.Distance(b) < 1e-9) continue;
            spine_builder.Add(BRepBuilderAPI_MakeEdge(a, b));
        }
        if (closed && n_pts >= 3) {
            gp_Pnt a(spine_pts[3 * (n_pts - 1)], spine_pts[3 * (n_pts - 1) + 1], spine_pts[3 * (n_pts - 1) + 2]);
            gp_Pnt b(spine_pts[0], spine_pts[1], spine_pts[2]);
            if (a.Distance(b) >= 1e-9) {
                spine_builder.Add(BRepBuilderAPI_MakeEdge(a, b));
            }
        }
        if (!spine_builder.IsDone()) return nullptr;
        TopoDS_Wire spine = spine_builder.Wire();

        // Extract outer wire from profile so MakePipeShell builds a closed solid.
        TopoDS_Wire prof_wire;
        if (profile->shape.ShapeType() == TopAbs_FACE) {
            prof_wire = BRepTools::OuterWire(TopoDS::Face(profile->shape));
        } else {
            TopExp_Explorer ex(profile->shape, TopAbs_WIRE);
            if (ex.More()) prof_wire = TopoDS::Wire(ex.Current());
        }
        if (!prof_wire.IsNull()) {
            BRepOffsetAPI_MakePipeShell ps(spine);
            ps.SetMode(false);  // Frenet / corrected Frenet
            ps.Add(prof_wire, false, false);
            ps.Build();
            if (ps.IsDone()) {
                ps.MakeSolid();
                if (!ps.Shape().IsNull()) return wrap(ps.Shape());
            }
        }
        BRepOffsetAPI_MakePipe pipe(spine, profile->shape);
        return pipe.IsDone() ? wrap(pipe.Shape()) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

// Lofts a solid or shell through `n >= 2` section faces/wires.
KShape* kocc_loft(const KShape* const* sections, size_t n, int solid, int ruled) {
    try {
        if (!sections || n < 2) return nullptr;
        BRepOffsetAPI_ThruSections loft(solid != 0, ruled != 0, 1e-6);
        for (size_t i = 0; i < n; i++) {
            if (!sections[i]) return nullptr;
            TopoDS_Wire w;
            if (sections[i]->shape.ShapeType() == TopAbs_FACE) {
                w = BRepTools::OuterWire(TopoDS::Face(sections[i]->shape));
            } else {
                TopExp_Explorer ex(sections[i]->shape, TopAbs_WIRE);
                if (ex.More()) w = TopoDS::Wire(ex.Current());
            }
            if (w.IsNull()) return nullptr;
            loft.AddWire(w);
        }
        loft.Build();
        if (!loft.IsDone()) return nullptr;
        return wrap(loft.Shape());
    } catch (...) {
        return nullptr;
    }
}

// Applies a mold draft angle (radians) to side faces roughly parallel to the pull
// direction `(dx, dy, dz)` anchored at neutral plane point `(nx, ny, nz)`.
KShape* kocc_draft(
    const KShape* s,
    double angle,
    double dx,
    double dy,
    double dz,
    double nx,
    double ny,
    double nz
) {
    try {
        if (!s || std::fabs(angle) < 1e-9) return nullptr;
        gp_Vec pv(dx, dy, dz);
        if (pv.Magnitude() < 1e-12) return nullptr;
        gp_Dir pull(pv);
        gp_Pln neutral(gp_Pnt(nx, ny, nz), pull);
        BRepOffsetAPI_DraftAngle da(s->shape);
        TopTools_IndexedMapOfShape faces;
        TopExp::MapShapes(s->shape, TopAbs_FACE, faces);
        int added = 0;
        for (int i = 1; i <= faces.Extent(); i++) {
            const TopoDS_Face& f = TopoDS::Face(faces(i));
            gp_Pnt ctr;
            gp_Dir fn = face_mid_normal(f, ctr);
            if (std::fabs(fn.Dot(pull)) < 0.25) {
                da.Add(f, pull, angle, neutral);
                if (da.AddDone()) added++;
                else da.Remove(f);
            }
        }
        if (added == 0) return nullptr;
        da.Build();
        if (!da.IsDone()) return nullptr;
        return wrap(da.Shape());
    } catch (...) {
        return nullptr;
    }
}

// `s` through the affine map whose rows are m[0..4], m[4..8], m[8..12].
// A map that keeps angles moves the shape exactly; one that stretches it
// unevenly makes its surfaces general ones.
KShape* kocc_transform(const KShape* s, const double* m) {
    try {
        gp_GTrsf g;
        g.SetValue(1, 1, m[0]);
        g.SetValue(1, 2, m[1]);
        g.SetValue(1, 3, m[2]);
        g.SetValue(1, 4, m[3]);
        g.SetValue(2, 1, m[4]);
        g.SetValue(2, 2, m[5]);
        g.SetValue(2, 3, m[6]);
        g.SetValue(2, 4, m[7]);
        g.SetValue(3, 1, m[8]);
        g.SetValue(3, 2, m[9]);
        g.SetValue(3, 3, m[10]);
        g.SetValue(3, 4, m[11]);
        // Columns' lengths and angles: a similarity when they are equal and
        // square to one another.
        gp_XYZ c0(m[0], m[4], m[8]), c1(m[1], m[5], m[9]), c2(m[2], m[6], m[10]);
        double l0 = c0.Modulus(), l1 = c1.Modulus(), l2 = c2.Modulus();
        bool similar = std::fabs(l0 - l1) < 1e-12 * l0 && std::fabs(l0 - l2) < 1e-12 * l0 &&
                       std::fabs(c0.Dot(c1)) < 1e-12 * l0 * l1 && std::fabs(c0.Dot(c2)) < 1e-12 * l0 * l2 &&
                       std::fabs(c1.Dot(c2)) < 1e-12 * l1 * l2 && l0 > 0;
        if (similar) {
            gp_Trsf t;
            t.SetValues(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11]);
            BRepBuilderAPI_Transform b(s->shape, t, true);
            return b.IsDone() ? wrap(b.Shape()) : nullptr;
        }
        BRepBuilderAPI_GTransform b(s->shape, g, true);
        return b.IsDone() ? wrap(b.Shape()) : nullptr;
    } catch (...) {
        return nullptr;
    }
}

double kocc_volume(const KShape* s) {
    try {
        GProp_GProps p;
        BRepGProp::VolumeProperties(s->shape, p);
        return p.Mass();
    } catch (...) {
        return NAN;
    }
}

int kocc_valid(const KShape* s) {
    try {
        BRepCheck_Analyzer a(s->shape);
        return a.IsValid() ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

size_t kocc_count_faces(const KShape* s) {
    TopTools_IndexedMapOfShape m;
    TopExp::MapShapes(s->shape, TopAbs_FACE, m);
    return (size_t)m.Extent();
}

size_t kocc_count_edges(const KShape* s) {
    TopTools_IndexedMapOfShape m;
    TopExp::MapShapes(s->shape, TopAbs_EDGE, m);
    return (size_t)m.Extent();
}

// Returns geometric signatures of all faces in `s`:
// 8 doubles per face: [cx, cy, cz, nx, ny, nz, area, surface_kind].
KTopoSig* kocc_face_signatures(const KShape* s) {
    try {
        TopTools_IndexedMapOfShape faces;
        TopExp::MapShapes(s->shape, TopAbs_FACE, faces);
        std::vector<double> out;
        out.reserve((size_t)faces.Extent() * 8);
        for (int i = 1; i <= faces.Extent(); i++) {
            const TopoDS_Face& f = TopoDS::Face(faces(i));
            GProp_GProps props;
            BRepGProp::SurfaceProperties(f, props);
            gp_Pnt ctr;
            gp_Dir n = face_mid_normal(f, ctr);
            BRepAdaptor_Surface surf(f);
            out.push_back(ctr.X());
            out.push_back(ctr.Y());
            out.push_back(ctr.Z());
            out.push_back(n.X());
            out.push_back(n.Y());
            out.push_back(n.Z());
            out.push_back(props.Mass());
            out.push_back((double)(int)surf.GetType());
        }
        KTopoSig* sig = (KTopoSig*)std::malloc(sizeof(KTopoSig));
        sig->count = (size_t)faces.Extent();
        sig->data = (double*)std::malloc(sizeof(double) * (out.size() + 1));
        std::memcpy(sig->data, out.data(), sizeof(double) * out.size());
        return sig;
    } catch (...) {
        return nullptr;
    }
}

// Returns geometric signatures of all edges in `s`:
// 11 doubles per edge: [ax, ay, az, mx, my, mz, bx, by, bz, length, curve_kind].
KTopoSig* kocc_edge_signatures(const KShape* s) {
    try {
        TopTools_IndexedMapOfShape edges;
        TopExp::MapShapes(s->shape, TopAbs_EDGE, edges);
        std::vector<double> out;
        out.reserve((size_t)edges.Extent() * 11);
        for (int i = 1; i <= edges.Extent(); i++) {
            const TopoDS_Edge& e = TopoDS::Edge(edges(i));
            GProp_GProps props;
            BRepGProp::LinearProperties(e, props);
            BRepAdaptor_Curve c(e);
            double u0 = c.FirstParameter(), u1 = c.LastParameter();
            gp_Pnt a = c.Value(u0);
            gp_Pnt m = c.Value(0.5 * (u0 + u1));
            gp_Pnt b = c.Value(u1);
            out.push_back(a.X());
            out.push_back(a.Y());
            out.push_back(a.Z());
            out.push_back(m.X());
            out.push_back(m.Y());
            out.push_back(m.Z());
            out.push_back(b.X());
            out.push_back(b.Y());
            out.push_back(b.Z());
            out.push_back(props.Mass());
            out.push_back((double)(int)c.GetType());
        }
        KTopoSig* sig = (KTopoSig*)std::malloc(sizeof(KTopoSig));
        sig->count = (size_t)edges.Extent();
        sig->data = (double*)std::malloc(sizeof(double) * (out.size() + 1));
        std::memcpy(sig->data, out.data(), sizeof(double) * out.size());
        return sig;
    } catch (...) {
        return nullptr;
    }
}

void kocc_topo_sig_free(KTopoSig* sig) {
    if (!sig) return;
    std::free(sig->data);
    std::free(sig);
}

// Projects `s` through OCCT's exact Hidden-Line Removal engine (`HLRBRep_Algo`)
// along view direction `view` with horizontal right axis `x_axis`.
// Flags bitmask: bit 0 = hidden, bit 1 = silhouette/smooth outline.
KHlrBuffer* kocc_project_hlr(
    const KShape* s,
    double vx,
    double vy,
    double vz,
    double rx,
    double ry,
    double rz
) {
    try {
        if (!s) return nullptr;
        gp_Vec vv(vx, vy, vz);
        gp_Vec rv(rx, ry, rz);
        if (vv.Magnitude() < 1e-12 || rv.Magnitude() < 1e-12) return nullptr;
        gp_Ax2 cs(gp_Pnt(0, 0, 0), gp_Dir(vv), gp_Dir(rv));
        Handle(HLRBRep_Algo) hlr = new HLRBRep_Algo();
        hlr->Add(s->shape);
        hlr->Projector(HLRAlgo_Projector(cs));
        hlr->Update();
        hlr->Hide();
        HLRBRep_HLRToShape shapes(hlr);
        std::vector<double> out;
        emit_hlr_compound(shapes.VCompound(), 0.0, out);         // visible sharp
        emit_hlr_compound(shapes.OutLineVCompound(), 2.0, out);  // visible silhouette
        emit_hlr_compound(shapes.HCompound(), 1.0, out);         // hidden sharp
        emit_hlr_compound(shapes.OutLineHCompound(), 3.0, out);  // hidden silhouette
        KHlrBuffer* buf = (KHlrBuffer*)std::malloc(sizeof(KHlrBuffer));
        buf->len = out.size();
        buf->data = (double*)std::malloc(sizeof(double) * (out.size() + 1));
        if (!out.empty()) std::memcpy(buf->data, out.data(), sizeof(double) * out.size());
        return buf;
    } catch (...) {
        return nullptr;
    }
}

void kocc_hlr_free(KHlrBuffer* b) {
    if (!b) return;
    std::free(b->data);
    std::free(b);
}

// The shape as triangles no further than `deflection` from it, each with
// the face it is on. NULL when it cannot be meshed. Free with
// `kocc_mesh_free`.
KMesh* kocc_mesh(const KShape* s, double deflection, double angle) {
    try {
        BRepMesh_IncrementalMesh mesher(s->shape, deflection, false, angle, true);
        TopTools_IndexedMapOfShape faces;
        TopExp::MapShapes(s->shape, TopAbs_FACE, faces);
        std::vector<double> pos;
        std::vector<uint32_t> idx, face;
        for (int i = 1; i <= faces.Extent(); i++) {
            const TopoDS_Face& f = TopoDS::Face(faces(i));
            TopLoc_Location loc;
            Handle(Poly_Triangulation) t = BRep_Tool::Triangulation(f, loc);
            if (t.IsNull()) continue;
            const gp_Trsf& tr = loc.Transformation();
            uint32_t base = (uint32_t)(pos.size() / 3);
            for (int n = 1; n <= t->NbNodes(); n++) {
                gp_Pnt p = t->Node(n).Transformed(tr);
                pos.push_back(p.X());
                pos.push_back(p.Y());
                pos.push_back(p.Z());
            }
            bool reversed = f.Orientation() == TopAbs_REVERSED;
            for (int k = 1; k <= t->NbTriangles(); k++) {
                int a, b, c;
                t->Triangle(k).Get(a, b, c);
                if (reversed) std::swap(b, c);
                idx.push_back(base + a - 1);
                idx.push_back(base + b - 1);
                idx.push_back(base + c - 1);
                face.push_back((uint32_t)(i - 1));
            }
        }
        KMesh* m = (KMesh*)std::malloc(sizeof(KMesh));
        m->verts = pos.size() / 3;
        m->tris = face.size();
        m->positions = (double*)std::malloc(sizeof(double) * (pos.size() + 1));
        m->indices = (uint32_t*)std::malloc(sizeof(uint32_t) * (idx.size() + 1));
        m->faces = (uint32_t*)std::malloc(sizeof(uint32_t) * (face.size() + 1));
        std::memcpy(m->positions, pos.data(), sizeof(double) * pos.size());
        std::memcpy(m->indices, idx.data(), sizeof(uint32_t) * idx.size());
        std::memcpy(m->faces, face.data(), sizeof(uint32_t) * face.size());
        return m;
    } catch (...) {
        return nullptr;
    }
}

void kocc_mesh_free(KMesh* m) {
    if (!m) return;
    std::free(m->positions);
    std::free(m->indices);
    std::free(m->faces);
    std::free(m);
}

// Reads a STEP (AP203/AP214/AP242) file from `path`, scaling lengths to metres.
KShape* kocc_read_step(const char* path) {
    try {
        if (!path) return nullptr;
        STEPControl_Controller::Init();
        Interface_Static::SetCVal("xstep.cascade.unit", "M");
        STEPControl_Reader reader;
        if (reader.ReadFile(path) != IFSelect_RetDone) return nullptr;
        reader.TransferRoots();
        TopoDS_Shape s = reader.OneShape();
        if (s.IsNull()) return nullptr;
        return wrap(s);
    } catch (...) {
        return nullptr;
    }
}

// Explodes a compound STEP/B-rep shape into its individual solids (or shells if
// no solids are present), writing handles to `out_shapes[0..*out_count]`.
int kocc_explode_solids(const KShape* s, KShape*** out_shapes, size_t* out_count) {
    try {
        if (!s || !out_shapes || !out_count) return 0;
        std::vector<TopoDS_Shape> parts;
        for (TopExp_Explorer ex(s->shape, TopAbs_SOLID); ex.More(); ex.Next()) {
            parts.push_back(ex.Current());
        }
        if (parts.empty()) {
            for (TopExp_Explorer ex(s->shape, TopAbs_SHELL); ex.More(); ex.Next()) {
                parts.push_back(ex.Current());
            }
        }
        if (parts.empty() && !s->shape.IsNull()) {
            parts.push_back(s->shape);
        }
        KShape** arr = (KShape**)std::malloc(sizeof(KShape*) * (parts.size() + 1));
        for (size_t i = 0; i < parts.size(); i++) arr[i] = new KShape{parts[i]};
        *out_shapes = arr;
        *out_count = parts.size();
        return 1;
    } catch (...) {
        return 0;
    }
}

void kocc_shape_array_free(KShape** arr) { std::free(arr); }

// The shapes as one STEP AP214/AP242 file at `path`, its lengths in millimetres
// when `millimetres`, else metres. 1 when written.
int kocc_write_step(const KShape* const* shapes, size_t n, const char* path, int millimetres) {
    try {
        STEPControl_Controller::Init();
        Interface_Static::SetCVal("write.step.schema", "AP214IS");
        Interface_Static::SetCVal("write.step.unit", millimetres ? "MM" : "M");
        // Kreate's shapes are in metres whatever the file says.
        Interface_Static::SetCVal("xstep.cascade.unit", "M");
        STEPControl_Writer w;
        for (size_t i = 0; i < n; i++)
            if (w.Transfer(shapes[i]->shape, STEPControl_AsIs) != IFSelect_RetDone) return 0;
        return w.Write(path) == IFSelect_RetDone ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

int kocc_write_step_named(
    const KShape* const* shapes,
    const char* const* names,
    size_t n,
    const char* path,
    int millimetres,
    int ap242
) {
    try {
        STEPControl_Controller::Init();
        Interface_Static::SetCVal("write.step.schema", ap242 ? "AP242DIS" : "AP214IS");
        Interface_Static::SetCVal("write.step.unit", millimetres ? "MM" : "M");
        Interface_Static::SetCVal("xstep.cascade.unit", "M");
        STEPControl_Writer w;
        for (size_t i = 0; i < n; i++) {
            if (names && names[i] && names[i][0] != '\0') {
                Interface_Static::SetCVal("write.step.product.name", names[i]);
            }
            if (w.Transfer(shapes[i]->shape, STEPControl_AsIs) != IFSelect_RetDone) return 0;
        }
        return w.Write(path) == IFSelect_RetDone ? 1 : 0;
    } catch (...) {
        return 0;
    }
}

void kocc_free(KShape* s) { delete s; }

}  // extern "C"
