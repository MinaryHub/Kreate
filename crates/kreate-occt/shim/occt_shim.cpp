// The C face of the OpenCASCADE calls Kreate makes: a handful of functions
// over an opaque shape, every one catching what OCCT throws and answering
// NULL (or 0) instead, so nothing C++ ever unwinds into Rust.
//
// A face is handed over as numbers (see `loops_face`); everything else is
// shapes in and shapes out. Lengths are metres, as Kreate keeps them.

#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBuilderAPI_GTransform.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepGProp.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <BRep_Tool.hxx>
#include <GProp_GProps.hxx>
#include <Geom_BSplineCurve.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <Interface_Static.hxx>
#include <STEPControl_Writer.hxx>
#include <ShapeFix_Face.hxx>
#include <ShapeFix_Shape.hxx>
#include <Standard_Failure.hxx>
#include <TColStd_Array1OfInteger.hxx>
#include <TColStd_Array1OfReal.hxx>
#include <TColgp_Array1OfPnt.hxx>
#include <TopExp.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Wire.hxx>
#include <gp_Circ.hxx>
#include <gp_Elips.hxx>
#include <gp_GTrsf.hxx>
#include <gp_Pln.hxx>
#include <gp_Trsf.hxx>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

// The shapes as one STEP AP214 file at `path`, its lengths in millimetres
// when `millimetres`, else metres. 1 when written.
int kocc_write_step(const KShape* const* shapes, size_t n, const char* path, int millimetres) {
    try {
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

void kocc_free(KShape* s) { delete s; }

}  // extern "C"
