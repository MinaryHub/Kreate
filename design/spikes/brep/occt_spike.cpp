// P0 spike: the test set of design/brep-kernel.md §4, against OCCT.
// Each test prints PASS/FAIL, the numbers it measured, and its time.
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepCheck_Analyzer.hxx>
#include <BRepFilletAPI_MakeFillet.hxx>
#include <BRepGProp.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepOffsetAPI_DraftAngle.hxx>
#include <BRepOffsetAPI_MakePipe.hxx>
#include <BRepOffsetAPI_MakeThickSolid.hxx>
#include <BRepOffsetAPI_ThruSections.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <BRepAdaptor_Surface.hxx>
#include <BRepAdaptor_Curve.hxx>
#include <BRep_Tool.hxx>
#include <GC_MakeCircle.hxx>
#include <GC_MakeArcOfCircle.hxx>
#include <GProp_GProps.hxx>
#include <HLRBRep_Algo.hxx>
#include <HLRBRep_HLRToShape.hxx>
#include <HLRAlgo_Projector.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <STEPControl_Reader.hxx>
#include <STEPControl_Writer.hxx>
#include <Interface_Static.hxx>
#include <TopExp.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_IndexedMapOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <gp_Ax2.hxx>
#include <gp_Circ.hxx>
#include <gp_Dir.hxx>
#include <gp_Pln.hxx>
#include <gp_Pnt.hxx>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <string>

static double volume(const TopoDS_Shape& s) {
    GProp_GProps p;
    BRepGProp::VolumeProperties(s, p);
    return p.Mass();
}

static bool valid(const TopoDS_Shape& s) {
    BRepCheck_Analyzer a(s);
    return !s.IsNull() && a.IsValid();
}

static int count(const TopoDS_Shape& s, TopAbs_ShapeEnum t) {
    TopTools_IndexedMapOfShape m;
    TopExp::MapShapes(s, t, m);
    return m.Extent();
}

static std::map<std::string, int> surface_kinds(const TopoDS_Shape& s) {
    std::map<std::string, int> out;
    for (TopExp_Explorer e(s, TopAbs_FACE); e.More(); e.Next()) {
        BRepAdaptor_Surface a(TopoDS::Face(e.Current()));
        const char* k = "other";
        switch (a.GetType()) {
            case GeomAbs_Plane: k = "plane"; break;
            case GeomAbs_Cylinder: k = "cylinder"; break;
            case GeomAbs_Cone: k = "cone"; break;
            case GeomAbs_Sphere: k = "sphere"; break;
            case GeomAbs_Torus: k = "torus"; break;
            case GeomAbs_BSplineSurface: k = "bspline"; break;
            default: break;
        }
        out[k]++;
    }
    return out;
}

static void print_kinds(const std::map<std::string, int>& k) {
    for (auto& [n, c] : k) printf(" %s=%d", n.c_str(), c);
}

static int failures = 0;

static void run(const char* name, const std::function<bool(std::string&)>& f) {
    auto t0 = std::chrono::steady_clock::now();
    std::string note;
    bool ok = false;
    try {
        ok = f(note);
    } catch (const Standard_Failure& e) {
        note = std::string("exception: ") + e.GetMessageString();
    } catch (...) {
        note = "exception";
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("%s  %-34s %8.1f ms  %s\n", ok ? "PASS" : "FAIL", name, ms, note.c_str());
    if (!ok) failures++;
}

// 100 x 60 x 10 mm plate, four 6.6 mm holes 10 mm in from the corners, a
// 20 mm boss 15 mm tall on top. Metres, as Kreate keeps them.
static TopoDS_Shape plate_with_holes_and_boss() {
    TopoDS_Shape plate = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 0.1, 0.06, 0.01).Shape();
    for (double x : {0.01, 0.09})
        for (double y : {0.01, 0.05}) {
            TopoDS_Shape hole = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(x, y, -0.001), gp_Dir(0, 0, 1)), 0.0033, 0.012).Shape();
            plate = BRepAlgoAPI_Cut(plate, hole).Shape();
        }
    TopoDS_Shape boss = BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0.05, 0.03, 0.01), gp_Dir(0, 0, 1)), 0.01, 0.015).Shape();
    return BRepAlgoAPI_Fuse(plate, boss).Shape();
}

int main(int argc, char** argv) {
    const char* step_path = argc > 1 ? argv[1] : "spike.step";
    const double pi = M_PI;
    TopoDS_Shape part;

    run("1 plate, 4 holes cut, boss joined", [&](std::string& note) {
        part = plate_with_holes_and_boss();
        double want = 0.1 * 0.06 * 0.01 - 4 * pi * 0.0033 * 0.0033 * 0.01 + pi * 0.01 * 0.01 * 0.015;
        double v = volume(part);
        char b[128];
        snprintf(b, sizeof b, "volume err %.2e, faces %d", std::fabs(v - want) / want, count(part, TopAbs_FACE));
        note = b;
        return valid(part) && std::fabs(v - want) / want < 1e-6;
    });

    TopoDS_Shape filleted;
    run("2a fillet 4 selected edges R2", [&](std::string& note) {
        // The plate's four vertical corner edges: chosen by where they are,
        // as a stored edge reference would find them again.
        BRepFilletAPI_MakeFillet f(part);
        int n = 0;
        TopTools_IndexedMapOfShape edges;
        TopExp::MapShapes(part, TopAbs_EDGE, edges);
        for (int i = 1; i <= edges.Extent(); i++) {
            TopoDS_Edge edge = TopoDS::Edge(edges(i));
            BRepAdaptor_Curve c(edge);
            if (c.GetType() != GeomAbs_Line) continue;
            gp_Pnt a = c.Value(c.FirstParameter()), b = c.Value(c.LastParameter());
            bool vertical = std::fabs(a.X() - b.X()) < 1e-9 && std::fabs(a.Y() - b.Y()) < 1e-9;
            bool corner = (std::fabs(a.X()) < 1e-9 || std::fabs(a.X() - 0.1) < 1e-9) &&
                          (std::fabs(a.Y()) < 1e-9 || std::fabs(a.Y() - 0.06) < 1e-9);
            if (vertical && corner) {
                f.Add(0.002, edge);
                n++;
            }
        }
        f.Build();
        if (!f.IsDone()) return false;
        filleted = f.Shape();
        char b[160];
        snprintf(b, sizeof b, "edges %d, valid %d, kinds:", n, valid(filleted));
        note = b;
        auto k = surface_kinds(filleted);
        for (auto& [kk, c] : k) note += " " + kk + "=" + std::to_string(c);
        return n == 4 && valid(filleted) && k["cylinder"] >= 4 + 4 + 1;
    });

    run("2b fillet the boss's top edge R3", [&](std::string& note) {
        BRepFilletAPI_MakeFillet f(filleted);
        for (TopExp_Explorer e(filleted, TopAbs_EDGE); e.More(); e.Next()) {
            TopoDS_Edge edge = TopoDS::Edge(e.Current());
            BRepAdaptor_Curve c(edge);
            if (c.GetType() == GeomAbs_Circle && std::fabs(c.Circle().Radius() - 0.01) < 1e-9 &&
                std::fabs(c.Circle().Location().Z() - 0.025) < 1e-9)
                f.Add(0.003, edge);
        }
        f.Build();
        if (!f.IsDone()) return false;
        filleted = f.Shape();
        auto k = surface_kinds(filleted);
        note = "torus faces " + std::to_string(k["torus"]);
        return valid(filleted) && k["torus"] >= 1;
    });

    run("2c STEP AP214 write and read back", [&](std::string& note) {
        Interface_Static::SetCVal("write.step.schema", "AP214IS");
        Interface_Static::SetCVal("write.step.unit", "MM");
        STEPControl_Writer w;
        if (w.Transfer(filleted, STEPControl_AsIs) != IFSelect_RetDone) return false;
        if (w.Write(step_path) != IFSelect_RetDone) return false;
        STEPControl_Reader r;
        if (r.ReadFile(step_path) != IFSelect_RetDone) return false;
        r.TransferRoots();
        TopoDS_Shape back = r.OneShape();
        auto a = surface_kinds(filleted), b = surface_kinds(back);
        double va = volume(filleted), vb = volume(back);
        note = "read back:";
        for (auto& [kk, c] : b) note += " " + kk + "=" + std::to_string(c);
        char buf[64];
        snprintf(buf, sizeof buf, ", volume err %.1e", std::fabs(va - vb) / va);
        note += buf;
        return a == b && std::fabs(va - vb) / va < 1e-6;
    });

    run("3 shell 2 mm, top open", [&](std::string& note) {
        TopoDS_Shape box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 0.1, 0.06, 0.04).Shape();
        TopTools_ListOfShape open;
        for (TopExp_Explorer e(box, TopAbs_FACE); e.More(); e.Next()) {
            BRepAdaptor_Surface s(TopoDS::Face(e.Current()));
            if (s.GetType() == GeomAbs_Plane && std::fabs(s.Plane().Location().Z() - 0.04) < 1e-9 &&
                std::fabs(std::fabs(s.Plane().Axis().Direction().Z()) - 1) < 1e-9)
                open.Append(e.Current());
        }
        BRepOffsetAPI_MakeThickSolid m;
        m.MakeThickSolidByJoin(box, open, -0.002, 1e-6);
        if (!m.IsDone()) return false;
        double want = 0.1 * 0.06 * 0.04 - 0.096 * 0.056 * 0.038;
        double v = volume(m.Shape());
        char b[96];
        snprintf(b, sizeof b, "volume err %.2e", std::fabs(v - want) / want);
        note = b;
        return valid(m.Shape()) && std::fabs(v - want) / want < 1e-6;
    });

    // Shelling a filleted part with through holes: tried several ways, to
    // find where it gives way.
    auto shell_try = [&](const char* name, const TopoDS_Shape& solid, double t, bool open_bottom, GeomAbs_JoinType join) {
        run(name, [&, t, open_bottom, join](std::string& note) {
            TopTools_ListOfShape open;
            if (open_bottom)
                for (TopExp_Explorer e(solid, TopAbs_FACE); e.More(); e.Next()) {
                    BRepAdaptor_Surface s(TopoDS::Face(e.Current()));
                    if (s.GetType() == GeomAbs_Plane && std::fabs(s.Plane().Location().Z()) < 1e-9 &&
                        std::fabs(std::fabs(s.Plane().Axis().Direction().Z()) - 1) < 1e-9)
                        open.Append(e.Current());
                }
            BRepOffsetAPI_MakeThickSolid m;
            m.MakeThickSolidByJoin(solid, open, -t, 1e-6, BRepOffset_Skin, false, false, join);
            if (!m.IsDone()) {
                note = "not done";
                return false;
            }
            note = "valid " + std::to_string(valid(m.Shape()));
            return valid(m.Shape()) && volume(m.Shape()) > 0 && volume(m.Shape()) < volume(solid);
        });
    };
    TopoDS_Shape plain_plate = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 0.1, 0.06, 0.01).Shape();
    TopoDS_Shape boss_only = BRepAlgoAPI_Fuse(plain_plate, BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(0.05, 0.03, 0.01), gp_Dir(0, 0, 1)), 0.01, 0.015).Shape()).Shape();
    shell_try("3b1 shell plate+boss, bottom open", boss_only, 0.0015, true, GeomAbs_Arc);
    shell_try("3b2 shell holed part, bottom open", part, 0.0015, true, GeomAbs_Arc);
    shell_try("3b3 shell holed part, closed", part, 0.0015, false, GeomAbs_Arc);
    shell_try("3b4 shell filleted part, bottom open", filleted, 0.0015, true, GeomAbs_Arc);
    shell_try("3b5 shell filleted part, intersection join", filleted, 0.0015, true, GeomAbs_Intersection);
    shell_try("3b6 shell filleted part, closed", filleted, 0.0015, false, GeomAbs_Arc);

    run("3b7 shell first, then fillet outside", [&](std::string& note) {
        TopTools_ListOfShape open;
        for (TopExp_Explorer e(part, TopAbs_FACE); e.More(); e.Next()) {
            BRepAdaptor_Surface s(TopoDS::Face(e.Current()));
            if (s.GetType() == GeomAbs_Plane && std::fabs(s.Plane().Location().Z()) < 1e-9 &&
                std::fabs(std::fabs(s.Plane().Axis().Direction().Z()) - 1) < 1e-9)
                open.Append(e.Current());
        }
        BRepOffsetAPI_MakeThickSolid m;
        m.MakeThickSolidByJoin(part, open, -0.0015, 1e-6);
        if (!m.IsDone()) return false;
        TopoDS_Shape shelled = m.Shape();
        BRepFilletAPI_MakeFillet f(shelled);
        TopTools_IndexedMapOfShape edges;
        TopExp::MapShapes(shelled, TopAbs_EDGE, edges);
        int n = 0;
        for (int i = 1; i <= edges.Extent(); i++) {
            TopoDS_Edge edge = TopoDS::Edge(edges(i));
            BRepAdaptor_Curve c(edge);
            if (c.GetType() != GeomAbs_Line) continue;
            gp_Pnt a = c.Value(c.FirstParameter()), b = c.Value(c.LastParameter());
            bool vertical = std::fabs(a.X() - b.X()) < 1e-9 && std::fabs(a.Y() - b.Y()) < 1e-9;
            bool corner = (std::fabs(a.X()) < 1e-9 || std::fabs(a.X() - 0.1) < 1e-9) &&
                          (std::fabs(a.Y()) < 1e-9 || std::fabs(a.Y() - 0.06) < 1e-9);
            if (vertical && corner) { f.Add(0.002, edge); n++; }
        }
        f.Build();
        if (!f.IsDone()) { note = "fillet not done"; return false; }
        note = "edges " + std::to_string(n) + ", valid " + std::to_string(valid(f.Shape()));
        return n == 4 && valid(f.Shape());
    });

    run("3c draft 3 degrees on the box sides", [&](std::string& note) {
        TopoDS_Shape box = BRepPrimAPI_MakeBox(gp_Pnt(0, 0, 0), 0.1, 0.06, 0.04).Shape();
        BRepOffsetAPI_DraftAngle d(box);
        gp_Pln neutral(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1));
        int n = 0;
        for (TopExp_Explorer e(box, TopAbs_FACE); e.More(); e.Next()) {
            BRepAdaptor_Surface s(TopoDS::Face(e.Current()));
            if (std::fabs(s.Plane().Axis().Direction().Z()) < 1e-9) {
                d.Add(TopoDS::Face(e.Current()), gp_Dir(0, 0, 1), 3.0 * pi / 180.0, neutral);
                n++;
            }
        }
        d.Build();
        if (!d.IsDone()) return false;
        note = "faces drafted " + std::to_string(n);
        return valid(d.Shape()) && volume(d.Shape()) < 0.1 * 0.06 * 0.04;
    });

    run("4a revolve a cup", [&](std::string& note) {
        BRepBuilderAPI_MakePolygon p;
        for (auto [x, z] : std::initializer_list<std::pair<double, double>>{
                 {0, 0}, {0.03, 0}, {0.03, 0.05}, {0.028, 0.05}, {0.028, 0.002}, {0, 0.002}})
            p.Add(gp_Pnt(x, 0, z));
        p.Close();
        TopoDS_Face face = BRepBuilderAPI_MakeFace(p.Wire());
        TopoDS_Shape cup = BRepPrimAPI_MakeRevol(face, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))).Shape();
        double want = pi * (0.03 * 0.03 * 0.05 - 0.028 * 0.028 * 0.048);
        double v = volume(cup);
        char b[64];
        snprintf(b, sizeof b, "volume err %.2e", std::fabs(v - want) / want);
        note = b;
        return valid(cup) && std::fabs(v - want) / want < 1e-6;
    });

    run("4b sweep a 10 mm pipe along a bent path", [&](std::string& note) {
        TopoDS_Edge a = BRepBuilderAPI_MakeEdge(gp_Pnt(0, 0, 0), gp_Pnt(0.1, 0, 0));
        Handle(Geom_TrimmedCurve) arc = GC_MakeArcOfCircle(gp_Pnt(0.1, 0, 0), gp_Pnt(0.1 + 0.03 * std::sin(pi / 4), 0.03 - 0.03 * std::cos(pi / 4), 0), gp_Pnt(0.13, 0.03, 0)).Value();
        TopoDS_Edge b = BRepBuilderAPI_MakeEdge(arc);
        TopoDS_Edge c = BRepBuilderAPI_MakeEdge(gp_Pnt(0.13, 0.03, 0), gp_Pnt(0.13, 0.1, 0));
        TopoDS_Wire path = BRepBuilderAPI_MakeWire(a, b, c);
        TopoDS_Edge circle = BRepBuilderAPI_MakeEdge(gp_Circ(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(1, 0, 0)), 0.005));
        TopoDS_Face profile = BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakeWire(circle));
        BRepOffsetAPI_MakePipe pipe(path, profile);
        pipe.Build();
        if (!pipe.IsDone()) return false;
        double len = 0.1 + 0.03 * pi / 2 + 0.07;
        double want = pi * 0.005 * 0.005 * len;
        double v = volume(pipe.Shape());
        char buf[64];
        snprintf(buf, sizeof buf, "volume err %.2e", std::fabs(v - want) / want);
        note = buf;
        return valid(pipe.Shape()) && std::fabs(v - want) / want < 1e-4;
    });

    run("4c loft a square to a circle", [&](std::string& note) {
        BRepBuilderAPI_MakePolygon sq(gp_Pnt(-0.02, -0.02, 0), gp_Pnt(0.02, -0.02, 0), gp_Pnt(0.02, 0.02, 0), gp_Pnt(-0.02, 0.02, 0), true);
        TopoDS_Wire top = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(gp_Circ(gp_Ax2(gp_Pnt(0, 0, 0.05), gp_Dir(0, 0, 1)), 0.015)));
        BRepOffsetAPI_ThruSections loft(true, true);
        loft.AddWire(sq.Wire());
        loft.AddWire(top);
        loft.Build();
        if (!loft.IsDone()) return false;
        double v = volume(loft.Shape());
        char b[64];
        snprintf(b, sizeof b, "volume %.3e m3", v);
        note = b;
        return valid(loft.Shape()) && v > pi * 0.015 * 0.015 * 0.05 && v < 0.04 * 0.04 * 0.05;
    });

    run("5 hidden lines: front view of the part", [&](std::string& note) {
        Handle(HLRBRep_Algo) hlr = new HLRBRep_Algo();
        hlr->Add(filleted);
        // From the front: looking along +Y.
        HLRAlgo_Projector proj(gp_Ax2(gp_Pnt(0, -1, 0), gp_Dir(0, -1, 0), gp_Dir(1, 0, 0)));
        hlr->Projector(proj);
        hlr->Update();
        hlr->Hide();
        HLRBRep_HLRToShape out(hlr);
        TopoDS_Shape visible = out.VCompound(), hidden = out.HCompound(), outline = out.OutLineVCompound();
        int circles = 0, lines = 0;
        for (TopoDS_Shape s : {visible, hidden, outline}) {
            if (s.IsNull()) continue;
            for (TopExp_Explorer e(s, TopAbs_EDGE); e.More(); e.Next()) {
                BRepAdaptor_Curve c(TopoDS::Edge(e.Current()));
                if (c.GetType() == GeomAbs_Line) lines++;
                else circles++;
            }
        }
        note = "lines " + std::to_string(lines) + ", curves " + std::to_string(circles);
        return lines > 0;
    });

    run("6 tessellate for display (0.01 mm)", [&](std::string& note) {
        BRepMesh_IncrementalMesh m(filleted, 1e-5, false, 0.1, true);
        int tris = 0;
        for (TopExp_Explorer e(filleted, TopAbs_FACE); e.More(); e.Next()) {
            TopLoc_Location l;
            auto t = BRep_Tool::Triangulation(TopoDS::Face(e.Current()), l);
            if (!t.IsNull()) tris += t->NbTriangles();
        }
        note = "triangles " + std::to_string(tris);
        return tris > 0;
    });

    printf("\n%d failed\n", failures);
    return failures == 0 ? 0 : 1;
}
