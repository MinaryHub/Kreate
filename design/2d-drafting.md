# 2D 모드 (제도) — 설계

상태: 0단계 결정 반영 (2026-10-26). 1단계부터 이 문서를 기준으로 구현한다.

## 1. 목표와 범위

**목표.** AutoCAD(LT 수준)의 2D 제도를 Kreate 안에서 한다. 같은 도면이 구속·치수로 구동되는 파라메트릭 스케치이기도 해서, 그 영역이 3D 피처(돌출·회전 …)의 입력이 된다. 2D와 3D는 같은 문서, 같은 장면이고 언제든 오간다. 기존 스케치 모드는 이 모드로 흡수된다.

**결정 사항** (검토 회신 기준)

| # | 결정 |
|---|---|
| 1 | 범위: AutoCAD 제도 수준 — 레이어, 선종류·선가중치, 블록, 해치, 주석 치수·치수 스타일, 텍스트, 레이아웃과 출력까지 |
| 2 | 교환 포맷: DXF 읽기/쓰기. DWG는 범위 밖 (공개 구현이 GPL뿐이라 라이선스와 충돌) |
| 3 | 도면은 3D 장면 안의 **도면 객체**. 2D만 있는 문서는 도면 객체 하나짜리 `.kreate` |
| 4 | 치수는 하나의 엔티티, **구동/피구동 토글** |
| 5 | 명령 입력은 **F3 팔레트 + CAD 별칭** (아래 6.3) |
| 6 | 1차 대상: 기계 부품 제도. 가공(레이저·CNC)과 건축은 이후 확장 — 지금 설계가 막지 않아야 함 |
| 7 | 2D/3D 전환: **F2** (6.1) |
| 8 | 기본 치수 스타일: **ISO-25** |
| 9 | 별칭 표는 **사용자가 고칠 수 있음** (6.3) |
| 10 | DXF 쓰기: 기본 **AC1027**, 가공 장비용 **R12 호환** 선택지 (10장) |

**범위 밖.** DWG, AutoLISP/스크립트 언어 호환, AutoCAD식 3D 솔리드 명령, CAM 경로 생성(가공은 DXF 출력까지만 1차 고려).

## 2. 현재 상태 (출발점)

- 스케치는 **객체**다. 기하는 객체 메시의 정점과 `Object::sketch_lines`(정점 쌍), 그 위에 `Object::sketch: Vec<Constraint>` 17종 ([document.rs](../crates/kreate-scene/src/document.rs)).
- **원·호·스플라인·다각형은 엔티티가 아니라 구속**이다. 원은 "중심 둘레 링 점들이 같은 거리", 스플라인은 통과점 사이 샘플 점. 모든 곡선이 처음부터 점열이다 ([sketch.rs](../crates/kreate-scene/src/sketch.rs)).
- 솔버: 전체 스케치를 한 번에, 수치 야코비안 + dense 가우스 소거 LM. 점 수십~수백 개용.
- 영역 탐색: 점과 선분의 평면 그래프 ([regions.rs](../crates/kreate-mesh/src/regions.rs)). 피처는 스케치 Uid + 영역 시드 `[f64; 2]`로 영역을 고른다 ([feature.rs](../crates/kreate-scene/src/feature.rs)).
- 매개변수 바인딩은 `Bound::SketchDimension { sketch, index }` — **구속의 순서 번호**에 걸려 있다.
- 화면 표시: `SketchView`를 egui 페인터로 오버레이. GPU 2D 선 렌더러 없음.
- 연산자 `sketch.*` 26개, 도구 상태기계는 앱 크레이트 [sketching.rs](../apps/kreate/src/sketching.rs), 커넥터 `get_sketch`.
- 2D 입출력: 3D 부품의 투영선을 DXF/SVG로 쓰는 [drawing.rs](../crates/kreate-io/src/drawing.rs)(`LINE`, `TEXT`)뿐. 읽기 없음.
- 저장: `ObjectRecord.sketch`, `sketch_lines`, 포맷 1.4.

요약: 구속 솔버·도구 상태기계·영역→피처 연결은 재사용할 가치가 있고, **기하 표현(점열)과 솔버 규모, 화면 표시는 교체**해야 한다.

## 3. 데이터 모델

### 3.1 도면

구현 메모(1단계): `ObjectData`에 변형을 더하면 약 70곳의 `match`가 바뀌므로, 지금의 스케치·피처와 같이 **객체의 필드** `Object::drawing: Option<Box<Drawing>>`로 둔다. 엔티티 ID는 도면마다 세는 `EntityId(u32)`. 공유되는 블록·스타일만 문서 수준.

```rust
// kreate-scene::drawing
pub struct Object { …, pub drawing: Option<Box<Drawing>> }   // 스케치 필드를 대체

pub struct Drawing {
    pub name: String,
    pub entities: Arena<EntityId, Entity>,          // 순서 = 그린 순서 (그리기 순서는 별도 draw_order)
    pub layers: Vec<Layer>,                         // 0번은 항상 "0"
    pub constraints: Vec<Constraint>,               // EntityRef 참조
    pub current: DrawState,                         // 현재 레이어·색·선종류 …
    pub units: DrawingUnits,                        // 표시 단위, 각도 단위, 정밀도
    pub layouts: Vec<Layout>,                       // 3단계
}
```

- **평면 = 객체 변환의 로컬 XY.** 면에 붙은 스케치는 지금처럼 부모·`reattach`를 쓴다. 2D 전용 문서는 월드 XY 위의 도면 하나.
- **블록, 선종류, 텍스트 스타일, 치수 스타일은 문서 수준 데이터블록**(재질처럼 공유, 도면 여러 장이 같은 표제란 블록을 씀).
- 모든 엔티티는 **안정 Uid**를 가진다. 구속·치수·바인딩·피처 축·커넥터·공동 편집이 모두 이 ID로 참조한다 (지금의 "순서 번호" 참조를 없앤다).

### 3.2 엔티티

```rust
pub struct Entity {
    pub uid: Uid,
    pub kind: EntityKind,
    pub layer: LayerId,
    pub color: ColorRef,        // ByLayer | ByBlock | Aci(u8) | Rgb
    pub linetype: LinetypeRef,  // ByLayer | ByBlock | Named(LinetypeId)
    pub lineweight: Lineweight, // ByLayer | ByBlock | Mm(f32)
    pub construction: bool,     // 보조선: 영역·출력에 안 들어감
}

pub enum EntityKind {
    Point(DVec2),
    Curve(Curve2D),
    Text(Text),                 // 한 줄 / MText(서식 부분집합)
    Hatch(Hatch),               // 경계 루프(곡선 참조 또는 복사본) + 패턴/단색
    Insert(Insert),             // 블록 참조: 위치·회전·축척·속성값
    Dimension(Dimension),
    Leader(Leader),             // 지시선·풍선(BOM 번호)
}

pub enum Curve2D {
    Line { a: DVec2, b: DVec2 },
    Circle { c: DVec2, r: f64 },
    Arc { c: DVec2, r: f64, start: f64, sweep: f64 },   // 반시계, 라디안
    Ellipse { c: DVec2, major: DVec2, ratio: f64, start: f64, sweep: f64 },
    Polyline { verts: Vec<(DVec2, f64)>, closed: bool }, // (점, bulge) — DXF LWPOLYLINE 그대로
    Spline { degree: u8, ctrl: Vec<DVec2>, knots: Vec<f64>, weights: Option<Vec<f64>>,
             fit: Vec<DVec2> },                           // fit 점이 있으면 fit 점이 원본
}
```

- 모든 좌표는 **f64**. 건축 규모(수 km)에서도 정밀도 문제가 없게 하고, 화면 표시는 카메라 기준 상대 좌표로 f32 변환한다.
- 이산화는 **소비하는 곳에서만**: 화면(화면 오차 기준), 영역·피처(설정 가능한 현 오차), 내보내기 메시. 저장·편집·DXF는 해석 곡선 그대로.

### 3.3 구속

```rust
pub struct EntityRef { pub entity: Uid, pub part: Part }
pub enum Part { Whole, Start, End, Center, Mid, Vertex(u32) /* 폴리라인 꼭짓점 */ }

pub enum Constraint {
    Coincident(EntityRef, EntityRef), Horizontal(EntityRef), Vertical(EntityRef),
    Parallel(Uid, Uid), Perpendicular(Uid, Uid), Tangent(Uid, Uid), Equal(Uid, Uid),
    Concentric(Uid, Uid), PointOnCurve(EntityRef, Uid), Symmetric(EntityRef, EntityRef, Uid),
    Midpoint(EntityRef, Uid), Fixed(EntityRef), Smooth(Uid, Uid) /* G2 */,
}
```

치수는 구속이 아니라 **엔티티**다 (3.4). 구동 치수는 솔버에 방정식을 하나 더 넣는다.

### 3.4 치수

```rust
pub struct Dimension {
    pub kind: DimKind,          // Linear{H|V|Aligned}, Angular, Radius, Diameter, ArcLength,
                                // Ordinate{X|Y}, Baseline/Continue 는 Linear 묶음
    pub refs: Vec<EntityRef>,   // 측정 대상
    pub text_at: DVec2,         // 사용자가 놓은 글자 위치
    pub style: DimStyleId,
    pub driving: bool,          // true: 값이 기하를 구동 / false: 기하를 재서 표시
    pub value: f64,             // 구동일 때 목표값, 피구동일 때 마지막 측정값
    pub text_override: Option<String>,  // "<>" 치환, 공차·접두어(Ø, R, M)
    pub tolerance: Option<Tolerance>,   // ±, 한계, 끼워맞춤(H7 등)
}
```

- 구동 ↔ 피구동 전환은 한 번의 연산. 과구속이 되는 구동 치수는 만들 때 거부하고 "피구동으로 둘까요"를 제안 (Fusion 방식).
- 매개변수 바인딩: `Bound::SketchDimension { sketch, index }` → **`Bound::DrawingDimension { drawing: Uid, dim: Uid }`**.

### 3.5 레이어·스타일

- `Layer { name, color, linetype, lineweight, on, frozen, locked, plot }`. "0"과 "Defpoints"는 DXF 의미 그대로.
- 기계 제도 기본 레이어 세트(설정에서 바꿈): 외형선, 숨은선, 중심선, 치수, 해치, 보조선, 표제란 — ISO 128 선종류·굵기.
- 선종류: CONTINUOUS, DASHED, HIDDEN, CENTER, PHANTOM, DOT, DASHDOT, BORDER, DIVIDE (+ ×2, ×0.5). 패턴은 도면 단위 길이, 전역 `ltscale`.
- 치수 스타일: **ISO-25 기본**(mm, 기계), ANSI 선택. 화살표, 글자 높이, 간격, 치수보조선 띄움/연장, 단위 정밀도, 공차 표기.

## 4. 기하 커널: 새 크레이트 `kreate-geom2d`

`kreate-mesh`처럼 GPU·UI 의존 없이 wasm까지 빌드되는 독립 크레이트.

- **교차**: 선·원·호·타원·폴리라인 사이 해석해, 스플라인은 분할 + 뉴턴. 접점·겹침(동일 직선/동심원) 판정 포함.
- **최근접점·투영·접선·법선**, 곡선 위 매개변수 ↔ 점.
- **트림/연장/끊기**: 교차 매개변수로 자르고 늘림. 폴리라인 세그먼트 단위.
- **오프셋**: 선·호는 정확, 폴리라인은 bulge 보존 오프셋 + 자기교차 정리, 스플라인은 근사(허용 오차 명시).
- **필렛/모따기**: 아무 두 곡선 쌍 (선-선, 선-호, 호-호), 폴리라인 꼭짓점 전부.
- **영역(경계) 탐색**: 곡선 평면 그래프(호 그대로) → 면·구멍. 해치 경계 자동 찾기와 피처 프로파일이 같은 코드를 쓴다. 지금의 `regions.rs`를 곡선판으로 대체.
- **허용 오차 정책**: 도면 범위 기준 상대 1e-9, 점 일치 판정은 별도 `eps_point`. 문서에 한 곳에서 정의.
- 견고성: 결정적 난수 속성 테스트(교차의 대칭성, 트림 후 끝점이 교차점 위, 오프셋 거리 일정)와, `kreate-io`처럼 변이 테스트.

## 5. 솔버 v2

- **변수화**: 점(x,y), 선(두 끝점), 원(cx,cy,r), 호(cx,cy,r,θ0,θ1), 타원, 스플라인 제어점. 구속·구동 치수는 **해석적 야코비안**.
- **연결 요소 분해**: 구속 그래프의 연결 요소마다 따로 푼다. 제도 도면은 대부분 비구속이라, 수만 엔티티 도면에서 실제 풀이는 작은 묶음 몇 개다.
- **희소 LM** (요소가 크면): 정규방정식 대신 희소 QR 또는 CG. 첫 구현은 요소별 dense로 두되 요소 크기 상한을 측정.
- **드래그**: 끌리는 점에 약한 목표 방정식, 나머지는 "제자리 유지" 가중치 — 지금 `sketch.move`의 거동을 보존.
- **자유도 표시**: 요소마다 DOF를 계산해 완전 구속(검정)·미구속(파랑)·과구속(빨강)을 Fusion처럼 색으로.
- 실패 시: 수렴 못 하면 변경을 되돌리고 어떤 구속이 충돌하는지 보고.

## 6. 모드·화면·입력

### 6.1 모드

- `Mode::Sketch` → **`Mode::Drafting`** ("2D"). 들어가면: 도면 평면에 맞춘 정사영, 궤도 회전 잠금(팬·줌만), 2D 그리드, 2D 배경.
- **2D/3D 전환은 F2.** 지금 F2는 오브젝트 모드의 "이름 바꾸기"(Blender 관례)라서, 이름 바꾸기는 **Ctrl+F2**와 아웃라이너 더블클릭으로 옮긴다. 키맵 편집기에서 되돌릴 수 있다.
- **3D 전환**: F2로 3D 보기로 나간다. 도면은 3D에서 평면 위 선으로 계속 보이고, 3D에서 도면 객체를 고르고 들어오면 다시 2D.
- 새 문서를 2D로 시작하는 진입점: 환영 카드와 File ▸ New ▸ 2D Drawing.
- 워크스페이스 "Drafting": 뷰포트 + 레이어 패널 + 특성(선택 엔티티) + 상태 표시줄 토글(SNAP, GRID, ORTHO, POLAR, OSNAP, OTRACK, DYN, LWT).

### 6.2 선택과 편집

- 클릭 선택, **윈도(왼→오, 완전히 안)** / **크로싱(오→왼, 걸치면)**, Shift로 빼기, 올가미.
- **그립**: 끝점·중점·중심·사분점을 끌어 편집. 구속이 있으면 솔버를 거쳐 움직인다.
- 2D 피킹은 GPU ID 버퍼가 아니라 **CPU 공간 색인**(그리드 또는 R-tree)으로 한다 — 커서 근처 허용 거리 안의 최근접 곡선과 스냅 후보가 필요하기 때문.

### 6.3 명령 입력: F3 + 별칭 (5번 회신에 대한 권장)

일관성을 위해 **F3 팔레트를 유일한 명령 입력으로** 두되, CAD 사용자의 손버릇을 받는다.

- F3 팔레트에 **별칭 표**를 둔다: `L` 선, `PL` 폴리선, `C` 원, `A` 호, `REC` 사각형, `POL` 다각형, `SPL` 스플라인, `EL` 타원, `TR` 자르기, `EX` 연장, `O` 간격띄우기, `F` 모깎기, `CHA` 모따기, `M` 이동, `CO` 복사, `RO` 회전, `SC` 축척, `MI` 대칭, `AR` 배열, `S` 늘리기, `BR` 끊기, `J` 결합, `X` 분해, `DIM`/`DLI`/`DAL`/`DAN`/`DRA`/`DDI` 치수, `H` 해치, `B` 블록, `I` 삽입, `T`/`MT` 문자, `LA` 레이어, `MA` 특성 일치.
- 별칭이 정확히 일치하면 **Enter 한 번에 실행**(목록에서 고르지 않아도 됨). 2D 모드에서는 **F3 없이 바로 글자를 쳐도** 팔레트가 그 글자로 열린다 — AutoCAD 명령줄과 같은 손놀림이지만 UI는 하나다.
- 명령 중의 숫자·좌표는 **커서 옆 동적 입력**: `25` (현재 방향으로 길이), `25<30` (길이·각도), `@10,5` (상대), `#10,5` (절대), Tab으로 필드 이동. 지금 `sketching.rs`의 숫자 입력을 일반화.
- Enter/Space로 **직전 명령 반복**, Esc 취소, 명령 중 `U`로 한 단계 되돌리기.
- 별칭은 언어와 무관하게 영어(한국 AutoCAD 사용자도 영어 별칭을 씀). 명령 이름과 설명은 한/영 번역.
- **별칭 표는 사용자가 고친다.** 회사마다 AutoCAD의 `acad.pgp`를 고쳐 쓰는 일이 흔하고, 손에 익은 별칭이 다르면 도구를 바꾸지 않는다. 키맵 편집기 옆 "명령 별칭" 표에서 추가·변경·삭제하고, 설정 폴더의 텍스트 파일로 저장한다(키맵과 같은 방식). `acad.pgp`를 그대로 가져오기를 지원하면 이전이 한 번에 끝난다.

### 6.4 스냅

- **객체 스냅**: 끝점, 중점, 중심, 사분점, 교차, 겉보기 교차, 연장, 접점, 수직, 평행, 근처, 노드, 삽입점. 기본 켜짐 세트는 끝점·중점·중심·교차.
- **추적**: 직교(F8), 극좌표 각도 증분, 객체 스냅 추적(정렬선).
- 스냅 표시 기호와 툴팁은 AutoCAD와 같은 모양(사용자가 이미 아는 언어).

## 7. 화면 표시: 2D 선 렌더러

`kreate-render`에 새 패스. egui 오버레이는 엔티티 수천 개에서 한계가 온다.

- 선분: 인스턴스 사각형(화면 공간 굵기, 안티앨리어싱), 곡선은 **화면 오차 기준 적응 분할**(줌마다 다시, 캐시).
- 선종류: 셰이더에서 **곡선을 따라 누적한 거리**로 패턴 — 패턴이 꼭짓점마다 끊기지 않게 폴리라인 단위로 거리 누적.
- 선가중치: 화면에서는 픽셀(LWT 토글), 출력에서는 mm.
- 문자: `kreate-fonts` 외곽선을 삼각형 메시로(작은 크기에서도 정확), 같은 글자는 캐시.
- 해치: 단색은 삼각분할 채우기, 패턴은 경계로 자른 선 묶음(ANSI31 등 표준 패턴).
- 성능 목표: **선분 10만 개 도면을 통합 GPU에서 팬·줌 60fps**, 열기 1초 이내.

## 8. 3D와의 연결

- **피처**: 도면의 닫힌 영역이 그대로 돌출·회전·… 의 프로파일. 피처는 지금처럼 `drawing: Uid` + 영역 시드. 회전축 `axis: [u32; 2]`(점 번호) → `AxisRef::Entity(Uid) | DrawingX | DrawingY`.
- 이산화는 피처를 풀 때 현 오차(설정, 기본 0.01 mm)로. 이후 "해석 곡면" 작업에서 호→원기둥 정보를 그대로 넘길 수 있게 곡선 출처를 유지한다.
- **3D → 2D 도면**(3단계): 부품을 정면·평면·측면·등각·단면·상세로 투영해 도면 객체에 넣는다. [drawing.rs](../crates/kreate-io/src/drawing.rs)의 은선 처리를 재사용하되 결과는 엔티티(선·호)로, 숨은선은 숨은선 레이어로. 부품이 바뀌면 다시 투영하고, 사용자가 붙인 치수는 투영 엔티티의 Uid로 따라간다.

## 9. 레이아웃과 출력 (3단계)

- `Layout { paper: PaperSize, viewports: Vec<Viewport2D>, entities }` — 표제란·테두리는 레이아웃의 블록, 뷰포트는 모델 영역을 축척으로 보여주는 창.
- 주석 축척: 치수·문자를 "출력 시 3.5 mm"로 정의하고 뷰포트 축척에 맞춰 표시.
- 출력: **PDF(벡터)·SVG**, 선가중치는 mm, 레이어 `plot` 플래그 반영. PDF는 `pdf-writer`(MIT/Apache) 검토.
- 표제란 속성(도면명·번호·재질·축척·작성자)은 문서 정보와 물리 재질에서 채운다. 부품 목록(BOM) 표는 기존 parts list에서.

## 10. DXF

- **읽기/쓰기 ASCII DXF.** 읽기는 R12~최신 전부.
- **쓰기 버전 판단**: 선택지를 여럿 두면 사용자가 고를 근거가 없고 테스트 조합만 늘어난다. 실제로 갈리는 경우는 둘뿐이다.
  - **AC1027 (2013), 기본** — AutoCAD·LibreCAD·QCAD·FreeCAD·DraftSight가 모두 읽고, LWPOLYLINE(bulge), SPLINE, MTEXT, HATCH, 트루컬러, 선가중치, XDATA를 쓸 수 있어 도면이 손실 없이 간다. 더 새 버전(2018)은 2D 엔티티에 얻는 것이 없고 읽는 프로그램만 줄어든다.
  - **R12 (AC1009), "가공 장비 호환"** — 레이저 커터·플라스마·CNC 소프트웨어 상당수가 R12만 제대로 읽는다. LINE·ARC·CIRCLE·POLYLINE(bulge)·TEXT만 쓰고, 스플라인과 타원은 호/선분으로 근사(허용 오차 지정), 해치·치수는 분해해서 선으로, 레이어와 색(ACI)은 유지한다. 가공 확장(13장)의 출발점이 된다.
  - 그래서 내보내기 대화상자에는 "DXF"와 "DXF (가공 장비용 R12)" 두 가지만 둔다.
- 엔티티: POINT, LINE, CIRCLE, ARC, ELLIPSE, LWPOLYLINE, POLYLINE, SPLINE, TEXT, MTEXT(서식 부분집합), HATCH, INSERT(+ATTRIB), DIMENSION, LEADER/MLEADER(읽기), SOLID.
- 테이블: LAYER, LTYPE, STYLE, DIMSTYLE, BLOCK_RECORD, 헤더 `$INSUNITS`, `$LTSCALE`, `$DIMSCALE`.
- **치수는 익명 블록(`*D…`)도 같이 써야** 다른 프로그램에서 보인다 — 치수 그래픽을 블록으로 생성.
- **구속·구동 여부·Kreate 전용 정보**는 XDATA / 확장 사전(앱 이름 `KREATE`)에 넣는다. 다른 프로그램은 무시하고, Kreate로 다시 열면 파라메트릭 도면으로 복원된다.
- 모르는 엔티티는 원본 그룹 코드 그대로 보관했다가 다시 쓴다(컨테이너의 `UnknownEntry`와 같은 원칙).
- 구현: 자체 리더/라이터(다른 포맷과 같은 방식, 변이 테스트 하네스에 편입). 외부 `dxf` 크레이트는 엔티티 커버리지와 유지 상태를 확인 후 참고만.
- 테스트 자료: LibreCAD·QCAD·FreeCAD로 만든 실제 DXF 묶음 + 골든 파일 + 왕복 비교.

## 11. 저장 포맷과 이전

- 컨테이너 **minor 5**: `SceneFile::drawings`, `ObjectRecord::drawing`, 문서 수준 `blocks`, `linetypes`, `text_styles`, `dim_styles`.
- **이전 단계 4→5** (`migrate.rs`, JSON 위에서):
  - `sketch_lines` → Line 엔티티 (점 일치 → Coincident).
  - `Circle`/`Arc` 구속 링 → Circle/Arc 엔티티 (반지름·각도는 링 점에서), 중심→첫 점 `Distance` → 구동 지름/반지름 치수.
  - `Spline` → Spline(fit 점 = `through`), `Polygon` → 닫힌 Polyline + Equal 구속.
  - 나머지 구속 → `EntityRef` 기반, 순서 번호 → Uid 대응표를 만들어 `Bound::SketchDimension`과 피처 축을 다시 건다.
  - 스케치 객체의 메시는 버린다(점은 엔티티로 옮겨짐).
- **규칙대로, 이전은 실제 옛 파일로 테스트한다.** 그래서 0단계에서 **현재 빌드로 모든 스케치 종류가 든 골든 `.kreate`를 먼저 커밋**한다. 검증: 옛 파일에서 만든 피처 솔리드의 부피·경계가 이전 후와 같아야 한다.
- 이전 버전(0.3.x)이 새 파일을 열면 도면은 `extra`로 보존되지만 보이지 않는다 — 릴리스 노트에 명시.

## 12. 연산자·커넥터·공동 편집

- 새 네임스페이스 **`draft.*`** (`draft.line`, `draft.trim`, `draft.dimension`, `draft.layer_set` …). 각 도구는 연산자 하나 + 앱 쪽 모달 상태기계(`sketching.rs`를 `drafting.rs`로 일반화).
- `sketch.*`는 한 메이저 버전 동안 **별칭**으로 유지(스크립트, 저장된 대화, MCP 프롬프트 호환). 인자는 점 번호 → Uid로 바뀌므로 별칭이 변환한다.
- 커넥터: `get_sketch` → **`get_drawing`**(엔티티·레이어·치수·구속·DOF·영역), 그리기 도구 노출, 지시문 갱신. "이 부품의 정면도를 그리고 치수를 넣어 줘"가 목표 시나리오.
- 공동 편집: 엔티티 Uid 단위 변경을 동기화. `kreate-collab`의 현재 단위(객체 단위인지)를 1단계에서 확인.
- 실행 취소: 연산자 하나가 한 단계, 드래그 솔브는 한 단계로 합침.

## 13. 향후 확장을 막지 않기 (가공·건축)

- **가공**: 열린 윤곽 찾기·닫기 검사, 커프(kerf) 오프셋, 작업별 레이어 → DXF. 곡선이 해석적이라 G-code의 G2/G3로 바로 이어질 수 있다.
- **건축**: 다중선(벽), 문·창 블록과 동적 속성, 면적·방 계산, 인치·피트 단위와 분수 표기, km 좌표(f64 + 카메라 상대 표시로 이미 대응).
- 둘 다 엔티티 종류·스타일·명령의 **추가**로 들어오도록, 엔티티와 명령을 레지스트리 방식으로 둔다.

## 14. 테스트 전략

- 기하: 해석 기대값 단위 테스트 + 결정적 난수 속성 테스트.
- 솔버: 수렴 모음(구속 조합별), DOF 계산, 실패 보고.
- DXF: 골든 + 왕복 + 실제 파일 묶음 + 변이 테스트.
- 이전: 옛 골든 `.kreate` → 피처 솔리드 동일성.
- UI: 도구별 composite 스냅샷(한/영), 별칭·동적 입력 상태기계 단위 테스트.
- 성능: 10만 선분 도면 열기·팬·줌 벤치(CI에는 수치 기록, 회귀 시 경고).

## 15. 위험

| 위험 | 대응 |
|---|---|
| 데이터 모델 교체가 피처·바인딩·커넥터·공동 편집 전부를 지나감 | 1단계 완료 조건을 "기존 스케치 테스트 전부 이전 + 옛 파일의 솔리드 동일"로 못박음. 새 기능은 그 뒤 |
| 솔버 규모 | 연결 요소 분해를 처음부터, 요소 크기 계측 |
| DXF 치수·MTEXT 호환 | 실제 파일 묶음으로 조기 검증, 치수 블록 생성은 2단계 초반 |
| 작은 글자·선종류 화면 품질 | 렌더러 프로토타입을 2단계 첫 주에, 스냅샷으로 확인 |
| 범위 팽창 (AutoCAD는 거대함) | 아래 단계별 완료 조건 밖은 다음 단계로 |

## 16. 단계와 완료 조건

| 단계 | 내용 | 완료 조건 | 기간(감) |
|---|---|---|---|
| **0 설계** | 이 문서 승인, 현재 빌드로 스케치 골든 파일 커밋, DXF 실물 파일 수집 | 문서 승인 | 1주 |
| **1 토대** | `kreate-geom2d`, `Drawing`·엔티티·기본 레이어, 솔버 v2, 4→5 이전, `Mode::Drafting`(정사영·그리드·CPU 피킹·기본 스냅), 기존 도구 전부를 곡선 위로 이식, 피처·바인딩·커넥터·`sketch.*` 별칭 | 기존 스케치 테스트 전부 통과, 옛 파일 솔리드 동일, 커넥터로 그리기 가능 | 5~7주 |
| **2 제도** | GPU 2D 렌더러(선종류·선가중치·문자·해치), 레이어 패널, 편집 명령 전부, 그립·윈도/크로싱, 객체 스냅·추적·동적 입력, F3 별칭, 주석 치수 전 종류·치수 스타일(ISO/ANSI)·공차, 문자·스타일, 해치, 블록·속성, DXF 읽기/쓰기 | 실제 DXF 묶음 왕복, 기계 부품 도면 한 장을 처음부터 끝까지 그려 LibreCAD에서 같게 보임 | 6~8주 |
| **3 도면·출력** | 레이아웃·뷰포트·표제란, 주석 축척, PDF/SVG 출력, 3D→2D 투영 뷰(정면·단면·상세)와 갱신, 중심선 자동, 기하공차(GD&T) 틀, BOM 표, 커넥터 제도 도구 | 3D 부품에서 치수 넣은 도면 PDF가 나옴 | 4~6주 |
| **4 확장** | 가공·건축 | 별도 문서 | — |

## 17. 결정 기록

1. 2D/3D 전환: F2. 기존 F2(이름 바꾸기)는 Ctrl+F2와 아웃라이너 더블클릭으로 이동.
2. 기본 치수 스타일: ISO-25. ANSI는 선택지로.
3. 명령 별칭: 사용자 편집 가능, `acad.pgp` 가져오기.
4. DXF 쓰기: 기본 AC1027, 가공 장비용 R12 선택지.
