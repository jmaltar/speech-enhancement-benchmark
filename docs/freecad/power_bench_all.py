# FreeCAD 1.1 macro: the power-measurement bench for EACH board, as physical parts (2026-11-02, rev. 2).
# One scene per board under test; each is saved as docs/freecad/power_bench_<board>.FCStd and rendered
# to docs/freecad/render/<board>_{top,overview}.png (plus h723_closeup.png, the middle of the bench).
# Run headless through a wrapper that always calls os._exit (a script error leaves the offscreen GUI hanging):
#   QT_QPA_PLATFORM=offscreen ~/Applications/FreeCAD.AppImage <runner.py>
# then crop with docs/freecad/crop.py. Units mm; X right, Y away from the viewer, Z up; table z = 0.
# Parts are schematic stand-ins at roughly real size; header-pin positions are approximate, so the
# instructions say to wire by the printed labels.
#
# ONE fixture board, the spare F411, for every session (rev. 2; rev. 1 used two F411s in rotation):
#   it is the USB breakout AND the logger. It runs the logger firmware (USB peripheral never enabled,
#   A11/A12 left as analog inputs, so the USB data lines pass straight through its header pins), reads
#   the INA226 over I2C (B6 = SCL, B7 = SDA) and reports over UART (A2 = TX, A3 = RX) to the debugger's
#   serial port. The debugger stays wired for good: TXD/RXD/GND to the fixture, SWD (DIO, CLK, GND) to
#   the STM32 being measured so images can be flashed by script; its 3V3 and 5V pins are never connected.
# Microcontroller boards (H723, S3, main F411): laptop -> standard USB-A to USB-C cable -> fixture
#   (5V, G, A11 = D-, A12 = D+ on its header) -> jumpers, 5V through M1 (R100) -> USB-A female breakout
#   -> the board's usual cable -> the board. The S3 flashes over COM, so no SWD wires in its scene.
# Pi 5: official supply -> USB-C pass-through breakout (to buy) with M3 (R010) between its VBUS
#   terminals -> Pi 5. The fixture reads M3; its USB-C only powers it in that session.
import os
import FreeCAD as App
import FreeCADGui as Gui
import Part
import Draft
from FreeCAD import Vector as V

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "render")
os.makedirs(OUT, exist_ok=True)

COL = {
    "laptop": (0.30, 0.30, 0.33), "metal": (0.78, 0.78, 0.80), "black": (0.10, 0.10, 0.10),
    "green_pcb": (0.15, 0.58, 0.32), "purple_pcb": (0.45, 0.22, 0.60), "blue_pcb": (0.15, 0.32, 0.70),
    "bp_pcb": (0.16, 0.16, 0.18), "pi_pcb": (0.10, 0.50, 0.25), "s3_pcb": (0.12, 0.12, 0.14),
    "dbg_pcb": (0.75, 0.10, 0.10),
    "terminal": (0.10, 0.50, 0.90), "gold": (0.90, 0.72, 0.20), "red": (0.90, 0.08, 0.08),
    "white": (0.97, 0.97, 0.97), "wgreen": (0.05, 0.70, 0.20), "wblack": (0.05, 0.05, 0.05),
    "orange": (1.00, 0.55, 0.00), "yellow": (0.95, 0.85, 0.05), "wblue": (0.10, 0.35, 1.00),
    "magenta": (0.85, 0.10, 0.75), "cyan": (0.00, 0.75, 0.85), "purple": (0.50, 0.20, 0.85),
    "brown": (0.55, 0.30, 0.10), "chip": (0.30, 0.30, 0.34), "cable": (0.18, 0.18, 0.18),
    "psu": (0.92, 0.92, 0.90), "eth": (0.55, 0.60, 0.70), "can": (0.70, 0.72, 0.75),
}

# Objects left out of the close-up render (by name prefix); labels go by group "far".
FAR = ("laptop", "c1_aplug", "dbg_aplug", "dbgc", "dut", "h_cable", "h_cplug", "w_swd")

# WeAct Black Pill header, counted from the USB-C end (approximate; wire by the printed labels).
#   back row:  B12 B13 B14 B15 A8 A9 A10 A11 A12 A15 B3 B4 B5 B6 B7 B8 B9 5V G 3V3
#   front row: VB C13 C14 C15 R A0 A1 A2 A3 A4 A5 A6 A7 B0 B1 B2 B10 3V3 G 5V
FIXTURE_PINS = {"A11": ("back", 7), "A12": ("back", 8), "B6": ("back", 13), "B7": ("back", 14),
                "5V": ("back", 17), "G": ("back", 18), "3V3": ("back", 19),
                "A2": ("front", 7), "A3": ("front", 8), "G2": ("front", 18)}
# Debugger's 10-pin header in the order its silkscreen lists the signals.
DBG_PINS = ("3V3", "SWDIO", "SWCLK", "GND", "SWO", "NRST", "TXD", "RXD", "GND2", "5V")


class Scene:
    def __init__(self, name):
        self.doc = App.newDocument(name)
        self.groups = {}

    def add(self, name, shape, color, group="all"):
        o = self.doc.addObject("Part::Feature", name)
        o.Shape = shape
        o.ViewObject.ShapeColor = COL[color]
        self.groups.setdefault(group, []).append(o)
        return o

    def label(self, text, pos, size=6.0, group="all", color=(0.0, 0.0, 0.0)):
        t = Draft.make_text(text if isinstance(text, list) else [text],
                            placement=App.Placement(V(*pos), App.Rotation()))
        t.ViewObject.FontSize = size
        t.ViewObject.DisplayMode = "World"
        t.ViewObject.TextColor = tuple(float(c) for c in color)
        t.ViewObject.LineSpacing = 1.1
        self.groups.setdefault(group, []).append(t)
        return t

    def pin(self, x, y, name, group="all"):
        self.add(name + "_pin", Part.makeCylinder(0.4, 6, V(x, y, 1.6)), "gold", group)
        self.add(name + "_dupont", box(x - 1.3, y - 1.3, 4.5, 2.6, 2.6, 9), "black", group)
        return (x, y, 13.5)

    def bare_pins(self, x0, y, n, name, group="all", skip=(), z0=1.6):
        for k in range(n):
            x = x0 + 2.54 * k
            if any(abs(x - s) < 1.2 for s in skip):
                continue
            self.add(f"{name}_{k}", Part.makeCylinder(0.35, 5, V(x, y, z0)), "gold", group)

    def render(self, name, direction, ortho=False, w=3600, h=2200):
        Gui.Selection.clearSelection()
        v = Gui.ActiveDocument.ActiveView
        v.setCameraType("Orthographic" if ortho else "Perspective")
        v.setViewDirection(direction)
        v.fitAll()
        v.saveImage(os.path.join(OUT, name + ".png"), w, h, "White")

    def set_far_visible(self, visible):
        for o in self.doc.Objects:
            if o.Name.startswith(FAR):
                o.ViewObject.Visibility = visible
        for t in self.groups.get("far", []):
            t.ViewObject.Visibility = visible


def box(x0, y0, z0, dx, dy, dz):
    return Part.makeBox(dx, dy, dz, V(x0, y0, z0))


def tube(points, r):
    s = None
    for a, b in zip(points[:-1], points[1:]):
        a, b = V(*a), V(*b)
        d = b - a
        if d.Length < 1e-6:
            continue
        c = Part.makeCylinder(r, d.Length, a, d)
        s = c if s is None else s.fuse(c)
    for p in points[1:-1]:
        s = s.fuse(Part.makeSphere(r, V(*p)))
    return s


def arch(a, b, z):
    """A jumper wire from pin a up to height z, across, and down onto pin b."""
    return [a, (a[0], a[1], z), (b[0], b[1], z), b]


R = 1.0


def laptop(sc, y_ports, x0=-95, y0=-30, depth=110):
    lap = box(x0, y0, 0, 55, depth, 18)
    for yp in y_ports:
        lap = lap.cut(box(x0 + 43, yp, 6.5, 14, 13, 5.5))
    sc.add("laptop", lap, "laptop")
    sc.label(["Laptop"], (x0 + 5, y0 + depth - 14, 18.1), 6.5, color=(1.0, 1.0, 1.0), group="far")


def usb_a_plug(sc, name, y, x_port=-52):
    sc.add(name + "_metal", box(x_port + 4, y + 1, 7, 20, 11, 4.5), "metal")
    sc.add(name + "_mold", box(x_port + 24, y - 1, 4.5, 16, 15, 9.5), "black")
    return (x_port + 40, y + 6.5, 9.2)


def ina(sc, name, x0, y0, shunt, title):
    """An INA226 breakout: PCB, blue screw terminal (IN+ left, IN- right), five header pins at the back."""
    sc.add(name + "_pcb", box(x0, y0, 0, 36, 26, 1.6), "purple_pcb")
    sc.add(name + "_terminal", box(x0 + 7, y0, 1.6, 22, 8, 9), "terminal")
    inp = (x0 + 12, y0 + 4, 11.6)
    inm = (x0 + 24, y0 + 4, 11.6)
    sc.add(name + "_screw_inp", Part.makeCylinder(1.8, 1, V(inp[0], inp[1], 10.6)), "metal")
    sc.add(name + "_screw_inm", Part.makeCylinder(1.8, 1, V(inm[0], inm[1], 10.6)), "metal")
    sc.add(name + "_shunt", box(x0 + 14, y0 + 12, 1.6, 7, 3.2, 0.8), "black")
    pins = {n: sc.pin(x0 + 5 + 5 * k, y0 + 22, f"{name}_{n}") for k, n in enumerate(("VBS", "GND", "VCC", "SCL", "SDA"))}
    sc.label([title, f"shunt {shunt}"], (x0 + 38, y0 + 18, 0.1), 6)
    sc.label("VBS GND VCC SCL SDA", (x0 + 2, y0 + 28, 0.1), 2.9)
    sc.label("IN+", (x0 - 8, y0 + 1, 0.1), 5, color=(0.8, 0.0, 0.0))
    sc.label("IN-", (x0 + 37, y0 + 1, 0.1), 5, color=(0.8, 0.0, 0.0))
    # VBS (bus-voltage sense) to the IN- side
    sc.add(name + "_w_vbs", tube([pins["VBS"], (pins["VBS"][0], pins["VBS"][1], 21),
                                  (inm[0], inm[1] + 6, 21), inm], 0.7), "red")
    return pins, inp, inm


def black_pill(sc, name, x0, y0, title, used, title_pos, swd_used=(), label_color=(0.0, 0.0, 0.0), title_group="all",
               title_size=5.6):
    """A WeAct Black Pill (53 x 21 mm), USB-C at its left end, 4-pin SWD header at its right end.
    `used` maps pin name -> (row, index); `swd_used` names SWD-header pins (3V3, DIO, CLK, GND) that get a wire.
    Returns (pins, usb-c point); SWD-header pins appear in `pins` as SWD_<name>."""
    sc.add(name + "_pcb", box(x0, y0, 0, 53, 21, 1.6), "bp_pcb")
    sc.add(name + "_usbc", box(x0 - 2, y0 + 5.5, 1.6, 7.5, 9, 3.2), "metal")
    sc.add(name + "_chip", box(x0 + 15, y0 + 6.5, 1.6, 7, 7, 1.2), "chip")
    pos = {}
    for n, (row, k) in used.items():
        pos[n] = (x0 + 3.0 + 2.54 * k, y0 + (19 if row == "back" else 2))
    sc.bare_pins(x0 + 3.0, y0 + 19, 20, name + "_back", skip=[p[0] for n, p in pos.items() if used[n][0] == "back"])
    sc.bare_pins(x0 + 3.0, y0 + 2, 20, name + "_front", skip=[p[0] for n, p in pos.items() if used[n][0] == "front"])
    pins = {n: sc.pin(x, y, f"{name}_{n}") for n, (x, y) in pos.items()}
    for k, n in enumerate(("3V3", "DIO", "CLK", "GND")):
        x, y = x0 + 50.5, y0 + 6.7 + 2.54 * k
        if n in swd_used:
            pins["SWD_" + n] = sc.pin(x, y, f"{name}_swd_{n}")
        else:
            sc.add(f"{name}_swd_{n}_bare", Part.makeCylinder(0.35, 5, V(x, y, 1.6)), "gold")
    sc.label(title, title_pos, title_size, color=label_color, group=title_group)
    return pins, (x0 - 2, y0 + 10)


def fixture(sc, x0, y0, title, title_pos, wired=tuple(FIXTURE_PINS), title_size=5.6):
    """The spare F411: USB breakout + logger. Pins used: see FIXTURE_PINS, plus GND on its SWD header (for the module).
    `wired` limits which header pins get a wire (the Pi session uses no USB pins)."""
    used = {n: rc for n, rc in FIXTURE_PINS.items() if n in wired}
    fx, usbc = black_pill(sc, "fx", x0, y0, title, used, title_pos, swd_used=("GND",), title_size=title_size)
    yb, yf = y0 + 21.5, y0 - 5.5
    for n, dy in (("A11", 0), ("A12", 4), ("B6", 0), ("B7", 4), ("5V", 0), ("G", 4), ("3V3", 0)):
        if n in fx:
            sc.label(n, (fx[n][0] - 1.6, yb + dy, 0.1), 3.0)
    for n, t, dy in (("A2", "A2", 0), ("A3", "A3", -4), ("G2", "G", 0)):
        if n in fx:
            sc.label(t, (fx[n][0] - 1.4, yf + dy, 0.1), 3.0)
    sc.label("GND", (fx["SWD_GND"][0] + 2.0, fx["SWD_GND"][1] - 1, 0.1), 2.6)
    return fx, usbc


def debugger(sc, x0, y0, used, label_pos, label_lines, size=5.0):
    """The debugger (stand-in), its 10-pin header along the back edge; `used` = header pins that get a wire."""
    sc.add("dbg_pcb", box(x0, y0, 0, 45, 18, 1.6), "dbg_pcb")
    sc.add("dbg_chip", box(x0 + 22, y0 + 4, 1.6, 7, 7, 1.2), "chip")
    pos = {}
    for k, n in enumerate(DBG_PINS):
        x, y = x0 + 6 + 2.54 * k, y0 + 15
        if n in used:
            pos[n] = sc.pin(x, y, "dbg_" + n)
        else:
            sc.add(f"dbg_hdr_{k}", Part.makeCylinder(0.35, 5, V(x, y, 1.6)), "gold")
    sc.label(["3V3 DIO CLK GND SWO RST TXD RXD GND 5V"], (x0 + 3, y0 + 21, 0.1), 2.3)
    sc.label(label_lines, label_pos, size)
    return pos, (x0, y0 + 9)


def uart_wires(sc, dbg, fx, z=24):
    """Debugger TXD -> fixture A3 (its RX), RXD <- A2 (its TX), GND -> fixture G."""
    sc.add("w_uart_tx", tube(arch(dbg["TXD"], fx["A3"], z), R), "magenta")
    sc.add("w_uart_rx", tube(arch(dbg["RXD"], fx["A2"], z + 2), R), "cyan")
    sc.add("w_uart_gnd", tube(arch(dbg["GND2"], fx["G2"], z + 4), R), "wblack")


def i2c_wires(sc, fx, m, z=24):
    """Fixture to the module: B6 -> SCL, B7 -> SDA, 3V3 -> VCC, GND (SWD header) -> GND."""
    for n, tgt, c, zz in (("SWD_GND", "GND", "wblack", z), ("3V3", "VCC", "orange", z + 2),
                          ("B7", "SDA", "wblue", z + 4), ("B6", "SCL", "yellow", z + 6)):
        sc.add("w_i2c_" + n, tube(arch(fx[n], m[tgt], zz), R), c)


def female_breakout(sc, x0=170):
    sc.add("bo_pcb", box(x0, 0, 0, 24, 20, 1.6), "green_pcb")
    sc.add("bo_socket", box(x0 + 11, 3.4, 1.6, 15, 13.2, 6.6).cut(box(x0 + 14, 4, 7, 12, 11, 4.5)), "metal")
    pins = {n: sc.pin(x0 + 4, y, "bo_" + n) for n, y in (("VBUS", 17), ("Dm", 12.5), ("Dp", 8), ("GND", 3.5))}
    sc.label(["your USB-A", "female breakout"], (x0 - 4, -24, 0.1), 6)
    for y, t in ((17, "VBUS"), (12.5, "D-"), (8, "D+"), (3.5, "GND")):
        sc.label(t, (x0 - 13, y - 1.5, 0.1), 3.6)
    return pins


def mcu_chain(sc, dut_cable_label, swd):
    """Laptop -> A-to-C cable -> fixture -> M1 -> female breakout -> plug of the board's cable; debugger in front."""
    laptop(sc, y_ports=(3, -26))
    end = usb_a_plug(sc, "c1_aplug", 3)
    sc.add("c1_cable", tube([end, (-2, 9.5, 8), (4, -8, 5), (10, -8, 4), (13, 7.5, 3.4)], 2.3), "cable")
    sc.add("c1_cplug_mold", box(13, 3.5, 1.2, 8, 8, 4.5), "black")
    sc.add("c1_cplug_metal", box(21, 4.5, 2.1, 3, 6, 2.6), "metal")
    sc.label(["standard USB-A to USB-C", "cable into the spare F411"], (-36, 34, 0.1), 5)
    fx, _ = fixture(sc, 24, -2, ["spare F411: USB breakout + logger", "(logger firmware, USB off, no reset jumper)"],
                    (-36, 68, 0.1), title_size=4.6)
    bo = female_breakout(sc)
    m1, inp, inm = ina(sc, "m1", 105, 42, "R100", "M1: INA226")
    sc.add("w_5v_in", tube([fx["5V"], (fx["5V"][0], fx["5V"][1], 22), (95, 32, 24), (inp[0], 42, 20), inp], R), "red")
    sc.add("w_5v_out", tube([inm, (inm[0], 42, 20), (150, 32, 22), (174, 17, 20), bo["VBUS"]], R), "red")
    sc.add("w_dm", tube([fx["A11"], (fx["A11"][0], 17, 15), (fx["A11"][0] + 8, 12.5, 15), (166, 12.5, 15), bo["Dm"]], R), "white")
    sc.add("w_dp", tube([fx["A12"], (fx["A12"][0], 17, 16.5), (fx["A12"][0] + 10, 8, 16.5), (166, 8, 16.5), bo["Dp"]], R), "wgreen")
    sc.add("w_gnd", tube([fx["G"], (fx["G"][0], 17, 18), (fx["G"][0] + 8, 3.5, 18), (166, 3.5, 18), bo["GND"]], R), "wblack")
    i2c_wires(sc, fx, m1, z=28)
    sc.label(["red 5V goes through M1;", "A11 to D-, A12 to D+, G to GND"], (22, 94, 0.1), 5)
    # the plug of the board's usual cable in the breakout
    sc.add("h_plug_metal", box(184, 4, 7, 18, 11, 4.5), "metal")
    sc.add("h_plug_mold", box(202, 2, 4.5, 16, 15, 9.5), "black")
    sc.label(dut_cable_label, (205, -46, 0.1), 6, group="far")
    # the debugger, in front, on the laptop's second port
    used = ("SWDIO", "SWCLK", "GND", "TXD", "RXD", "GND2") if swd else ("TXD", "RXD", "GND2")
    lines = (["debugger, wired for good:", "TXD, RXD, GND to the spare F411;", "DIO, CLK, GND to the board's SWD;",
              "3V3 and 5V never connected"] if swd else
             ["debugger: TXD, RXD, GND to the", "spare F411 only (the S3 flashes", "over COM); 3V3 and 5V never connected"])
    dbg, dend = debugger(sc, 120, -100, used, (30, -86, 0.1), lines, size=4.4)
    aend = usb_a_plug(sc, "dbg_aplug", -26)
    sc.add("dbgc_cable", tube([aend, (0, -19.5, 7), (10, -60, 4), (30, -112, 3.5), (100, -112, 3.5),
                               (dend[0] - 8, dend[1], 3.4)], 2.0), "cable")
    sc.add("dbg_plug", box(dend[0] - 8, dend[1] - 3.5, 1.2, 8, 7, 4), "black")
    uart_wires(sc, dbg, fx, z=22)
    return m1, dbg


def swd_wires(sc, dbg, dut, z=18):
    """Debugger SWDIO/SWCLK/GND to the board's SWD pins: up, back past the debugger, right, then up to the board."""
    for k, (n, tgt, c) in enumerate((("SWDIO", "SWD_DIO", "purple"), ("SWCLK", "SWD_CLK", "brown"), ("GND", "SWD_GND", "wblack"))):
        a, b, zz, yy = dbg[n], dut[tgt], z + 2 * k, -74 - 2.5 * k
        sc.add("w_swd_" + n, tube([a, (a[0], a[1], zz), (a[0], yy, zz), (b[0], yy, zz), (b[0], b[1], zz), b], R), c)


def board_h723(sc, dbg):
    sc.add("h_cable", tube([(218, 9.5, 9.2), (232, 9.5, 8), (246, 28, 5), (256, 28, 3.6)], 2.3), "cable")
    sc.add("h_cplug", box(256, 24, 1.4, 12, 8, 4.5), "black")
    sc.add("dut_pcb", box(268, 0, 0, 42, 62, 1.6), "blue_pcb")
    sc.add("dut_usbc", box(268, 24.5, 1.6, 7, 9, 3.2), "metal")
    sc.add("dut_chip", box(282, 23, 1.6, 16, 16, 1.6), "chip")
    dut = {}
    for k, n in enumerate(("3V3", "DIO", "CLK", "GND")):
        x = 293 + 2.54 * k
        if n == "3V3":
            sc.add("dut_swd_3v3", Part.makeCylinder(0.35, 5, V(x, 4, 1.6)), "gold")
        else:
            dut["SWD_" + n] = sc.pin(x, 4, "dut_swd_" + n)
    sc.label("SWD: 3V3 DIO CLK GND", (288, 8, 1.7), 2.4, color=(1.0, 1.0, 1.0), group="far")
    sc.label(["WeAct H723", "SWD wired, 3V3 pin left free"], (268, 78, 0.1), 5.6, group="far")
    swd_wires(sc, dbg, dut)


def board_s3(sc, dbg):
    sc.add("h_cable", tube([(218, 9.5, 9.2), (232, 9.5, 8), (246, 18, 5), (256, 18, 3.6)], 2.3), "cable")
    sc.add("h_cplug", box(256, 14, 1.4, 12, 8, 4.5), "black")
    sc.add("dut_pcb", box(268, 0, 0, 64, 28, 1.6), "s3_pcb")
    sc.add("dut_usbc_usb", box(268, 14.5, 1.6, 7, 9, 3.2), "metal")
    sc.add("dut_usbc_com", box(268, 2.5, 1.6, 7, 9, 3.2), "metal")
    sc.add("dut_can", box(292, 4, 1.6, 26, 18, 3.2), "can")
    sc.label("USB", (271, 31.0, 0.1), 4.2, group="far")
    sc.label("COM", (271, -6.5, 0.1), 4.2, color=(0.8, 0.0, 0.0), group="far")
    sc.label(["ESP32-S3 board", "cable in the port marked USB;", "COM left unplugged"], (262, -24, 0.1), 5.6, group="far")


def board_f411(sc, dbg):
    sc.add("h_cable", tube([(218, 9.5, 9.2), (232, 9.5, 8), (246, 9.5, 5), (256, 9.5, 3.6)], 2.3), "cable")
    sc.add("h_cplug", box(256, 6, 1.4, 12, 8, 4.5), "black")
    dut, _ = black_pill(sc, "dut", 268, -1, ["main F411 (the board measured)", "SWD wired, 3V3 pin left free"], {},
                        (262, 36, 0.1), swd_used=("DIO", "CLK", "GND"), title_group="far")
    swd_wires(sc, dbg, dut)


def build_mcu(dut):
    sc = Scene("power_bench_" + dut)
    cable = {"h723": ["the H723's usual", "USB-A to USB-C cable"], "s3": ["the S3's usual", "USB-A to USB-C cable"],
             "f411": ["the F411's usual", "USB-A to USB-C cable"]}[dut]
    m1, dbg = mcu_chain(sc, cable, swd=(dut != "s3"))
    {"h723": board_h723, "s3": board_s3, "f411": board_f411}[dut](sc, dbg)
    return sc


def build_pi():
    sc = Scene("power_bench_pi5")
    # the official supply and its captive USB-C cable (label below it: multi-line labels grow downward)
    sc.add("psu", box(-100, -10, 0, 50, 45, 30), "psu")
    sc.label(["official Pi 5", "USB-C supply"], (-100, -16, 0.1), 6)
    sc.add("psu_cable", tube([(-50, 12, 12), (-20, 12, 8), (0, 0, 5), (20, 10, 4), (30, 10, 3.5)], 2.6), "cable")
    sc.add("psu_cplug", box(30, 6, 1.2, 10, 8, 4.5), "black")
    # the USB-C pass-through breakout (to buy): socket in, plug out, VBUS on two screw terminals
    sc.add("pt_pcb", box(40, -2, 0, 36, 24, 1.6), "green_pcb")
    sc.add("pt_in", box(40, 5.5, 1.6, 7.5, 9, 3.2), "metal")
    sc.add("pt_out", box(76, 7, 1.6, 12, 7, 2.6), "metal")
    sc.add("pt_term", box(52, 14, 1.6, 14, 8, 9), "terminal")
    t_in, t_out = (55.5, 18, 11.6), (62.5, 18, 11.6)
    for n, p in (("in", t_in), ("out", t_out)):
        sc.add("pt_screw_" + n, Part.makeCylinder(1.6, 1, V(p[0], p[1], 10.6)), "metal")
    sc.label(["USB-C pass-through breakout (to buy):", "socket in, plug out,", "CC1/CC2 passed through,",
              "VBUS on two screw terminals:", "left = VBUS in, right = VBUS out"], (-12, -16, 0.1), 5.0)
    # the Pi 5: USB-C power port on its left long edge, GPIO header along the right long edge
    x0, y0 = 90, -20
    sc.add("pi_pcb", box(x0, y0, 0, 56, 85, 1.6), "pi_pcb")
    sc.add("pi_usbc", box(x0 - 1.5, 7, 1.6, 7.5, 9, 3.2), "metal")
    sc.add("pi_hdmi1", box(x0 - 1.5, 24, 1.6, 7.5, 7, 3.2), "metal")
    sc.add("pi_hdmi2", box(x0 - 1.5, 37, 1.6, 7.5, 7, 3.2), "metal")
    sc.add("pi_cooler", box(x0 + 12, y0 + 24, 3.0, 30, 32, 8), "black")
    sc.add("pi_eth", box(x0 + 30, y0 + 85 - 4, 1.6, 16, 21, 13.5), "metal")
    sc.add("pi_usbA", box(x0 + 8, y0 + 85 - 4, 1.6, 14, 17, 15.5), "metal")
    xo, xi = x0 + 53, x0 + 50.5          # outer row (even pins), inner row (odd pins)
    yk = lambda k: y0 + 8 + 2.54 * k     # pin pair k holds pins 2k+1 (inner) and 2k+2 (outer)
    p6 = (xo, yk(2))
    for k in range(20):
        for xr in (xi, xo):
            if abs(xr - p6[0]) < 0.5 and abs(yk(k) - p6[1]) < 0.5:
                continue
            sc.add(f"pi_hdr_{k}_{int(xr*10)}", Part.makeCylinder(0.35, 5, V(xr, yk(k), 1.6)), "gold")
    pi_g = sc.pin(*p6, "pi_P6")
    sc.label(["Raspberry Pi 5", "GPIO pin 6 = GND", "(pins 1-2 at this end)"], (x0 + 68, y0 + 18, 0.1), 5.6)
    # M3 to the left of the Pi, behind the pass-through; thick red wires to the terminals
    m3, inp, inm = ina(sc, "m3", 18, 40, "R010", "M3: INA226")
    sc.add("w_vin", tube([t_in, (t_in[0], t_in[1], 22), (inp[0], 30, 22), inp], 1.6), "red")
    sc.add("w_vout", tube([inm, (inm[0], 30, 24), (t_out[0], t_out[1], 24), t_out], 1.6), "red")
    sc.label(["thick wire (18 AWG+)", "in the screw terminals"], (-34, 52, 0.1), 5)
    # GND: M3 GND to the Pi's GPIO pin 6, routed round the board's top-right corner
    gx, gy = m3["GND"][0], m3["GND"][1]
    sc.add("w_m3_gnd_pi", tube([m3["GND"], (gx, gy, 26), (gx, 82, 26), (x0 + 62, 82, 26), (x0 + 62, pi_g[1], 26),
                                (pi_g[0], pi_g[1], 20), pi_g], R), "wblack")
    sc.label("M3 GND to Pi pin 6", (x0 + 64, 76, 0.1), 5)
    # Ethernet out of the Pi
    sc.add("eth_cable", tube([(x0 + 38, y0 + 102, 8), (x0 + 38, y0 + 128, 6), (x0 + 70, y0 + 140, 6)], 2.4), "eth")
    sc.label("Ethernet", (x0 + 44, y0 + 116, 0.1), 5)
    # the fixture (spare F411) reads M3; its USB-C only powers it here; the debugger carries its readings
    laptop(sc, y_ports=(95, 116), x0=-100, y0=70, depth=70)
    fx, usbc = fixture(sc, 0, 100, ["spare F411 = logger (same firmware;", "its USB-C only powers it here)"],
                       (62, 132, 0.1), wired=("B6", "B7", "3V3", "A2", "A3", "G2"))
    end = usb_a_plug(sc, "fx_plug", 95, x_port=-57)
    sc.add("fx_cable", tube([end, (usbc[0] - 8, usbc[1], 3.2)], 2.0), "cable")
    sc.add("fx_cplug", box(usbc[0] - 8, usbc[1] - 3.5, 1.2, 8, 7, 4), "black")
    i2c_wires(sc, fx, m3, z=32)
    dbg, dend = debugger(sc, 4, 140, ("TXD", "RXD", "GND2"), (54, 160, 0.1),
                         ["debugger: TXD, RXD, GND to the", "spare F411; 3V3 and 5V never connected"])
    aend = usb_a_plug(sc, "dbg_aplug", 116, x_port=-57)
    sc.add("dbgc_cable", tube([aend, (-10, 135, 5), (dend[0] - 8, dend[1], 3.4)], 2.0), "cable")
    sc.add("dbg_plug", box(dend[0] - 8, dend[1] - 3.5, 1.2, 8, 7, 4), "black")
    uart_wires(sc, dbg, fx, z=44)
    return sc


for key, builder in (("h723", lambda: build_mcu("h723")), ("s3", lambda: build_mcu("s3")),
                     ("f411", lambda: build_mcu("f411")), ("pi5", build_pi)):
    sc = builder()
    sc.doc.recompute()
    sc.doc.saveAs(os.path.join(HERE, f"power_bench_{key}.FCStd"))
    sc.render(f"{key}_top", (0, 0, -1), ortho=True)
    sc.render(f"{key}_overview", (0.10, 1.0, -1.15))
    if key == "h723":
        sc.set_far_visible(False)
        sc.render("h723_closeup", (0.0, 0.85, -1.0))
        sc.set_far_visible(True)
        sc.doc.recompute()
    App.closeDocument(sc.doc.Name)
    print("RENDERED", key)

import os as _os
_os._exit(0)
