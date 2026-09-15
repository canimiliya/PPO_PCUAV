from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import DroneConfig
from .environment import CANONICAL_WIRES, Scenario
from .simulation import FlightLog


BACKGROUND = (255, 255, 255)
INK = (32, 32, 32)
TOWER = (211, 211, 211)
TOWER_LIGHT = (226, 226, 226)
COLORS = {
    "PPO-PID": (44, 160, 44),
    "A*-PID": (214, 39, 40),
    "RRT*-PID": (23, 190, 207),
}
LINE_PATTERNS = {
    "PPO-PID": None,
    "A*-PID": (18.0, 9.0),
    "RRT*-PID": (20.0, 7.0, 4.0, 7.0),
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf"]
        if bold
        else ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]
    )
    names.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


class PlotCanvas:
    def __init__(
        self,
        box: tuple[int, int, int, int],
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ):
        self.left, self.top, self.right, self.bottom = box
        self.xmin, self.xmax = xlim
        self.ymin, self.ymax = ylim

    def point(self, x: float, y: float) -> tuple[float, float]:
        px = self.left + (x - self.xmin) / (self.xmax - self.xmin) * (
            self.right - self.left
        )
        py = self.bottom - (y - self.ymin) / (self.ymax - self.ymin) * (
            self.bottom - self.top
        )
        return float(px), float(py)

    def points(self, values: np.ndarray) -> list[tuple[float, float]]:
        return [self.point(float(x), float(y)) for x, y in np.asarray(values)]


def _nice_ticks(low: float, high: float, count: int = 5) -> np.ndarray:
    if high <= low:
        return np.array([low], dtype=float)
    raw = (high - low) / max(1, count)
    magnitude = 10.0 ** np.floor(np.log10(raw))
    normalized = raw / magnitude
    step = (1.0 if normalized <= 1.0 else 2.0 if normalized <= 2.0 else 5.0) * magnitude
    start = np.ceil(low / step) * step
    return np.arange(start, high + step * 0.25, step)


def _axes(
    draw: ImageDraw.ImageDraw,
    canvas: PlotCanvas,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    xticks: np.ndarray | None = None,
    yticks: np.ndarray | None = None,
    title_size: int = 25,
) -> None:
    box = (canvas.left, canvas.top, canvas.right, canvas.bottom)
    draw.rectangle(box, outline=INK, width=2)
    xticks = _nice_ticks(canvas.xmin, canvas.xmax) if xticks is None else xticks
    yticks = _nice_ticks(canvas.ymin, canvas.ymax) if yticks is None else yticks
    tick_font = _font(18)
    for value in xticks:
        x, y = canvas.point(float(value), canvas.ymin)
        draw.line((x, y, x, y + 7), fill=INK, width=2)
        label = f"{value:g}"
        draw.text((x, y + 10), label, fill=INK, font=tick_font, anchor="ma")
    for value in yticks:
        x, y = canvas.point(canvas.xmin, float(value))
        draw.line((x - 7, y, x, y), fill=INK, width=2)
        label = f"{value:g}"
        draw.text((x - 12, y), label, fill=INK, font=tick_font, anchor="rm")
    draw.text(
        ((canvas.left + canvas.right) / 2, canvas.top - 14),
        title,
        fill=INK,
        font=_font(title_size),
        anchor="ms",
    )
    draw.text(
        ((canvas.left + canvas.right) / 2, canvas.bottom + 47),
        xlabel,
        fill=INK,
        font=_font(20),
        anchor="ma",
    )
    label = Image.new("RGBA", (220, 44), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((110, 22), ylabel, fill=INK, font=_font(20), anchor="mm")
    label = label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    draw._image.paste(
        label,
        (canvas.left - 67, int((canvas.top + canvas.bottom - label.height) / 2)),
        label,
    )


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    fill: tuple[int, int, int],
    width: int,
    pattern: tuple[float, ...] | None = None,
) -> None:
    if len(points) < 2:
        return
    if pattern is None:
        draw.line(points, fill=fill, width=width, joint="curve")
        return

    pattern_index = 0
    pattern_used = 0.0
    drawing = True
    for start, end in zip(points[:-1], points[1:]):
        x0, y0 = start
        x1, y1 = end
        length = float(np.hypot(x1 - x0, y1 - y0))
        if length < 1e-9:
            continue
        used = 0.0
        while used < length - 1e-9:
            remaining = pattern[pattern_index] - pattern_used
            advance = min(remaining, length - used)
            t0, t1 = used / length, (used + advance) / length
            if drawing:
                draw.line(
                    (
                        x0 + (x1 - x0) * t0,
                        y0 + (y1 - y0) * t0,
                        x0 + (x1 - x0) * t1,
                        y0 + (y1 - y0) * t1,
                    ),
                    fill=fill,
                    width=width,
                )
            used += advance
            pattern_used += advance
            if pattern_used >= pattern[pattern_index] - 1e-9:
                pattern_index = (pattern_index + 1) % len(pattern)
                pattern_used = 0.0
                drawing = not drawing


def _star(center: tuple[float, float], outer: float, inner: float) -> list[tuple[float, float]]:
    cx, cy = center
    points = []
    for index in range(10):
        radius = outer if index % 2 == 0 else inner
        angle = -np.pi / 2 + index * np.pi / 5
        points.append((cx + radius * np.cos(angle), cy + radius * np.sin(angle)))
    return points


def _tower_segments(wires: np.ndarray) -> tuple[list[tuple[np.ndarray, np.ndarray, bool]], list[np.ndarray]]:
    wires = np.asarray(wires, dtype=np.float64)
    z_bottom = float(np.mean(np.sort(wires[:, 1])[:2]))
    z_top = float(np.mean(np.sort(wires[:, 1])[-2:]))
    p1 = np.array([21.0, 0.0])
    p2 = np.array([25.0, 6.0])
    p3 = np.array([29.0, 0.0])
    p4 = np.array([23.5, z_bottom - 3.5])
    p5 = np.array([26.5, z_bottom - 3.5])
    p6 = np.array([25.9, z_top])
    p7 = np.array([24.1, z_top])

    segments: list[tuple[np.ndarray, np.ndarray, bool]] = []

    def polyline(points: list[np.ndarray], main: bool = True) -> None:
        for start, end in zip(points[:-1], points[1:]):
            segments.append((start, end, main))

    polyline([p1, p2, p3, p5, p4, p1])
    polyline([p4, p5, p6, p7, p4])

    def x_at(a: np.ndarray, b: np.ndarray, z: float) -> float:
        if abs(float(b[1] - a[1])) < 1e-9:
            return float(a[0])
        ratio = (z - float(a[1])) / float(b[1] - a[1])
        return float(a[0] + ratio * (b[0] - a[0]))

    def lattice(
        left_a: np.ndarray,
        left_b: np.ndarray,
        right_a: np.ndarray,
        right_b: np.ndarray,
        z_start: float,
        z_end: float,
        count: int,
    ) -> None:
        if z_end <= z_start:
            return
        levels = np.linspace(z_start, z_end, max(3, count))
        for z0, z1 in zip(levels[:-1], levels[1:]):
            left0 = np.array([x_at(left_a, left_b, float(z0)), z0])
            right0 = np.array([x_at(right_a, right_b, float(z0)), z0])
            left1 = np.array([x_at(left_a, left_b, float(z1)), z1])
            right1 = np.array([x_at(right_a, right_b, float(z1)), z1])
            segments.extend(
                [(left0, right0, False), (left0, right1, False), (right0, left1, False)]
            )

    lattice(p1, p4, p3, p5, 2.0, float(p4[1] - 1.0), 9)
    lattice(p4, p7, p5, p6, float(p4[1] + 1.0), float(p7[1] - 1.0), 6)

    def cross2(a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    edges = [(p4, p5), (p5, p6), (p6, p7), (p7, p4)]

    def intersection(origin: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
        result, best_t = None, float("inf")
        for start, end in edges:
            edge = end - start
            denominator = cross2(direction, edge)
            if abs(denominator) < 1e-10:
                continue
            delta = start - origin
            t = cross2(delta, edge) / denominator
            u = cross2(delta, direction) / denominator
            if t >= 0.0 and 0.0 <= u <= 1.0 and t < best_t:
                result, best_t = origin + t * direction, t
        return result

    def nearest(x: float, z: float) -> np.ndarray:
        return wires[int(np.argmin((wires[:, 0] - x) ** 2 + (wires[:, 1] - z) ** 2))]

    angle = np.radians(20.0)
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    arms = [
        (nearest(20.0, z_bottom), np.array([1.0, 0.0]), p4, False),
        (nearest(20.0, z_bottom), np.array([cosine, sine]), p4, False),
        (nearest(30.0, z_bottom), np.array([-1.0, 0.0]), p5, False),
        (nearest(30.0, z_bottom), np.array([-cosine, sine]), p5, False),
        (nearest(22.0, z_top), np.array([cosine, -sine]), p7, True),
        (nearest(28.0, z_top), np.array([-cosine, -sine]), p6, True),
    ]
    for wire, direction, anchor, triangular in arms:
        hit = intersection(wire, direction)
        if hit is None:
            continue
        segments.append((wire, hit, True))
        if triangular:
            segments.extend([(anchor, hit, False), (wire, anchor, False)])
        segments.append((wire + np.array([0.0, 1.2]), wire, False))
    return segments, [p1, p2, p3, p4, p5, p6, p7]


def _draw_tower(draw: ImageDraw.ImageDraw, canvas: PlotCanvas, wires: np.ndarray) -> None:
    segments, _ = _tower_segments(wires)
    for start, end, main in segments:
        draw.line(
            (*canvas.point(*start), *canvas.point(*end)),
            fill=TOWER if main else TOWER_LIGHT,
            width=3 if main else 2,
        )


def _draw_world(
    draw: ImageDraw.ImageDraw,
    canvas: PlotCanvas,
    scenario: Scenario,
    *,
    title: str,
) -> tuple[np.ndarray, np.ndarray]:
    _axes(
        draw,
        canvas,
        title=title,
        xlabel="X (m)",
        ylabel="Z (m)",
        xticks=np.arange(0.0, 51.0, 10.0),
        yticks=np.arange(0.0, 61.0, 10.0),
    )
    wires = CANONICAL_WIRES.copy()
    _draw_tower(draw, canvas, wires)
    for index, wire in enumerate(wires):
        point = canvas.point(*wire)
        draw.ellipse(
            (point[0] - 7, point[1] - 7, point[0] + 7, point[1] + 7), fill=(0, 0, 0)
        )
        draw.text(
            (point[0] + 10, point[1] - 3),
            f"W{index}",
            fill=INK,
            font=_font(17),
            anchor="lm",
        )
    goal = wires[scenario.wire_index] + np.array([0.0, 3.48])
    ground = np.array([scenario.ground_x, scenario.ground_z], dtype=np.float64)
    start, target = (ground, goal) if scenario.mode == "ground_to_wire" else (goal, ground)
    start_point, target_point = canvas.point(*start), canvas.point(*target)
    draw.ellipse(
        (
            start_point[0] - 10,
            start_point[1] - 10,
            start_point[0] + 10,
            start_point[1] + 10,
        ),
        fill=(0, 238, 0),
    )
    draw.polygon(_star(target_point, 13, 6), fill=(153, 0, 153))
    return start, target


def _legend(draw: ImageDraw.ImageDraw, origin: tuple[int, int]) -> None:
    left, top = origin
    width, height = 270, 202
    draw.rounded_rectangle(
        (left, top, left + width, top + height),
        radius=8,
        fill=(255, 255, 255),
        outline=(200, 200, 200),
        width=2,
    )
    font = _font(18)
    rows = [
        ("Start", "start"),
        ("Goal", "goal"),
        ("A* + smooth + PID", "A*-PID"),
        ("RRT* + smooth + PID", "RRT*-PID"),
        ("PPO + PID", "PPO-PID"),
    ]
    for index, (label, kind) in enumerate(rows):
        y = top + 23 + index * 37
        if kind == "start":
            draw.ellipse((left + 17, y - 8, left + 33, y + 8), fill=(0, 238, 0))
        elif kind == "goal":
            draw.polygon(_star((left + 25, y), 10, 4.5), fill=(153, 0, 153))
        else:
            _draw_polyline(
                draw,
                [(left + 10, y), (left + 43, y)],
                fill=COLORS[kind],
                width=5,
                pattern=LINE_PATTERNS[kind],
            )
        draw.text((left + 53, y), label, fill=INK, font=font, anchor="lm")


def _series_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    logs: list[FlightLog],
    *,
    title: str,
    ylabel: str,
    extractor,
    symmetric: bool = False,
) -> None:
    values = [np.asarray(extractor(log), dtype=float) for log in logs]
    maximum_time = max(float(log.times[-1]) for log in logs)
    low = min(float(np.min(value)) for value in values)
    high = max(float(np.max(value)) for value in values)
    if symmetric:
        bound = max(abs(low), abs(high), 1.0) * 1.08
        low, high = -bound, bound
    else:
        low = min(0.0, low)
        high = max(high * 1.08, 1.0)
    canvas = PlotCanvas(box, (0.0, maximum_time), (low, high))
    _axes(draw, canvas, title=title, xlabel="t (s)", ylabel=ylabel, title_size=24)
    if low < 0.0 < high:
        y = canvas.point(0.0, 0.0)[1]
        draw.line((canvas.left, y, canvas.right, y), fill=(225, 225, 225), width=1)
    for log, value in zip(logs, values):
        series = np.column_stack((log.times, value))
        _draw_polyline(
            draw,
            canvas.points(series),
            fill=COLORS.get(log.label, (60, 60, 60)),
            width=4 if log.label == "PPO-PID" else 3,
            pattern=LINE_PATTERNS.get(log.label),
        )


def save_comparison(
    logs: list[FlightLog], scenario: Scenario, output_path: Path
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (2048, 1152), BACKGROUND)
    draw = ImageDraw.Draw(image)
    mode = "Up-line" if scenario.mode == "ground_to_wire" else "Down-line"
    wire = CANONICAL_WIRES[scenario.wire_index]
    goal = wire + np.array([0.0, 3.48])
    ground = np.array([scenario.ground_x, scenario.ground_z])
    start, target = (ground, goal) if scenario.mode == "ground_to_wire" else (goal, ground)
    heading = (
        f"{mode} | start=({start[0]:.1f},{start[1]:.1f}) -> "
        f"{'W' + str(scenario.wire_index) + ' goal' if scenario.mode == 'ground_to_wire' else 'ground'}"
    )
    draw.text((1024, 5), heading, fill=INK, font=_font(27), anchor="ma")

    world = PlotCanvas((65, 83, 1367, 1082), (-5.0, 55.0), (0.0, 60.0))
    _draw_world(draw, world, scenario, title="Trajectory (X-Z)")
    for log in logs:
        _draw_polyline(
            draw,
            world.points(log.states[:, :2]),
            fill=COLORS.get(log.label, (60, 60, 60)),
            width=6 if log.label == "PPO-PID" else 4,
            pattern=LINE_PATTERNS.get(log.label),
        )
    _legend(draw, (1080, 96))

    _series_panel(
        draw,
        (1462, 83, 2010, 526),
        logs,
        title="Swing Angle (deg)",
        ylabel="deg",
        extractor=lambda log: np.degrees(log.states[:, 3]),
        symmetric=True,
    )
    _series_panel(
        draw,
        (1462, 633, 2010, 1082),
        logs,
        title="Speed Magnitude (m/s)",
        ylabel="m/s",
        extractor=lambda log: np.linalg.norm(log.states[:, 4:6], axis=1),
    )
    image.save(output_path, optimize=True)
    return output_path


def _draw_drone(
    draw: ImageDraw.ImageDraw,
    canvas: PlotCanvas,
    state: np.ndarray,
    config: DroneConfig,
) -> None:
    x, z = float(state[0]), float(state[1])
    theta, alpha = -float(state[2]), float(state[3])
    center = np.array([x, z])
    along = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([-np.sin(theta), np.cos(theta)])
    body_left = center - along
    body_right = center + along
    connector_length = 0.165
    propeller_length = 1.375
    connector_left = body_left + connector_length * normal
    connector_right = body_right + connector_length * normal
    propeller_half = 0.5 * propeller_length * along

    load_center = center + config.effective_rope_length * np.array(
        [np.sin(alpha), -np.cos(alpha)]
    )
    cable_end = load_center - 0.5 * config.payload_height * np.array(
        [np.sin(alpha), -np.cos(alpha)]
    )
    long_half = 0.5 * config.payload_height * np.array(
        [np.sin(alpha), -np.cos(alpha)]
    )
    short_half = 0.5 * config.payload_width * np.array(
        [np.cos(alpha), np.sin(alpha)]
    )
    corners = [
        load_center - long_half - short_half,
        load_center - long_half + short_half,
        load_center + long_half + short_half,
        load_center + long_half - short_half,
    ]

    draw.line((*canvas.point(*center), *canvas.point(*cable_end)), fill=(44, 160, 44), width=4)
    draw.polygon([canvas.point(*corner) for corner in corners], fill=(176, 176, 176), outline=INK)
    draw.line((*canvas.point(*body_left), *canvas.point(*body_right)), fill=(31, 119, 180), width=6)
    for endpoint, connector in ((body_left, connector_left), (body_right, connector_right)):
        draw.line((*canvas.point(*endpoint), *canvas.point(*connector)), fill=(0, 0, 0), width=3)
        draw.line(
            (
                *canvas.point(*(connector - propeller_half)),
                *canvas.point(*(connector + propeller_half)),
            ),
            fill=(214, 39, 40),
            width=5,
        )


def save_gif(
    log: FlightLog,
    scenario: Scenario,
    output_path: Path,
    maximum_frames: int = 140,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = np.unique(
        np.linspace(0, len(log.states) - 1, min(maximum_frames, len(log.states))).astype(int)
    )
    config = DroneConfig()
    frames: list[Image.Image] = []
    mode = "Up-line" if scenario.mode == "ground_to_wire" else "Down-line"
    for frame_index in frame_indices:
        image = Image.new("RGB", (900, 900), BACKGROUND)
        draw = ImageDraw.Draw(image)
        world = PlotCanvas((70, 61, 840, 831), (-5.0, 55.0), (0.0, 60.0))
        _draw_world(
            draw,
            world,
            scenario,
            title=f"PPO-PID {mode} Navigation | t={log.times[frame_index]:.1f} s",
        )
        trail = log.states[: frame_index + 1, :2]
        _draw_polyline(
            draw,
            world.points(trail),
            fill=(31, 119, 180),
            width=4,
        )
        _draw_drone(draw, world, log.states[frame_index], config)
        frames.append(image.quantize(colors=128, method=Image.Quantize.MEDIANCUT))

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
        disposal=2,
        optimize=False,
    )
    return output_path
