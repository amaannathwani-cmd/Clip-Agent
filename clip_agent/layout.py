"""Work out where the streamer's face is and choose a vertical layout.

Layouts:
  split  - gameplay on top, the streamer's facecam on the bottom (most gaming streams)
  face   - a 9:16 crop that follows the streamer (just-chatting / IRL, where the face is big)
  blur   - the whole frame on a blurred background (no face found)
"""
from dataclasses import dataclass
from pathlib import Path

W, H = 1080, 1920
TOP_H = 1100           # gameplay panel height in the split layout
BOTTOM_H = H - TOP_H   # facecam panel height


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int


@dataclass
class Layout:
    kind: str
    src_w: int
    src_h: int
    cam: Box | None = None    # facecam crop in source pixels (split)
    game: Box | None = None   # gameplay crop in source pixels (split)
    crop: Box | None = None   # 9:16 crop (face)

    @property
    def caption_y(self) -> int:
        return {"split": TOP_H, "face": 1450}.get(self.kind, 1390)


def _even(v: float) -> int:
    return max(2, int(v) // 2 * 2)


def _fit_box(cx: float, cy: float, w: float, h: float, src_w: int, src_h: int) -> Box:
    """A w x h box centred on (cx, cy), shrunk to fit and shifted to stay inside the frame."""
    scale = min(1.0, src_w / w, src_h / h)
    w, h = w * scale, h * scale
    x = min(max(cx - w / 2, 0), src_w - w)
    y = min(max(cy - h / 2, 0), src_h - h)
    return Box(_even(x), _even(y), _even(w), _even(h))


def detect_faces(path: Path, samples: int = 16) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    """Return (width, height, face boxes) from evenly spaced frames, keeping the largest face per frame."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = []
    for i in range(samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / samples))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        found = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                         minSize=(max(24, src_h // 30), max(24, src_h // 30)))
        if len(found):
            faces.append(tuple(int(v) for v in max(found, key=lambda f: f[2] * f[3])))
    cap.release()
    return src_w, src_h, faces


def _median(vals):
    vals = sorted(vals)
    return vals[len(vals) // 2]


def choose_layout(src_w: int, src_h: int, faces: list, samples: int = 16) -> Layout:
    if src_w <= 0 or src_h <= 0:
        return Layout("blur", src_w, src_h)
    # Need a face in enough frames, and those detections must agree on where it is (a facecam doesn't move).
    if len(faces) < max(3, samples * 0.3):
        return Layout("blur", src_w, src_h)
    cx = _median([x + w / 2 for x, y, w, h in faces])
    cy = _median([y + h / 2 for x, y, w, h in faces])
    fw = _median([w for x, y, w, h in faces])
    steady = [f for f in faces if abs(f[0] + f[2] / 2 - cx) < fw and abs(f[1] + f[3] / 2 - cy) < fw]
    if len(steady) < len(faces) * 0.6:
        return Layout("blur", src_w, src_h)

    if fw > src_w * 0.12:
        # Big face: the stream is mostly the streamer. Follow them with a 9:16 crop.
        crop_w = src_h * W / H
        return Layout("face", src_w, src_h, crop=_fit_box(cx, src_h / 2, crop_w, src_h, src_w, src_h))

    # Small face in a corner: it's a facecam overlay. Frame it for the bottom panel.
    cam_h = fw * 2.8
    cam = _fit_box(cx, cy + fw * 0.15, cam_h * W / BOTTOM_H, cam_h, src_w, src_h)
    game_w = src_h * W / TOP_H
    game = _fit_box(src_w / 2, src_h / 2, game_w, src_h, src_w, src_h)
    return Layout("split", src_w, src_h, cam=cam, game=game)


def plan(path: Path) -> Layout:
    try:
        src_w, src_h, faces = detect_faces(path)
        return choose_layout(src_w, src_h, faces)
    except Exception as e:
        print(f"  face detection failed ({e}); using blur layout")
        return Layout("blur", 0, 0)


def video_filter(layout: Layout) -> str:
    """ffmpeg filter_complex producing [v] at 1080x1920 with subs.ass burned in."""
    if layout.kind == "split":
        g, c = layout.game, layout.cam
        return (
            "[0:v]split=2[a][b];"
            f"[a]crop={g.w}:{g.h}:{g.x}:{g.y},scale={W}:{TOP_H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{TOP_H}[top];"
            f"[b]crop={c.w}:{c.h}:{c.x}:{c.y},scale={W}:{BOTTOM_H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{BOTTOM_H}[bot];"
            "[top][bot]vstack=inputs=2,setsar=1,ass=subs.ass[v]"
        )
    if layout.kind == "face":
        c = layout.crop
        return (f"[0:v]crop={c.w}:{c.h}:{c.x}:{c.y},scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H},setsar=1,ass=subs.ass[v]")
    return (
        "[0:v]split=2[a][b];"
        f"[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur=25:3,eq=brightness=-0.12[bg];"
        f"[b]crop='min(iw,ih*4/3)':ih,scale={W}:-2[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2-160,setsar=1,ass=subs.ass[v]"
    )
