"""Always-on movie recorder for the MuJoCo execution stage.

One helper consolidates the frame-capture pattern repeated across the mujoco
examples (mujoco_rideunder.build_movie, _encode_frames; mujoco_falsework.render):
render offscreen while stepping, accumulate frames, then write a GIF (PIL) and
an MP4 (ffmpeg) plus a final still. Recording is on by default. A caller opts
out with record=False, which the tests use and which makes every method a
no-op with no mujoco or PIL import.

The recorder can be re-pointed at a new model between build steps with bind, so
one recorder spans the per-step models of a multi-block build and all frames
land in a single movie. mujoco and PIL are imported lazily inside the methods,
so importing this module never needs either.
"""

import os
import shutil
import subprocess

# The homebrew ffmpeg path on this machine, tried when ffmpeg is not on PATH.
FFMPEG_FALLBACK = "/opt/homebrew/bin/ffmpeg"

# Unit-scale side view of the xz plane. The build lives around z = 2 with a
# bounding-box diagonal near 12, so this frames the pedestal and the stack.
DEFAULT_CAMERA = {
    "lookat": [0.0, 0.0, 2.0],
    "distance": 12.0,
    "azimuth": 90.0,
    "elevation": -10.0,
}


def find_ffmpeg():
    """Path to an ffmpeg binary, or None. PATH first, then the homebrew path."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    if os.path.exists(FFMPEG_FALLBACK):
        return FFMPEG_FALLBACK
    return None


class FrameRecorder:
    """Offscreen frame capture over one or more MuJoCo models.

    Construct with a model and data to bind immediately, then call capture once
    per simulation step; a frame is rendered every `stride` steps. finalize
    writes the movie files. With record=False nothing is imported or rendered
    and finalize reports the skip.
    """

    def __init__(
        self,
        model=None,
        data=None,
        *,
        height=720,
        width=1280,
        stride=10,
        camera_overrides=None,
        record=True,
        fps=25,
        brighten=1.0,
    ):
        self.height = int(height)
        self.width = int(width)
        self.stride = max(1, int(stride))
        self.camera_overrides = dict(camera_overrides or {})
        self.record = bool(record)
        self.fps = int(fps)
        self.brighten = float(brighten)
        self.frames = []
        self._renderer = None
        self._model = None
        self._data = None
        self._cam = None
        self._count = 0
        if self.record and model is not None:
            self.bind(model, data)

    def bind(self, model, data):
        """Point the recorder at a model and data, rebuilding the renderer.

        Called on construction and whenever a new per-step model appears. A no
        op when recording is off.
        """
        if not self.record:
            return
        import mujoco

        if self._renderer is not None and model is not self._model:
            self._renderer.close()
            self._renderer = None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                model, height=self.height, width=self.width
            )
        self._model = model
        self._data = data
        self._cam = self._make_camera(mujoco, model)

    def _make_camera(self, mujoco, model):
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, cam)
        cam.lookat[:] = self.camera_overrides.get("lookat", DEFAULT_CAMERA["lookat"])
        cam.distance = float(
            self.camera_overrides.get("distance", DEFAULT_CAMERA["distance"])
        )
        cam.azimuth = float(
            self.camera_overrides.get("azimuth", DEFAULT_CAMERA["azimuth"])
        )
        cam.elevation = float(
            self.camera_overrides.get("elevation", DEFAULT_CAMERA["elevation"])
        )
        return cam

    def capture(self, model=None, data=None, force=False):
        """Render one frame if recording and the stride is due.

        Pass model and data to follow a rebuilt per-step model; the recorder
        rebinds when the model identity changes. Omit them to reuse the bound
        pair. force renders regardless of the stride (used for a final frame).
        """
        if not self.record:
            return
        if model is not None and (self._renderer is None or model is not self._model):
            self.bind(model, data)
        elif data is not None:
            self._data = data
        self._count += 1
        if not (force or self._count % self.stride == 0):
            return
        self._renderer.update_scene(self._data, camera=self._cam)
        img = self._renderer.render().copy()
        if self.brighten != 1.0:
            import numpy as np

            img = np.clip(
                img.astype("float32") * self.brighten, 0.0, 255.0
            ).astype("uint8")
        self.frames.append(img)

    def finalize(self, path_base, *, gif=True, still=True):
        """Write the movie files and return a dict of paths or skip reasons.

        Writes {path_base}.mp4 (ffmpeg), {path_base}.gif (PIL), and
        {path_base}_final.png (last frame). Missing ffmpeg or PIL degrades to a
        recorded skip reason, never an exception.
        """
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        out = {
            "mp4": None,
            "gif": None,
            "still": None,
            "n_frames": len(self.frames),
        }
        if not self.record:
            out["skipped"] = "record=False"
            return out
        if not self.frames:
            out["skipped"] = "no frames captured"
            return out
        try:
            from PIL import Image
        except Exception as exc:  # noqa: BLE001
            out["skipped"] = f"PIL missing ({type(exc).__name__})"
            return out

        os.makedirs(os.path.dirname(path_base) or ".", exist_ok=True)
        imgs = [Image.fromarray(f) for f in self.frames]

        if still:
            still_path = path_base + "_final.png"
            imgs[-1].save(still_path)
            out["still"] = still_path
        if gif:
            gif_path = path_base + ".gif"
            imgs[0].save(
                gif_path,
                save_all=True,
                append_images=imgs[1:],
                duration=int(1000 / self.fps),
                loop=0,
            )
            out["gif"] = gif_path

        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            out["mp4_skip"] = "ffmpeg not found"
            return out
        tmp = path_base + "_frames"
        os.makedirs(tmp, exist_ok=True)
        try:
            for k, im in enumerate(imgs):
                im.save(os.path.join(tmp, f"f{k:05d}.png"))
            mp4 = path_base + ".mp4"
            proc = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-framerate",
                    str(self.fps),
                    "-i",
                    os.path.join(tmp, "f%05d.png"),
                    "-pix_fmt",
                    "yuv420p",
                    mp4,
                ],
                capture_output=True,
            )
            if proc.returncode == 0:
                out["mp4"] = mp4
            else:
                out["mp4_skip"] = f"ffmpeg rc={proc.returncode}"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return out
