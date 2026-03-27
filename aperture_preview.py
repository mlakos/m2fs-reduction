"""
aperture_preview.py
====================
Interactive aperture-trace viewer to run BEFORE echelle apall extraction.

After the notebook has assembled the mosaics, call this script to inspect
and optionally correct the star→aperture mapping for each science frame.

Usage
-----
    python aperture_preview.py <stacked_science_frame.fits> [options]

Options
-------
    --nap   INT     Total number of apertures to display (default: auto-detect
                    from the image height using ~5-px separation guess).
    --sep   FLOAT   Expected spatial separation between consecutive apertures
                    in pixels (default: 5.0).  Used only when --nap is not
                    given.
    --col   INT     Column (dispersion pixel) used for the spatial cut shown
                    in the preview (default: middle of the image).
    --nstars INT    Number of unique stars in the pattern (default: 4).
    --out   PATH    Write the final aperture→star mapping to this file
                    (default: <image_base>_star_map.txt).

Fiber / aperture pattern
------------------------
Default auto-assignment is sequential groups of four apertures from top to
bottom:

    1 1 1 1  2 2 2 2  3 3 3 3  4 4 4 4 ...

Manual edits can override this pattern; edited apertures are treated as locks
and are preserved during forward reflow.

Interactive controls (when the figure is focused)
-------------------------------------------------
        e   — hover the mouse over any trace and press 'e' to reassign that
           aperture's star.  A text prompt appears in the terminal.
        f   — after an 'e' edit, press 'f' to recompute affiliations forward
            from that aperture while preserving all manual edits.
        q/g — first press arms quit confirmation; press q or g again to accept
            the current mapping and close the window.
    r   — reset all assignments to the auto-generated pattern.
    h   — print a short help reminder to the terminal.
"""

import sys
import os
import argparse
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # needs a display; change to Qt5Agg if preferred
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Cursor
from astropy.io import fits


# ---------------------------------------------------------------------------
# Pattern generator
# ---------------------------------------------------------------------------

def default_pattern(n_apertures, n_stars=4):
    """
    Return an array of length *n_apertures* with 1-based star indices,
    assigned as sequential groups of four apertures from top to bottom.

    Example:
        1 1 1 1  2 2 2 2  3 3 3 3 ...
    """
    return (np.arange(n_apertures, dtype=int) // 4) + 1


def recompute_forward_with_locks(pattern, manual_locks, anchor_idx):
    """Recompute affiliations forward from *anchor_idx* preserving locks.

    Rules:
      - All manual-locked apertures remain unchanged.
      - Recompute only apertures after *anchor_idx*.
      - Unlocked apertures follow sequential 4-per-star groups from the most
        recent anchor (explicit manual lock or the initial anchor aperture).
    """
    pat = np.array(pattern, dtype=int).copy()
    locks = np.array(manual_locks, dtype=bool)
    n = len(pat)
    if n == 0 or anchor_idx is None:
        return pat

    anchor_idx = int(anchor_idx)
    if anchor_idx < 0 or anchor_idx >= n:
        return pat

    current_anchor_idx = anchor_idx
    current_anchor_star = int(pat[anchor_idx])

    for i in range(anchor_idx + 1, n):
        if locks[i]:
            current_anchor_idx = i
            current_anchor_star = int(pat[i])
            continue

        delta = i - current_anchor_idx
        pat[i] = current_anchor_star + (delta // 4)

    return pat


# ---------------------------------------------------------------------------
# Aperture detection from a spatial cut
# ---------------------------------------------------------------------------

def find_aperture_centers(image_data, col=None, expected_sep=5.0):
    """
    Simple peak-finder on a spatial cut at column *col*.
    Returns sorted array of row-pixel positions.
    """
    from scipy.signal import find_peaks

    if col is None:
        col = image_data.shape[1] // 2

    cut = image_data[:, col].astype(float)
    # smooth slightly
    kernel = np.ones(3) / 3.0
    cut_s  = np.convolve(cut, kernel, mode='same')

    min_sep = max(2, int(expected_sep * 0.5))
    peaks, props = find_peaks(cut_s, distance=min_sep,
                               height=np.percentile(cut_s, 30))
    return np.sort(peaks)


def _db_stem_variants(image_path):
    """Return likely IRAF database stem variants for an image path."""
    stem = Path(image_path).stem
    variants = [stem]
    for suffix in ("-sl_nflat", "-sl_med", "-sl", "-F", "_nflat"):
        if stem.endswith(suffix):
            variants.append(stem[: -len(suffix)])
    # Preserve order while removing duplicates.
    out = []
    seen = set()
    for item in variants:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def find_iraf_aperture_db(image_path):
    """Locate an IRAF aperture DB file corresponding to image_path."""
    db_dir = Path("database")
    if not db_dir.exists():
        return None

    candidates = []
    for stem in _db_stem_variants(image_path):
        candidates.append(db_dir / f"ap.{stem}")
        candidates.append(db_dir / f"ap._{stem}")
        candidates.append(db_dir / f"ap{stem}")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _evaluate_trace_curve(curve_values, ncols):
    """Evaluate IRAF curve metadata onto detector x-pixel coordinates."""
    if len(curve_values) < 4:
        return None

    fit_type = int(round(curve_values[0]))
    ncoeff = int(round(curve_values[1]))
    xmin = float(curve_values[2])
    xmax = float(curve_values[3])
    if len(curve_values) < 4 + ncoeff:
        return None
    coeffs = np.array(curve_values[4: 4 + ncoeff], dtype=float)

    if len(coeffs) == 0 or xmax == xmin:
        return None

    lo = max(0.0, min(xmin, xmax))
    hi = min(float(ncols - 1), max(xmin, xmax))
    if hi - lo < 1.0:
        return None

    # Respect the IRAF fit domain so plotted trace length matches DB content.
    x_pixels = np.linspace(lo, hi, 220)

    # IRAF stores curve fits on a normalized domain in [-1, 1].
    xnorm = 2.0 * (x_pixels - xmin) / (xmax - xmin) - 1.0

    if fit_type == 2:  # chebyshev
        y = np.polynomial.chebyshev.chebval(xnorm, coeffs)
        return x_pixels, y
    if fit_type == 1:  # legendre
        y = np.polynomial.legendre.legval(xnorm, coeffs)
        return x_pixels, y
    return None


def load_iraf_traces(image_path, ncols):
    """Load aperture centres and trace models from IRAF DB, if available.

    Returns a dict with keys:
        source      - "database" or "peaks"
        db_path     - path used (or None)
        parsed      - number of valid aperture blocks parsed
        skipped     - number of skipped/malformed blocks
        centers     - np.ndarray of 1-based aperture centres in DB order
        trace_rows  - dict[int, tuple[np.ndarray, np.ndarray]] mapping
                  0-based aperture index to (x_pixels, y(x))
    """
    result = {
        "source": "peaks",
        "db_path": None,
        "parsed": 0,
        "skipped": 0,
        "centers": None,
        "trace_rows": {},
    }

    db_path = find_iraf_aperture_db(image_path)
    if not db_path:
        return result

    result["db_path"] = db_path
    with open(db_path, "r") as f:
        lines = f.readlines()

    entries = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("begin"):
            i += 1
            continue

        parts = line.split()
        if len(parts) < 4 or parts[1] != "aperture":
            i += 1
            continue

        try:
            ap_num = int(parts[3])
        except ValueError:
            result["skipped"] += 1
            i += 1
            continue

        j = i + 1
        center_y = None
        curve_values = None
        while j < len(lines):
            inner = lines[j].strip()
            if inner.startswith("begin"):
                break

            if inner.startswith("center"):
                toks = inner.split()
                if len(toks) >= 3:
                    try:
                        center_y = float(toks[2])
                    except ValueError:
                        center_y = None

            if inner.startswith("curve"):
                toks = inner.split()
                try:
                    nvals = int(toks[1])
                except (IndexError, ValueError):
                    nvals = 0

                vals = []
                k = j + 1
                while k < len(lines) and len(vals) < nvals:
                    probe = lines[k].strip()
                    if not probe:
                        k += 1
                        continue
                    if probe.startswith("begin"):
                        break
                    try:
                        vals.append(float(probe))
                    except ValueError:
                        pass
                    k += 1
                curve_values = vals if len(vals) == nvals else None

            j += 1

        if center_y is None or curve_values is None:
            result["skipped"] += 1
            i = j
            continue

        curve_eval = _evaluate_trace_curve(curve_values, ncols)
        if curve_eval is None:
            result["skipped"] += 1
            i = j
            continue
        x_pixels, delta = curve_eval

        entries.append((ap_num, center_y, x_pixels, center_y + delta))
        result["parsed"] += 1
        i = j

    if not entries:
        return result

    entries.sort(key=lambda t: t[0])
    centers = np.array([e[1] for e in entries], dtype=float)
    trace_rows = {idx: (e[2], e[3]) for idx, e in enumerate(entries)}

    result["source"] = "database"
    result["centers"] = centers
    result["trace_rows"] = trace_rows
    return result


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

STAR_COLORS = [
    "#e41a1c",   # red
    "#377eb8",   # blue
    "#4daf4a",   # green
    "#984ea3",   # purple
    "#ff7f00",   # orange
    "#a65628",   # brown
    "#f781bf",   # pink
    "#999999",   # grey
]

def star_color(star_idx_1based):
    return STAR_COLORS[(star_idx_1based - 1) % len(STAR_COLORS)]


# ---------------------------------------------------------------------------
# Interactive figure
# ---------------------------------------------------------------------------

class TraceViewer:
    """
    Single-panel aperture viewer with editable star assignments.

    Panel: aperture-only traces + labels + star-zone bands.

    Supports keys: 'e', 'f', 'q', 'g', 'r', 'h'.
    """

    def __init__(self, image_data, centers, pattern, image_name, col, n_stars,
                 trace_rows=None, save_prefix=None):
        self.data       = image_data
        self.centers    = np.array(centers, dtype=float)
        self.pattern    = np.array(pattern, dtype=int)
        self.image_name = image_name
        self.col        = col
        self.n_stars    = n_stars
        self.n_ap       = len(centers)
        self.trace_rows = trace_rows or {}
        self.save_prefix = str(save_prefix) if save_prefix is not None else os.path.splitext(image_name)[0]
        self.manual_lock = np.zeros(self.n_ap, dtype=bool)
        self.deleted     = np.zeros(self.n_ap, dtype=bool)  # Track deleted/unused apertures
        self.last_edit_idx = None
        self.done       = False
        self._hovered_ap = None
        self._quit_armed = False

        self._build_figure()

    # ------------------------------------------------------------------
    def _save_confirmation_figures(self, dpi=300):
        """Save confirmation snapshots for later reference.

        Exports two PNG files:
          1) Current interactive preview pane.
          2) FITS image with aperture overlays.
        """
        preview_png = f"{self.save_prefix}_aperture_preview.png"
        overlay_png = f"{self.save_prefix}_aperture_overlay.png"

        # 1) Save the currently displayed preview scene.
        self.fig.savefig(preview_png, dpi=dpi, bbox_inches="tight")

        # 2) Save FITS image with aperture overlays.
        nrows, ncols = self.data.shape
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        finite = np.isfinite(self.data)
        if np.any(finite):
            vmin, vmax = np.nanpercentile(self.data[finite], [5.0, 99.0])
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
                vmin, vmax = np.nanmin(self.data), np.nanmax(self.data)
        else:
            vmin, vmax = 0.0, 1.0

        ax.imshow(
            self.data,
            origin="lower",
            cmap="gray",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        for i, cen in enumerate(self.centers):
            is_deleted = self.deleted[i]
            c = "#cccccc" if is_deleted else star_color(int(self.pattern[i]))
            lw = 1.0 if is_deleted else 1.4
            alpha = 0.35 if is_deleted else 0.90
            ls = "--" if is_deleted else "-"

            if i in self.trace_rows:
                x_pixels, y_values = self.trace_rows[i]
                y_trace = np.clip(y_values, 0.0, nrows - 1)
                ax.plot(x_pixels, y_trace, color=c, lw=lw, alpha=alpha, linestyle=ls)
                label_x = float(np.clip(x_pixels[0], 0, ncols - 1))
                label_y = float(np.clip(y_trace[0], 0, nrows - 1))
            else:
                ax.plot([0, ncols - 1], [cen, cen], color=c, lw=lw, alpha=alpha, linestyle=ls)
                label_x = 0.0
                label_y = float(cen)

            prefix = "(X) " if is_deleted else ""
            ax.text(
                label_x,
                label_y + 1.0,
                f"{prefix}{i + 1}",
                color=c,
                fontsize=6,
                ha="left",
                va="bottom",
                style="italic" if is_deleted else "normal",
            )

        valid_stars = [s for s in sorted(np.unique(self.pattern)) if s != 0]
        patches = [
            mpatches.Patch(color=star_color(s), label=f"Star {s}")
            for s in valid_stars
        ]
        if np.any(self.deleted):
            patches.append(mpatches.Patch(color="#cccccc", label="Deleted"))
        if patches:
            ax.legend(handles=patches, loc="upper right", fontsize=8, framealpha=0.7)

        ax.set_xlim(0, ncols - 1)
        ax.set_ylim(0, nrows - 1)
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")
        ax.set_title(f"Aperture overlays: {self.image_name}")
        fig.tight_layout()
        fig.savefig(overlay_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        print(f"  Saved confirmation figure (300 dpi): {preview_png}")
        print(f"  Saved FITS+aperture overlay (300 dpi): {overlay_png}")

    # ------------------------------------------------------------------
    def _build_figure(self):
        self.fig, self.ax_cut = plt.subplots(1, 1, figsize=(16, 9))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)

        self._draw_scene()
        plt.tight_layout(pad=0.2)

    # ------------------------------------------------------------------
    def _draw_legend(self):
        n = max(self.pattern)
        patches = [
            mpatches.Patch(color=star_color(s), label=f"Star {s}")
            for s in range(1, n + 1)
        ]
        self.ax_cut.legend(handles=patches, loc="upper right", fontsize=6, framealpha=0.6)

    # ------------------------------------------------------------------
    def _draw_star_zones(self):
        for star in sorted(np.unique(self.pattern)):
            idx = np.where(self.pattern == star)[0]
            if len(idx) == 0:
                continue
            y0 = float(np.min(self.centers[idx]) - 1.5)
            y1 = float(np.max(self.centers[idx]) + 1.5)
            self.ax_cut.axhspan(y0, y1, color=star_color(star), alpha=0.16, zorder=0)

    # ------------------------------------------------------------------
    def _draw_scene(self):
        nrows, ncols = self.data.shape
        self.ax_cut.clear()

        # Aperture-only pane
        self.ax_cut.set_xlim(0.0, 1.0)
        self.ax_cut.set_ylim(0, nrows - 1)
        self.ax_cut.set_xticks([])

        self._draw_star_zones()
        have_curve_overlays = len(self.trace_rows) > 0

        for i, cen in enumerate(self.centers):
            is_deleted = self.deleted[i]
            c = "#cccccc" if is_deleted else star_color(int(self.pattern[i]))
            is_hovered = (self._hovered_ap == i)
            line_lw = 3 if is_hovered else 2
            edge = "#ffee58" if is_hovered else c
            line_alpha = 0.25 if is_deleted else 0.70

            # Overlay true IRAF trace fit where available.
            if i in self.trace_rows:
                x_pixels, y_values = self.trace_rows[i]
                y_trace = np.clip(y_values, 0.0, nrows - 1)
                denom = float(max(1, ncols - 1))
                x_trace = 0.15 + 0.70 * (x_pixels / denom)
                self.ax_cut.plot(
                    x_trace, y_trace,
                    color=edge,
                    lw=2.0 if is_hovered else 1.2,
                    alpha=0.95 if is_hovered else line_alpha,
                    solid_capstyle="round",
                    zorder=3 if not is_deleted else 1,
                    linestyle="--" if is_deleted else "-",
                )

            # Draw straight center guides only when no curve overlays are available.
            if not have_curve_overlays:
                self.ax_cut.plot(
                    [0.15, 0.85], [cen, cen],
                    color=edge, lw=line_lw, alpha=line_alpha, solid_capstyle="round",
                    zorder=4 if not is_deleted else 1,
                    linestyle="--" if is_deleted else "-",
                )

            # Small aperture index label above each line.
            label_style = "(X) " if is_deleted else ""
            self.ax_cut.text(
                0.50, cen + 0.9, f"{label_style}{i + 1}",
                fontsize=6, ha="center", va="bottom", color=c, zorder=5,
                style="italic" if is_deleted else "normal",
            )

        self._draw_legend()
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    def _refresh_colors(self):
        self._draw_scene()

    # ------------------------------------------------------------------
    def _nearest_aperture(self, y_data):
        """Return index of aperture whose centre is closest to y_data."""
        if len(self.centers) == 0:
            return None
        dists = np.abs(self.centers - y_data)
        idx   = np.argmin(dists)
        if dists[idx] < 10:
            return idx
        return None

    # ------------------------------------------------------------------
    def _on_motion(self, event):
        if event.inaxes != self.ax_cut:
            if self._hovered_ap is not None:
                self._hovered_ap = None
                self._draw_scene()
            return
        y = event.ydata
        if y is None:
            return
        new_hover = self._nearest_aperture(y)
        if new_hover != self._hovered_ap:
            self._hovered_ap = new_hover
            self._draw_scene()

    # ------------------------------------------------------------------
    def _on_key(self, event):
        key = event.key

        # ---- h: help ---------------------------------------------------
        if key == "h":
            print(textwrap.dedent("""
                Interactive aperture viewer controls
                ─────────────────────────────────────
                e   Hover over a trace and press 'e' to manually reassign
                    the star for that aperture.  You will be prompted in the
                    terminal.
                d   Mark/unmark the hovered aperture as deleted (will not be
                    extracted).  Deleted traces appear grayed out and italicized.
                f   Recompute affiliations forward from the last edited aperture.
                    Manual edits and deleted apertures are locked and preserved.
                r   Reset ALL assignments to the auto-generated pattern.
                q/g Accept current mapping, save PNGs, and continue.
                h   Print this help text.
            """))
            return

        # ---- r: reset --------------------------------------------------
        if key == "r":
            self.pattern        = default_pattern(self.n_ap, self.n_stars)
            self.manual_lock[:] = False
            self.deleted[:]     = False
            self.last_edit_idx = None
            self._refresh_colors()
            print("  [reset] All assignments restored to auto-generated pattern.")
            return

        # ---- e: reassign hovered aperture ------------------------------
        if key == "e":
            if self._hovered_ap is None:
                print("  [e] No aperture hovered. Move the mouse closer to a trace.")
                return
            ap_idx     = self._hovered_ap
            old_star   = self.pattern[ap_idx]
            try:
                new_star = int(input(
                    f"\n  Aperture {ap_idx + 1} is currently assigned to Star {old_star}.\n"
                    "  Enter new star number (>=1): "
                ).strip())
            except (ValueError, EOFError):
                print("  [e] Invalid input; assignment unchanged.")
                return
            if new_star < 1:
                print("  [e] Star number must be >= 1; assignment unchanged.")
                return
            self.pattern[ap_idx] = new_star
            self.manual_lock[ap_idx] = True
            self.deleted[ap_idx] = False  # Clear deleted flag if reassigning
            self.last_edit_idx = ap_idx
            self._refresh_colors()
            print(
                f"  [e] Aperture {ap_idx + 1} -> Star {new_star} (locked). "
                "Press 'f' to recompute forward from here."
            )
            return

        # ---- d: mark aperture as deleted/unused -------------------------
        if key == "d":
            if self._hovered_ap is None:
                print("  [d] No aperture hovered. Move the mouse closer to a trace.")
                return
            ap_idx = self._hovered_ap
            self.deleted[ap_idx] = not self.deleted[ap_idx]
            self.manual_lock[ap_idx] = True  # Lock deleted state
            self.last_edit_idx = ap_idx
            self._refresh_colors()
            status = "DELETED (will not be extracted)" if self.deleted[ap_idx] else "restored"
            print(f"  [d] Aperture {ap_idx + 1} marked as {status}.")
            return

        # ---- f: recompute forward with lock preservation ----------------
        if key == "f":
            if self.last_edit_idx is None:
                print("  [f] No edited aperture anchor available (use 'e' first).")
                return
            # Treat both manual locks and deleted apertures as locked (not recomputed)
            combined_locks = self.manual_lock | self.deleted
            self.pattern = recompute_forward_with_locks(
                self.pattern, combined_locks, self.last_edit_idx
            )
            self._refresh_colors()
            print(
                f"  [f] Forward affiliation recompute from aperture {self.last_edit_idx + 1}. "
                "Manual locks and deleted apertures preserved."
            )
            return

        # ---- q/g: accept mapping and finish -----------------------------
        if key in ("q", "g"):
            print("\n  [q/g] Mapping accepted.  Closing preview window.")
            try:
                self._save_confirmation_figures(dpi=300)
            except Exception as exc:
                print(f"  [warn] Could not save confirmation figures: {exc}")
            self.done = True
            plt.close(self.fig)
            return

    # ------------------------------------------------------------------
    def show(self):
        plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive aperture-trace preview before apall extraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("image", help="Stacked science FITS image (mosaic).")
    p.add_argument("--nap",    type=int,   default=None,
                   help="Number of apertures (default: auto-detect).")
    p.add_argument("--sep",    type=float, default=8.0,
                   help="Expected separation between apertures in px (default: 8).")
    p.add_argument("--col",    type=int,   default=None,
                   help="Dispersion column for the spatial cut (default: image centre).")
    p.add_argument("--nstars", type=int, default=24,
                   help="Number of unique stars in the repeating pattern (default: 24).")
    p.add_argument("--out",    type=str,   default=None,
                   help="Output file for the final aperture→star mapping.")
    return p.parse_args()


def load_image(path):
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                return hdu.data.astype(float)
    raise ValueError(f"No 2-D image data found in {path}")


def save_mapping(path, image_name, centers, pattern):
    with open(path, "w") as f:
        f.write(f"# Aperture → Star mapping for {image_name}\n")
        f.write(f"# {'ap':>4}  {'center_px':>10}  {'star':>5}\n")
        for i, (cen, star) in enumerate(zip(centers, pattern)):
            f.write(f"  {i+1:>4}  {cen:>10.2f}  {star:>5}\n")
    print(f"  Mapping saved to: {path}")


def main():
    args = parse_args()

    # ── load image ──────────────────────────────────────────────────────────
    print(f"\nLoading {args.image} …")
    data = load_image(args.image)
    nrows, ncols = data.shape
    print(f"  Image shape: {nrows} rows × {ncols} columns")

    col = args.col if args.col is not None else ncols // 2

    trace_info = load_iraf_traces(args.image, ncols)
    trace_rows = trace_info["trace_rows"]

    # ── find aperture centres ────────────────────────────────────────────────
    if trace_info["source"] == "database":
        print(
            "  Using IRAF trace database "
            f"({trace_info['parsed']} apertures parsed, {trace_info['skipped']} skipped):\n"
            f"    {trace_info['db_path']}"
        )
        centers = trace_info["centers"]
    else:
        print(f"  Finding aperture centres (spatial cut at col={col}) …")
        try:
            from scipy.signal import find_peaks   # noqa: F401  (just check availability)
            centers = find_aperture_centers(data, col=col, expected_sep=args.sep)
        except ImportError:
            print("  scipy not available – using uniform grid as fallback.")
            n = args.nap if args.nap else int(nrows / max(1, args.sep))
            centers = np.linspace(args.sep, nrows - args.sep, n)

    if args.nap is not None and len(centers) != args.nap:
        print(f"  Detected {len(centers)} peaks; trimming/padding to requested {args.nap}.")
        if len(centers) > args.nap:
            centers = centers[:args.nap]
            trace_rows = {k: v for k, v in trace_rows.items() if k < args.nap}
        else:
            # pad with uniform grid entries
            extra = np.linspace(centers[-1] + args.sep,
                                nrows - 1, args.nap - len(centers))
            centers = np.concatenate([centers, extra])

    n_ap = len(centers)
    print(f"  Apertures found: {n_ap}")

    # ── build initial star pattern ──────────────────────────────────────────
    pattern = default_pattern(n_ap, n_stars=args.nstars)

    unique_stars = np.unique(pattern)
    print(f"  Stars in pattern: {unique_stars}")
    for s in unique_stars:
        aps = np.where(pattern == s)[0] + 1
        print(f"    Star {s}: {len(aps)} apertures")

    # ── interactive viewer ──────────────────────────────────────────────────
    image_name = os.path.basename(args.image)
    outpath = args.out or f"{os.path.splitext(image_name)[0]}_star_map.txt"
    save_prefix = str(Path(outpath).with_suffix(""))

    viewer     = TraceViewer(
        data,
        centers,
        pattern,
        image_name,
        col,
        args.nstars,
        trace_rows=trace_rows,
        save_prefix=save_prefix,
    )

    print("\n  Opening interactive preview window …")
    print("  Press [h] inside the window for a control summary.\n")
    viewer.show()

    if not viewer.done:
        raise RuntimeError(
            "Preview closed without confirmation. Press 'q' twice to accept the mapping "
            "and trigger PNG exports."
        )

    # After window closes
    final_pattern = viewer.pattern.copy()
    final_centers = viewer.centers
    # Mark deleted apertures as 0 so they are skipped during extraction
    final_pattern[viewer.deleted] = 0

    # ── print final mapping ──────────────────────────────────────────────────
    print("\n── Final aperture → star mapping ──────────────────────────────────")
    for i, (cen, star) in enumerate(zip(final_centers, final_pattern)):
        print(f"  ap {i+1:>4}  row {cen:>7.1f}  star {star}")

    # ── save mapping ─────────────────────────────────────────────────────────
    save_mapping(outpath, image_name, final_centers, final_pattern)

    return final_centers, final_pattern


# ── convenience wrapper for calling from a notebook / other script ──────────

def run_preview(image_path, nap=None, sep=5.0, col=None, n_stars=4, out=None):
    """
    Programmatic entry point – mirrors the CLI but returns (centers, pattern).

    Example (from notebook after mosaics are assembled):

        from aperture_preview import run_preview
        centers, pattern = run_preview("jsimonScl_h3_sci_15Sep2014_R-b-d-cr.fits",
                                        n_stars=4)
        # then call apall …
    """
    data = load_image(image_path)
    nrows, ncols = data.shape
    _col = col if col is not None else ncols // 2

    trace_info = load_iraf_traces(image_path, ncols)
    trace_rows = trace_info["trace_rows"]

    if trace_info["source"] == "database":
        print(
            "  Trace source: database "
            f"({trace_info['parsed']} parsed, {trace_info['skipped']} skipped)"
        )
        print(f"  DB file: {trace_info['db_path']}")
        centers = trace_info["centers"]
    else:
        print("  Trace source: peak finder fallback")
        print(f"  Finding aperture centres (spatial cut at col={_col}) …")

        try:
            centers = find_aperture_centers(data, col=_col, expected_sep=sep)
        except ImportError:
            n = nap if nap else int(nrows / max(1, sep))
            centers = np.linspace(sep, nrows - sep, n)

    if nap is not None and len(centers) != nap:
        if len(centers) > nap:
            centers = centers[:nap]
            trace_rows = {k: v for k, v in trace_rows.items() if k < nap}
        else:
            extra = np.linspace(centers[-1] + sep, nrows - 1, nap - len(centers))
            centers = np.concatenate([centers, extra])

    pattern = default_pattern(len(centers), n_stars=n_stars)
    image_name = os.path.basename(image_path)
    outpath = out or f"{os.path.splitext(image_name)[0]}_star_map.txt"
    save_prefix = str(Path(outpath).with_suffix(""))

    viewer = TraceViewer(
        data,
        centers,
        pattern,
        image_name,
        _col,
        n_stars,
        trace_rows=trace_rows,
        save_prefix=save_prefix,
    )

    print(f"\n  Opening interactive preview for {image_name} …")
    print("  Press [h] inside the window for controls.\n")
    viewer.show()

    if not viewer.done:
        raise RuntimeError(
            "Preview closed without confirmation. Press 'q' twice to accept the mapping "
            "and trigger PNG exports."
        )

    final_pattern = viewer.pattern.copy()
    # Mark deleted apertures as 0 so they are skipped during extraction
    final_pattern[viewer.deleted] = 0
    save_mapping(outpath, image_name, viewer.centers, final_pattern)

    return viewer.centers, final_pattern


if __name__ == "__main__":
    main()
