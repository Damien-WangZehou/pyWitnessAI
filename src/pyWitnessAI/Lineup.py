import os
import random
import logging
from PIL import Image
import math
from typing import List, Optional, Dict, Literal, Tuple

# logging.basicConfig(
#     level=logging.DEBUG,  # Set to DEBUG for detailed trace
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('lineuploader.log'),
#         logging.StreamHandler()  # to console
#     ]
# )

class Lineup:
    def __init__(self, folder_path: str, guilty_suspect: Optional[str] = None, innocent_suspect: Optional[str] = None):
        self.folder_path = folder_path
        self.guilty_suspect = guilty_suspect
        self.innocent_suspect = innocent_suspect

        '''
        Example of image_groups structure:
        { "guilty_suspect": ["/path/Video1_Perpetrator.png"],
          "innocent_suspect": ["/path/Video1_Innocent.png"],
          "filler": ["/path/Filler1.png", "/path/Filler2.png", ...]}
        '''
        self.image_groups = self._load_and_classify(folder_path)

        # Name mapping for easy reference
        self.names: Dict[str, str] = {}
        for p in sum(self.image_groups.values(), []):
            base_name = os.path.splitext(os.path.basename(p))[0]
            self.names[base_name] = p

        self.perp_name: Optional[str] = self._first_name("guilty_suspect")
        self.innocent_name: Optional[str] = self._first_name("innocent_suspect")
        self.filler_names: List[str] = [os.path.splitext(os.path.basename(p))[0]
                                        for p in self.image_groups.get("filler", [])]

        # Error when neither exists
        if not self.image_groups["guilty_suspect"] and not self.image_groups["innocent_suspect"]:
            raise ValueError("Missing guilty suspect image. Please rename the perpetrator image or provide it.")

        # Default lineup (if perp -> TP; otherwise -> TA)
        self.default_target: Literal["targetPresent", "targetAbsent"] = (
            "targetPresent" if self.perp_name else "targetAbsent"
        )

        # Default: If none of targetLineup and lineup_size provided,
        # all images will be used in a TA overlap/or TP lineup
        self.lineup: List[str] = self.generate_lineup()

    def _normalize_provided_path(self, p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        if os.path.isabs(p):
            return p
        joined = os.path.join(self.folder_path, p)
        return os.path.normpath(joined) if os.path.exists(joined) else os.path.normpath(p)

    def _load_and_classify(self, folder_path: str) -> Dict[str, List[str]]:
        # image_paths = glob.glob(os.path.join(folder_path, "*.png"))
        exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
        candidates: List[str] = []
        for name in os.listdir(folder_path):
            path = os.path.join(folder_path, name)
            if not os.path.isfile(path):
                continue
            _, ext = os.path.splitext(name)
            if ext.lower() in exts:
                candidates.append(path)

        # Remove duplicates (case-insensitive, absolute path)
        seen = set()
        uniq_paths: List[str] = []
        for p in candidates:
            key = os.path.normcase(os.path.abspath(p))
            if key not in seen:
                uniq_paths.append(os.path.normpath(p))
                seen.add(key)

        image_dict = {"guilty_suspect": [], "innocent_suspect": [], "filler": []}

        self.guilty_suspect = self._normalize_provided_path(self.guilty_suspect)
        self.innocent_suspect = self._normalize_provided_path(self.innocent_suspect)

        for image_path in uniq_paths:
            name_lowercase = os.path.basename(image_path).lower()
            if self.guilty_suspect and os.path.abspath(image_path) == os.path.abspath(self.guilty_suspect):
                image_dict["guilty_suspect"].append(image_path)
            elif self.innocent_suspect and os.path.abspath(image_path) == os.path.abspath(self.innocent_suspect):
                image_dict["innocent_suspect"].append(image_path)
            elif "perp" in name_lowercase:
                image_dict["guilty_suspect"].append(image_path)
                self.guilty_suspect = image_path
            elif "designated" in name_lowercase or "innocent" in name_lowercase:
                image_dict["innocent_suspect"].append(image_path)
                self.innocent_suspect = image_path
            else:
                image_dict["filler"].append(image_path)

        if self.guilty_suspect:
            image_dict["guilty_suspect"] = [self.guilty_suspect]
        if self.innocent_suspect:
            image_dict["innocent_suspect"] = [self.innocent_suspect]

        # Keep only unique paths in each category
        def _dedup(paths: List[str]) -> List[str]:
            s, out = set(), []
            for p in paths:
                k = os.path.normcase(os.path.abspath(p))
                if k not in s:
                    out.append(os.path.normpath(p))
                    s.add(k)
            return out

        image_dict["guilty_suspect"] = _dedup([self.guilty_suspect] if self.guilty_suspect else [])
        image_dict["innocent_suspect"] = _dedup([self.innocent_suspect] if self.innocent_suspect else [])
        image_dict["filler"] = _dedup(image_dict["filler"])

        logging.debug(f"Guilty suspect image: {image_dict['guilty_suspect']}")
        logging.debug(f"Innocent suspect image: {image_dict['innocent_suspect']}")
        logging.debug(f"Number of fillers: {len(image_dict['filler'])}")
        return image_dict

    def _first_name(self, category: str) -> Optional[str]:
        paths = self.image_groups.get(category, [])
        if not paths:
            return None
        return os.path.splitext(os.path.basename(paths[0]))[0]

    def generate_lineup(self, lineup_size: Optional[int] = None,
                        target_lineup: Optional[Literal["targetPresent", "targetAbsent"]] = None,
                        shuffle: bool = False) -> List[str]:
        """
        Rules:
        - If target_lineup and lineup_size provided, use them.
        - If neither provided, all images will be used in a TA overlap/or TP lineup.
        - Otherwise, use guilty/innocent suspect + fillers to fill the lineup.
        - lineup_size (default
        """

        total_images = (
                len(self.image_groups.get("guilty_suspect", [])) +
                len(self.image_groups.get("innocent_suspect", [])) +
                len(self.image_groups.get("filler", []))
        )

        # If no lineup_size provided, use all images
        if lineup_size is None:
            lineup_size = total_images

        if target_lineup is None and lineup_size == total_images:
            lineup: List[str] = []
            if self.image_groups["guilty_suspect"]:
                lineup.append(self.image_groups["guilty_suspect"][0])
            if self.image_groups["innocent_suspect"]:
                lineup.append(self.image_groups["innocent_suspect"][0])
            lineup.extend(self.image_groups["filler"])
            if shuffle:
                random.shuffle(lineup)
            self.lineup = lineup
            return self.lineup

        target = target_lineup or self.default_target

        if target == "targetAbsent":
            assert self.image_groups["innocent_suspect"], "Innocent suspect image required for targetAbsent."
            anchor = self.image_groups["innocent_suspect"][0]
        else:
            assert self.image_groups["guilty_suspect"], "Guilty suspect image required for targetPresent."
            anchor = self.image_groups["guilty_suspect"][0]

        need_fillers = max(0, lineup_size - 1)
        fillers = list(self.image_groups["filler"])

        if len(fillers) < need_fillers:
            raise ValueError("Insufficient fillers to generate lineup.")

        lineup = [anchor] + fillers[:need_fillers]
        if shuffle:
            random.shuffle(lineup)

        self.lineup = lineup[:lineup_size]
        return self.lineup

    def show_lineup(self,
                    rows: Optional[int] = None,
                    cols: Optional[int] = None,
                    max_cols: int = 6,
                    figsize_per_tile: Tuple[float, float] = (3.0, 3.0),
                    annotate_role: bool = True,
                    suptitle: Optional[str] = None):
        """
        Automatically arrange and display the lineup images in a grid.
        - If neither rows nor cols provided, use a square-like layout.
        - If only one provided, calculate the other to fit all images.
        - If both provided, use them directly (may leave empty spaces).
        - annotate_role: whether to append role info in titles.
        """
        n = len(self.lineup)
        if n == 0:
            raise ValueError("No images in lineup to display.")

        # Calculate rows and cols if needed
        if rows is None and cols is None:
            cols = min(max_cols, int(math.ceil(math.sqrt(n))))
            rows = int(math.ceil(n / cols))
        elif rows is None:
            cols = max(1, cols)
            cols = min(cols, max_cols)
            rows = int(math.ceil(n / cols))
        elif cols is None:
            rows = max(1, rows)
            cols = int(math.ceil(n / rows))
            cols = min(cols, max_cols)

        # Estimate average aspect ratio for height adjustment
        ratios = []
        for p in self.lineup:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                    if w > 0:
                        ratios.append(h / float(w))
            except Exception:
                continue
        avg_ratio = sum(ratios) / len(ratios) if ratios else 1.0

        tile_w, tile_h = figsize_per_tile
        fig_w = max(1.0, cols * tile_w)
        fig_h = max(1.0, rows * tile_h * avg_ratio)

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
        # Make axes a 2D list for uniform access
        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        # Role determination function
        def role_of(path: str) -> str:
            base = os.path.splitext(os.path.basename(path))[0]
            if base == self.perp_name:
                return "guilty_suspect"
            if base == self.innocent_name:
                return "innocent_suspect"
            return "filler"

        idx = 0
        for r in range(rows):
            for c in range(cols):
                ax = axes[r][c]
                ax.axis("off")
                if idx < n:
                    p = self.lineup[idx]
                    try:
                        img = Image.open(p)
                        ax.imshow(img)
                        title = os.path.splitext(os.path.basename(p))[0]
                        if annotate_role:
                            title += f"  [{role_of(p)}]"
                        ax.set_title(title, fontsize=9)
                    except Exception as e:
                        ax.text(0.5, 0.5, f"Failed to load\n{os.path.basename(p)}",
                                ha="center", va="center", fontsize=8)
                    idx += 1

        if suptitle:
            fig.suptitle(suptitle, fontsize=12)
        plt.tight_layout()
        plt.show()