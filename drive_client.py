"""Google Drive API integration — metadata filtering & streaming downloads."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build, Resource
from googleapiclient.http import MediaIoBaseDownload

from config import PipelineConfig

log = logging.getLogger(__name__)

IMAGE_MIME_TYPES = frozenset({
    "image/jpeg",
    "image/png",
    "image/tiff",
    "image/bmp",
    "image/webp",
})


@dataclass
class DriveImage:
    """Lightweight carrier for a Drive file that passed the metadata filter."""
    file_id: str
    name: str
    mime_type: str
    width: int
    height: int
    megapixels: float


class DriveClient:
    """Handles authentication, metadata querying, and byte streaming."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self._cfg = cfg
        self._service: Resource = self._authenticate()

    # ── Auth ──────────────────────────────────────────────────────────

    def _authenticate(self) -> Resource:
        creds: Credentials | None = None
        token_path = Path(self._cfg.token_path)

        if token_path.exists():
            creds = Credentials.from_authorized_user_file(
                str(token_path), list(self._cfg.scopes)
            )

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(self._cfg.credentials_path), list(self._cfg.scopes)
                )
                creds = flow.run_local_server(port=0)
            token_path.write_text(creds.to_json())

        return build("drive", "v3", credentials=creds)

    # ── Metadata query ────────────────────────────────────────────────

    def list_eligible_images(self) -> Generator[DriveImage, None, None]:
        """Yield images from the target folder that meet the megapixel floor."""
        query = (
            f"'{self._cfg.folder_id}' in parents and trashed = false"
        )
        page_token: str | None = None

        while True:
            resp = (
                self._service.files()
                .list(
                    q=query,
                    fields=(
                        "nextPageToken, "
                        "files(id, name, mimeType, size, "
                        "imageMediaMetadata(width, height))"
                    ),
                    pageSize=self._cfg.page_size,
                    pageToken=page_token,
                )
                .execute()
            )

            for f in resp.get("files", []):
                mime = f.get("mimeType", "")
                if mime not in IMAGE_MIME_TYPES:
                    continue

                meta = f.get("imageMediaMetadata")
                if not meta or "width" not in meta or "height" not in meta:
                    log.debug("Skipped %s — no image dimensions in metadata", f["name"])
                    continue

                w, h = int(meta["width"]), int(meta["height"])
                total_px = w * h

                if total_px < self._cfg.min_pixels:
                    log.info(
                        "SKIP (resolution) %s — %.1f MP < %.1f MP",
                        f["name"], total_px / 1e6, self._cfg.min_megapixels,
                    )
                    continue

                yield DriveImage(
                    file_id=f["id"],
                    name=f["name"],
                    mime_type=mime,
                    width=w,
                    height=h,
                    megapixels=round(total_px / 1e6, 2),
                )

            page_token = resp.get("nextPageToken")
            if not page_token:
                break

    # ── Byte streaming ────────────────────────────────────────────────

    def stream_file(self, file_id: str) -> bytes:
        """Download a file entirely into memory and return raw bytes."""
        request = self._service.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buf.getvalue()

    def download_original(self, file_id: str, dest: Path) -> Path:
        """Download the original file to *dest* (a directory). Returns the saved path."""
        # Fetch real filename for the saved copy.
        meta = (
            self._service.files()
            .get(fileId=file_id, fields="name")
            .execute()
        )
        out_path = dest / meta["name"]

        # Handle duplicate filenames.
        counter = 1
        while out_path.exists():
            stem = Path(meta["name"]).stem
            suffix = Path(meta["name"]).suffix
            out_path = dest / f"{stem}_{counter}{suffix}"
            counter += 1

        raw = self.stream_file(file_id)
        out_path.write_bytes(raw)
        log.info("Saved original → %s", out_path)
        return out_path
