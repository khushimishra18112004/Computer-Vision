"""DocuVision - a classical computer vision document scanner."""

from .pipeline import ScanResult, save_outputs, scan, scan_file

__version__ = "1.0.0"
__all__ = ["ScanResult", "scan", "scan_file", "save_outputs"]
