import io
import unittest

try:
    from PIL import Image
    import streamlit_app
    DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - CI installs these, local may not
    DEPS_AVAILABLE = False


class _FakeUploadedFile:
    """Minimal stand-in for Streamlit's ``UploadedFile``.

    Wraps raw bytes and exposes the ``.size`` attribute plus the file-like
    ``read``/``seek`` interface that ``PIL.Image.open`` needs.
    """

    def __init__(self, data: bytes):
        self._buffer = io.BytesIO(data)
        self.size = len(data)

    def read(self, *args, **kwargs):
        return self._buffer.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self._buffer.seek(*args, **kwargs)


def _make_image_bytes(fmt: str = 'PNG') -> bytes:
    img = Image.new('RGB', (10, 10), color=(123, 222, 64))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@unittest.skipUnless(DEPS_AVAILABLE, "streamlit/PIL are not installed")
class TestValidateUpload(unittest.TestCase):
    """Tests for the upload validation that guards the prediction path.

    This covers the ``# TODO input validation`` fix: uploads must be within
    the size limit and decodable as an image before they reach the model.
    """

    def test_valid_image_returns_pil_image(self):
        uploaded = _FakeUploadedFile(_make_image_bytes())
        image = streamlit_app.validate_upload(uploaded)
        self.assertIsInstance(image, Image.Image)

    def test_oversized_file_raises(self):
        oversized = _FakeUploadedFile(
            b'0' * (streamlit_app.MAX_UPLOAD_BYTES + 1))
        with self.assertRaises(ValueError):
            streamlit_app.validate_upload(oversized)

    def test_oversized_error_message_mentions_size_limit(self):
        oversized = _FakeUploadedFile(
            b'0' * (streamlit_app.MAX_UPLOAD_BYTES + 1))
        with self.assertRaises(ValueError) as ctx:
            streamlit_app.validate_upload(oversized)
        self.assertIn("too large", str(ctx.exception))

    def test_undecodable_file_raises(self):
        uploaded = _FakeUploadedFile(b'this is not a valid image')
        with self.assertRaises(ValueError):
            streamlit_app.validate_upload(uploaded)

    def test_undecodable_error_message_is_helpful(self):
        uploaded = _FakeUploadedFile(b'this is not a valid image')
        with self.assertRaises(ValueError) as ctx:
            streamlit_app.validate_upload(uploaded)
        self.assertIn("valid image", str(ctx.exception))

    def test_file_at_exact_size_limit_is_allowed(self):
        data = _make_image_bytes()
        uploaded = _FakeUploadedFile(data)
        uploaded.size = streamlit_app.MAX_UPLOAD_BYTES
        image = streamlit_app.validate_upload(uploaded)
        self.assertIsInstance(image, Image.Image)


if __name__ == '__main__':
    unittest.main()