import sys
import unittest
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "label_studio_backend"))

import security_utils as su


class SecurityUtilsTest(unittest.TestCase):
    def test_build_allowed_hosts_includes_primary_and_extras(self):
        hosts = su.build_allowed_hosts(
            "http://label-studio:8080",
            "cdn.example.com, files.example.com:8443",
        )
        self.assertEqual(
            hosts,
            [
                su.HostRule("label-studio", 8080),
                su.HostRule("cdn.example.com", None),
                su.HostRule("files.example.com", 8443),
            ],
        )

    def test_is_allowed_host_with_and_without_port(self):
        hosts = [
            su.HostRule("label-studio", 8080),
            su.HostRule("cdn.example.com", None),
        ]
        self.assertTrue(su.is_allowed_host(urlparse("http://label-studio:8080/a"), hosts))
        self.assertTrue(su.is_allowed_host(urlparse("https://cdn.example.com/file"), hosts))
        self.assertFalse(su.is_allowed_host(urlparse("http://label-studio:80/a"), hosts))
        self.assertFalse(su.is_allowed_host(urlparse("http://evil.example.com/a"), hosts))

    def test_validate_remote_http_url_rejects_unlisted_host(self):
        hosts = su.build_allowed_hosts("http://label-studio:8080", "")
        with self.assertRaises(ValueError):
            su.validate_remote_http_url("http://evil.example.com/file.pdf", hosts)

    def test_parse_authorization_token(self):
        self.assertEqual(su.parse_authorization_token("Bearer abc123"), "abc123")
        self.assertEqual(su.parse_authorization_token("Token abc123"), "abc123")
        self.assertEqual(su.parse_authorization_token("abc123"), "abc123")
        self.assertEqual(su.parse_authorization_token(""), "")

    def test_verify_shared_secret(self):
        self.assertTrue(su.verify_shared_secret("secret", "secret"))
        self.assertFalse(su.verify_shared_secret("wrong", "secret"))
        self.assertFalse(su.verify_shared_secret("secret", ""))


if __name__ == "__main__":
    unittest.main()
