"""Who may see a viewport (/api/viewports/<name>/info).

Regression: an automated security PR (#7) gated viewport_info behind
check_viewport_owner -- the owner-only check meant for changes -- so every
viewport returned 403 to anyone but its creator, including every anonymous
demo-mode visitor ("embeddings don't load"). Reads follow the viewport
list's rule instead: public for everyone, private for owner and admins.
"""

import json
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tee_project.settings")
import django  # noqa: E402

django.setup()

import pytest  # noqa: E402
from django.contrib.auth.models import AnonymousUser  # noqa: E402
from django.test import RequestFactory  # noqa: E402

from api import helpers  # noqa: E402


class _User:
    def __init__(self, username, superuser=False):
        self.username = username
        self.is_authenticated = True
        self.is_superuser = superuser


@pytest.fixture
def viewports(tmp_path, monkeypatch):
    monkeypatch.setattr(helpers, "VIEWPORTS_DIR", tmp_path)
    (tmp_path / "Public_config.json").write_text(json.dumps({"created_by": "alice", "private": False}))
    (tmp_path / "Secret_config.json").write_text(json.dumps({"created_by": "alice", "private": True}))
    return tmp_path


def _request(user):
    request = RequestFactory().get("/api/viewports/x/info")
    request.user = user
    return request


@pytest.mark.parametrize("user", [AnonymousUser(), _User("bob"), _User("alice"), _User("root", True)])
def test_public_viewport_is_readable_by_everyone(viewports, user):
    assert helpers.check_viewport_read(_request(user), "Public")[0]


@pytest.mark.parametrize("user,allowed", [
    (AnonymousUser(), False), (_User("bob"), False), (_User("alice"), True), (_User("root", True), True),
])
def test_private_viewport_is_owner_and_admin_only(viewports, user, allowed):
    assert helpers.check_viewport_read(_request(user), "Secret")[0] is allowed


def test_legacy_viewport_without_config_is_readable(viewports):
    assert helpers.check_viewport_read(_request(AnonymousUser()), "Legacy")[0]


def test_changes_stay_owner_only(viewports):
    """check_viewport_owner (delete/add years/cancel) is unchanged."""
    assert not helpers.check_viewport_owner(_request(_User("bob")), "Public")[0]
    assert helpers.check_viewport_owner(_request(_User("alice")), "Public")[0]


def test_info_endpoint_lets_anonymous_visitors_see_public_viewports(viewports, monkeypatch):
    from api.views import viewports as vp

    monkeypatch.setattr(vp, "read_viewport_file", lambda name: {"bounds": [0, 0, 1, 1]})
    monkeypatch.setattr(vp, "get_viewport_vq_config", lambda name: None)
    assert vp.viewport_info(_request(AnonymousUser()), "Public").status_code == 200
    assert vp.viewport_info(_request(AnonymousUser()), "Secret").status_code == 403
