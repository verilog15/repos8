#
# SecureDrop whistleblower submission system
# Copyright (C) 2017- Freedom of the Press Foundation and SecureDrop
# contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import argparse
import json
import os
import subprocess
import textwrap
from os.path import basename, dirname, exists, join
from unittest import mock

import pytest
import securedrop_admin
import yaml
from flaky import flaky
from prompt_toolkit.validation import ValidationError


class Document:
    def __init__(self, text):
        self.text = text


@flaky
class TestSecureDropAdmin:
    def test_verbose(self, capsys):
        securedrop_admin.setup_logger(verbose=True)
        securedrop_admin.sdlog.debug("VISIBLE")
        out, err = capsys.readouterr()
        assert "VISIBLE" in out

    def test_not_verbose(self, capsys):
        securedrop_admin.setup_logger(verbose=False)
        securedrop_admin.sdlog.debug("HIDDEN")
        securedrop_admin.sdlog.info("VISIBLE")
        out, err = capsys.readouterr()
        assert "HIDDEN" not in out
        assert "VISIBLE" in out

    def test_openssh_detection(self):
        with mock.patch("securedrop_admin.openssh_version", side_effect=[9]):
            assert securedrop_admin.ansible_command() == [
                "ansible-playbook",
                "--scp-extra-args='-O'",
            ]
        with mock.patch("securedrop_admin.openssh_version", side_effect=[8]):
            assert securedrop_admin.ansible_command() == ["ansible-playbook"]

    def test_update_check_decorator_when_no_update_needed(self, caplog):
        """
        When a function decorated with `@update_check_required` is run
          And the `--force` argument was not given
          And no update is required
        Then the update check should run to completion
          And no errors should be displayed
          And the program should not exit
          And the decorated function should be run
        """
        with mock.patch(
            "securedrop_admin.check_for_updates", side_effect=[[False, "1.5.0"]]
        ) as mocked_check, mock.patch(
            "securedrop_admin.get_git_branch", side_effect=["develop"]
        ), mock.patch("sys.exit") as mocked_exit:
            # The decorator itself interprets --force
            args = argparse.Namespace(force=False)
            rv = securedrop_admin.update_check_required("update_check_test")(lambda _: 100)(args)
            assert mocked_check.called
            assert not mocked_exit.called
            assert rv == 100
            assert caplog.text == ""

    def test_update_check_decorator_when_update_needed(self, caplog):
        """
        When a function decorated with `@update_check_required` is run
          And the `--force` argument was not given
          And an update is required
        Then the update check should run to completion
          And an error referencing the command should be displayed
          And the current branch state should be included in the output
          And the program should exit
        """
        with mock.patch(
            "securedrop_admin.check_for_updates", side_effect=[[True, "1.5.0"]]
        ) as mocked_check, mock.patch(
            "securedrop_admin.get_git_branch", side_effect=["bad_branch"]
        ), mock.patch("sys.exit") as mocked_exit:
            # The decorator itself interprets --force
            args = argparse.Namespace(force=False)
            securedrop_admin.update_check_required("update_check_test")(lambda _: _)(args)
            assert mocked_check.called
            assert mocked_exit.called
            assert "update_check_test" in caplog.text
            assert "bad_branch" in caplog.text

    def test_update_check_decorator_when_skipped(self, caplog):
        """
        When a function decorated with `@update_check_required` is run
          And the `--force` argument was given
        Then the update check should not run
          And a message should be displayed acknowledging this
          And the program should not exit
          And the decorated function should be run
        """
        with mock.patch(
            "securedrop_admin.check_for_updates", side_effect=[[True, "1.5.0"]]
        ) as mocked_check, mock.patch(
            "securedrop_admin.get_git_branch", side_effect=["develop"]
        ), mock.patch("sys.exit") as mocked_exit:
            # The decorator itself interprets --force
            args = argparse.Namespace(force=True)
            rv = securedrop_admin.update_check_required("update_check_test")(lambda _: 100)(args)
            assert not mocked_check.called
            assert not mocked_exit.called
            assert "--force" in caplog.text
            assert rv == 100

    def test_check_for_updates_update_needed(self, tmpdir, caplog):
        git_repo_path = str(tmpdir)
        args = argparse.Namespace(root=git_repo_path)
        current_tag = b"0.6"
        tags_available = b"0.6\n0.6-rc1\n0.6.1\n"

        with mock.patch("subprocess.check_call"):
            with mock.patch("subprocess.check_output", side_effect=[current_tag, tags_available]):
                update_status, tag = securedrop_admin.check_for_updates(args)
                assert "Update needed" in caplog.text
                assert update_status is True
                assert tag == "0.6.1"

    def test_check_for_updates_higher_version(self, tmpdir, caplog):
        git_repo_path = str(tmpdir)
        args = argparse.Namespace(root=git_repo_path)
        current_tag = b"0.6"
        tags_available = b"0.1\n0.10.0\n0.6.2\n0.6\n0.6-rc1\n0.9.0\n"

        with mock.patch("subprocess.check_call"):
            with mock.patch("subprocess.check_output", side_effect=[current_tag, tags_available]):
                update_status, tag = securedrop_admin.check_for_updates(args)
                assert "Update needed" in caplog.text
                assert update_status is True
                assert tag == "0.10.0"

    def test_check_for_updates_ensure_newline_stripped(self, tmpdir, caplog):
        """Regression test for #3426"""
        git_repo_path = str(tmpdir)
        args = argparse.Namespace(root=git_repo_path)
        current_tag = b"0.6.1\n"
        tags_available = b"0.6\n0.6-rc1\n0.6.1\n"

        with mock.patch("subprocess.check_call"):
            with mock.patch("subprocess.check_output", side_effect=[current_tag, tags_available]):
                update_status, tag = securedrop_admin.check_for_updates(args)
                assert "All updates applied" in caplog.text
                assert update_status is False
                assert tag == "0.6.1"

    def test_check_for_updates_update_not_needed(self, tmpdir, caplog):
        git_repo_path = str(tmpdir)
        args = argparse.Namespace(root=git_repo_path)
        current_tag = b"0.6.1"
        tags_available = b"0.6\n0.6-rc1\n0.6.1\n"

        with mock.patch("subprocess.check_call"):
            with mock.patch("subprocess.check_output", side_effect=[current_tag, tags_available]):
                update_status, tag = securedrop_admin.check_for_updates(args)
                assert "All updates applied" in caplog.text
                assert update_status is False
                assert tag == "0.6.1"

    def test_check_for_updates_if_most_recent_tag_is_rc(self, tmpdir, caplog):
        """During pre-release QA, the most recent tag ends in *-rc. Let's
        verify that users will not accidentally check out this tag."""
        git_repo_path = str(tmpdir)
        args = argparse.Namespace(root=git_repo_path)
        current_tag = b"0.6.1"
        tags_available = b"0.6\n0.6-rc1\n0.6.1\n0.6.1-rc1\n"

        with mock.patch("subprocess.check_call"):
            with mock.patch("subprocess.check_output", side_effect=[current_tag, tags_available]):
                update_status, tag = securedrop_admin.check_for_updates(args)
                assert "All updates applied" in caplog.text
                assert update_status is False
                assert tag == "0.6.1"

    @pytest.mark.parametrize(
        ("git_output", "expected_rv"),
        [
            (b"* develop\n", "develop"),
            (b" develop\n" b"* release/1.7.0\n", "release/1.7.0"),
            (
                b"* (HEAD detached at 1.7.0)\n" b"  develop\n" b"  release/1.7.0\n",
                "(HEAD detached at 1.7.0)",
            ),
            (b"  main\n" b"* valid_+!@#$%&_branch_name\n", "valid_+!@#$%&_branch_name"),
            (b"Unrecognized output.", None),
        ],
    )
    def test_get_git_branch(self, git_output, expected_rv):
        """
        When `git branch` completes with exit code 0
          And the output conforms to the expected format
          Then `get_git_branch` should return a description of the current HEAD

        When `git branch` completes with exit code 0
          And the output does not conform to the expected format
          Then `get_git_branch` should return `None`
        """
        args = argparse.Namespace(root=None)
        with mock.patch("subprocess.check_output", side_effect=[git_output]):
            rv = securedrop_admin.get_git_branch(args)
            assert rv == expected_rv

    def test_update_exits_if_not_needed(self, tmpdir, caplog):
        git_repo_path = str(tmpdir)
        args = argparse.Namespace(root=git_repo_path)

        with mock.patch("securedrop_admin.check_for_updates", return_value=(False, "0.6.1")):
            ret_code = securedrop_admin.update(args)
            assert "Applying SecureDrop updates..." in caplog.text
            assert "Updated to SecureDrop" not in caplog.text
            assert ret_code == 0

    def test_get_release_key_from_valid_keyserver(self, tmpdir, caplog):
        git_repo_path = str(tmpdir)
        args = argparse.Namespace(root=git_repo_path)
        with mock.patch("subprocess.check_call"):
            # Check that no exception is raised when the process is fast
            securedrop_admin.get_release_key_from_keyserver(args)

            # Check that use of the keyword arg also raises no exception
            securedrop_admin.get_release_key_from_keyserver(args, keyserver="test.com")

    @pytest.mark.parametrize(
        "git_output",
        [
            b"gpg: Signature made Thu 20 Jul "
            b"2022 08:12:25 PM EDT\n"
            b"gpg:                using RSA key "
            b"2359E6538C0613E652955E6C188EDD3B7B22E6A3\n"
            b'gpg: Good signature from "SecureDrop Release '
            b"Signing Key "
            b'<securedrop-release-key-2021@freedom.press>" [unknown]\n',
        ],
    )
    def test_update_signature_verifies(self, tmpdir, caplog, git_output):
        git_repo_path = str(tmpdir)
        args = argparse.Namespace(root=git_repo_path)
        patchers = [
            mock.patch("securedrop_admin.check_for_updates", return_value=(True, "0.6.1")),
            mock.patch("subprocess.check_call"),
            # securedrop-admin checks if there is a branch with the same name as a tag
            # that is being verified, and bails if there is. To ensure the verification
            # succeeds, we have to mock the "not a valid ref" output it looks for.
            mock.patch(
                "subprocess.check_output",
                side_effect=[
                    git_output,
                    subprocess.CalledProcessError(1, "cmd", b"not a valid ref"),
                ],
            ),
        ]

        for patcher in patchers:
            patcher.start()

        try:
            ret_code = securedrop_admin.update(args)
            assert "Applying SecureDrop updates..." in caplog.text
            assert "Signature verification successful." in caplog.text
            assert "Updated to SecureDrop" in caplog.text
            assert ret_code == 0
        finally:
            for patcher in patchers:
                patcher.stop()

    def test_update_unexpected_exception_git_refs(self, tmpdir, caplog):
        git_repo_path = str(tmpdir)
        args = argparse.Namespace(root=git_repo_path)

        git_output = (
            b"gpg: Signature made Tue 13 Mar 2022 01:14:11 AM UTC\n"
            b"gpg:                using RSA key "
            b"2359E6538C0613E652955E6C188EDD3B7B22E6A3\n"
            b'gpg: Good signature from "SecureDrop Release '
            b'Signing Key <securedrop-release-key-2021@freedom.press>" [unknown]\n'
        )

        patchers = [
            mock.patch("securedrop_admin.check_for_updates", return_value=(True, "0.6.1")),
            mock.patch("subprocess.check_call"),
            mock.patch(
                "subprocess.check_output",
                side_effect=[
                    git_output,
                    subprocess.CalledProcessError(1, "cmd", b"a random error"),
                ],
            ),
        ]

        for patcher in patchers:
            patcher.start()

        try:
            ret_code = securedrop_admin.update(args)
            assert "Applying SecureDrop updates..." in caplog.text
            assert "Signature verification successful." not in caplog.text
            assert "Updated to SecureDrop" not in caplog.text
            assert ret_code == 1
        finally:
            for patcher in patchers:
                patcher.stop()

    def test_outdated_signature_does_not_verify(self, tmpdir, caplog):
        """
        When a tag is signed with a release key that is no longer valid
            Then the signature of a current tag should not verify
        """

        git_repo_path = str(tmpdir)
        args = argparse.Namespace(root=git_repo_path)

        git_output = (
            b"gpg: Signature made Tue 13 Mar 2022 01:14:11 AM UTC\n"
            b"gpg:                using RSA key "
            b"22245C81E3BAEB4138B36061310F561200F4AD77\n"
            b'gpg: Good signature from "SecureDrop Release '
            b'Signing Key" [unknown]\n'
        )

        patchers = [
            mock.patch("securedrop_admin.check_for_updates", return_value=(True, "0.6.1")),
            mock.patch("subprocess.check_call"),
            mock.patch(
                "subprocess.check_output",
                side_effect=[
                    git_output,
                    subprocess.CalledProcessError(1, "cmd", b"not a valid ref"),
                ],
            ),
        ]

        for patcher in patchers:
            patcher.start()

        try:
            ret_code = securedrop_admin.update(args)
            assert "Applying SecureDrop updates..." in caplog.text
            assert "Signature verification successful." not in caplog.text
            assert "Updated to SecureDrop" not in caplog.text
            assert ret_code == 1
        finally:
            for patcher in patchers:
                patcher.stop()

    def test_update_signature_does_not_verify(self, tmpdir, caplog):
        git_repo_path = str(tmpdir)
        args = argparse.Namespace(root=git_repo_path)

        git_output = (
            b"gpg: Signature made Tue 13 Mar 2022 01:14:11 AM UTC\n"
            b"gpg:                using RSA key "
            b"2359E6538C0613E652955E6C188EDD3B7B22E6A3\n"
            b'gpg: BAD signature from "SecureDrop Release '
            b'Signing Key <securedrop-release-key-2021@freedom.press>" [unknown]\n'
        )

        with mock.patch("securedrop_admin.check_for_updates", return_value=(True, "0.6.1")):
            with mock.patch("subprocess.check_call"):
                with mock.patch("subprocess.check_output", return_value=git_output):
                    ret_code = securedrop_admin.update(args)
                    assert "Applying SecureDrop updates..." in caplog.text
                    assert "Update failed: Invalid signature format" in caplog.text
                    assert "Updated to SecureDrop" not in caplog.text
                    assert ret_code != 0

    def test_update_malicious_key_named_fingerprint(self, tmpdir, caplog):
        git_repo_path = str(tmpdir)
        args = argparse.Namespace(root=git_repo_path)

        git_output = (
            b"gpg: Signature made Tue 13 Mar 2022 01:14:11 AM UTC\n"
            b"gpg:                using RSA key "
            b"1234567812345678123456781234567812345678\n"
            b'gpg: Good signature from "2359E6538C0613E652'
            b'955E6C188EDD3B7B22E6A3" [unknown]\n'
        )

        with mock.patch("securedrop_admin.check_for_updates", return_value=(True, "0.6.1")):
            with mock.patch("subprocess.check_call"):
                with mock.patch("subprocess.check_output", return_value=git_output):
                    ret_code = securedrop_admin.update(args)
                    assert "Applying SecureDrop updates..." in caplog.text
                    assert "Update failed: Invalid signature format" in caplog.text
                    assert "Updated to SecureDrop" not in caplog.text
                    assert ret_code != 0

    def test_update_malicious_key_named_good_sig(self, tmpdir, caplog):
        git_repo_path = str(tmpdir)
        args = argparse.Namespace(root=git_repo_path)

        git_output = (
            b"gpg: Signature made Tue 13 Mar 2022 01:14:11 AM UTC\n"
            b"gpg:                using RSA key "
            b"1234567812345678123456781234567812345678\n"
            b"gpg: Good signature from Good signature from "
            b'"SecureDrop Release Signing Key <securedrop-release-key-2021@freedom.press>" '
            b"[unknown]\n"
        )

        with mock.patch("securedrop_admin.check_for_updates", return_value=(True, "0.6.1")):
            with mock.patch("subprocess.check_call"):
                with mock.patch("subprocess.check_output", return_value=git_output):
                    ret_code = securedrop_admin.update(args)
                    assert "Applying SecureDrop updates..." in caplog.text
                    assert "Update failed: Invalid signature format" in caplog.text
                    assert "Updated to SecureDrop" not in caplog.text
                    assert ret_code != 0

    def test_update_malicious_key_named_good_sig_fingerprint(self, tmpdir, caplog):
        git_repo_path = str(tmpdir)
        args = argparse.Namespace(root=git_repo_path)

        git_output = (
            b"gpg: Signature made Tue 13 Mar 2022 01:14:11 AM UTC\n"
            b"gpg:                using RSA key "
            b"1234567812345678123456781234567812345678\n"
            b"gpg: Good signature from 22245C81E3BAEB4138"
            b"955E6C188EDD3B7B22E6A3 Good signature from "
            b'"SecureDrop Release Signing Key <securedrop-release-key-2021@freedom.press>" '
            b"[unknown]\n"
        )

        with mock.patch("securedrop_admin.check_for_updates", return_value=(True, "0.6.1")):
            with mock.patch("subprocess.check_call"):
                with mock.patch("subprocess.check_output", return_value=git_output):
                    ret_code = securedrop_admin.update(args)
                    assert "Applying SecureDrop updates..." in caplog.text
                    assert "Update failed: Invalid signature format" in caplog.text
                    assert "Updated to SecureDrop" not in caplog.text
                    assert ret_code != 0

    def test_no_signature_on_update(self, tmpdir, caplog):
        git_repo_path = str(tmpdir)
        args = argparse.Namespace(root=git_repo_path)

        with mock.patch("securedrop_admin.check_for_updates", return_value=(True, "0.6.1")):
            with mock.patch("subprocess.check_call"):
                with mock.patch(
                    "subprocess.check_output",
                    side_effect=subprocess.CalledProcessError(
                        1, "git", "error: no signature found"
                    ),
                ):
                    ret_code = securedrop_admin.update(args)
                    assert "Applying SecureDrop updates..." in caplog.text
                    assert "Update failed: Missing or invalid signature" in caplog.text
                    assert "Updated to SecureDrop" not in caplog.text
                    assert ret_code != 0

    def test_exit_codes(self, tmpdir):
        """Ensure that securedrop-admin returns the correct
        exit codes for success or failure."""
        with mock.patch("securedrop_admin.install_securedrop", return_value=0):
            with pytest.raises(SystemExit) as e:
                securedrop_admin.main(["--root", str(tmpdir), "install"])
            assert e.value.code == securedrop_admin.EXIT_SUCCESS

        with mock.patch(
            "securedrop_admin.install_securedrop",
            side_effect=subprocess.CalledProcessError(1, "TestError"),
        ):
            with pytest.raises(SystemExit) as e:
                securedrop_admin.main(["--root", str(tmpdir), "install"])
            assert e.value.code == securedrop_admin.EXIT_SUBPROCESS_ERROR

        with mock.patch("securedrop_admin.install_securedrop", side_effect=KeyboardInterrupt):
            with pytest.raises(SystemExit) as e:
                securedrop_admin.main(["--root", str(tmpdir), "install"])
            assert e.value.code == securedrop_admin.EXIT_INTERRUPT


class TestSiteConfig:
    def test_exists(self, tmpdir):
        args = argparse.Namespace(
            site_config="DOES_NOT_EXIST",
            ansible_path=".",
            app_path=dirname(__file__),
            root=tmpdir,
        )
        assert not securedrop_admin.SiteConfig(args).exists()
        args = argparse.Namespace(
            site_config=__file__, ansible_path=".", app_path=dirname(__file__), root=tmpdir
        )
        assert securedrop_admin.SiteConfig(args).exists()

    def test_validate_not_empty(self):
        validator = securedrop_admin.SiteConfig.ValidateNotEmpty()

        assert validator.validate(Document("something"))
        with pytest.raises(ValidationError):
            validator.validate(Document(""))

    def test_validate_time(self):
        validator = securedrop_admin.SiteConfig.ValidateTime()

        assert validator.validate(Document("4"))
        with pytest.raises(ValidationError):
            validator.validate(Document(""))
        with pytest.raises(ValidationError):
            validator.validate(Document("four"))
        with pytest.raises(ValidationError):
            validator.validate(Document("4.30"))
        with pytest.raises(ValidationError):
            validator.validate(Document("25"))
        with pytest.raises(ValidationError):
            validator.validate(Document("-4"))

    def test_validate_ossec_username(self):
        validator = securedrop_admin.SiteConfig.ValidateOSSECUsername()

        assert validator.validate(Document("username"))
        with pytest.raises(ValidationError):
            validator.validate(Document("bad@user"))
        with pytest.raises(ValidationError):
            validator.validate(Document("test"))

    def test_validate_ossec_password(self):
        validator = securedrop_admin.SiteConfig.ValidateOSSECPassword()

        assert validator.validate(Document("goodpassword"))
        with pytest.raises(ValidationError):
            validator.validate(Document("password123"))
        with pytest.raises(ValidationError):
            validator.validate(Document(""))
        with pytest.raises(ValidationError):
            validator.validate(Document("short"))

    def test_validate_email(self):
        validator = securedrop_admin.SiteConfig.ValidateEmail()

        assert validator.validate(Document("good@mail.com"))
        with pytest.raises(ValidationError):
            validator.validate(Document("badmail"))
        with pytest.raises(ValidationError):
            validator.validate(Document(""))

    def test_validate_ossec_email(self):
        validator = securedrop_admin.SiteConfig.ValidateOSSECEmail()

        assert validator.validate(Document("good@mail.com"))
        with pytest.raises(ValidationError) as e:
            validator.validate(Document("ossec@ossec.test"))
        assert "something other than ossec@ossec.test" in str(e)

    def test_validate_optional_email(self):
        validator = securedrop_admin.SiteConfig.ValidateOptionalEmail()

        assert validator.validate(Document("good@mail.com"))
        assert validator.validate(Document(""))

    def test_validate_user(self):
        validator = securedrop_admin.SiteConfig.ValidateUser()
        with pytest.raises(ValidationError):
            validator.validate(Document("amnesia"))
        with pytest.raises(ValidationError):
            validator.validate(Document("root"))
        with pytest.raises(ValidationError):
            validator.validate(Document(""))
        assert validator.validate(Document("gooduser"))

    def test_validate_ip(self):
        validator = securedrop_admin.SiteConfig.ValidateIP()
        with pytest.raises(ValidationError):
            validator.validate(Document("599.20"))
        assert validator.validate(Document("192.168.1.1"))

    def test_validate_path(self):
        mydir = dirname(__file__)
        myfile = basename(__file__)
        validator = securedrop_admin.SiteConfig.ValidatePath(mydir)
        assert validator.validate(Document(myfile))
        with pytest.raises(ValidationError):
            validator.validate(Document("NONEXIST"))
        with pytest.raises(ValidationError):
            validator.validate(Document(""))

    def test_validate_optional_path(self):
        mydir = dirname(__file__)
        myfile = basename(__file__)
        validator = securedrop_admin.SiteConfig.ValidateOptionalPath(mydir)
        assert validator.validate(Document(myfile))
        assert validator.validate(Document(""))

    def test_validate_yes_no(self):
        validator = securedrop_admin.SiteConfig.ValidateYesNo()
        with pytest.raises(ValidationError):
            validator.validate(Document("something"))
        assert validator.validate(Document("yes"))
        assert validator.validate(Document("YES"))
        assert validator.validate(Document("no"))
        assert validator.validate(Document("NO"))

    def test_validate_fingerprint(self):
        validator = securedrop_admin.SiteConfig.ValidateFingerprint()
        assert validator.validate(Document("012345678901234567890123456789ABCDEFABCD"))
        assert validator.validate(Document("01234 5678901234567890123456789ABCDE   FABCD"))

        with pytest.raises(ValidationError) as e:
            validator.validate(Document("65A1B5FF195B56353CC63DFFCC40EF1228271441"))
        assert "TEST journalist" in str(e)

        with pytest.raises(ValidationError) as e:
            validator.validate(Document("600BC6D5142C68F35DDBCEA87B597104EDDDC102"))
        assert "TEST admin" in str(e)

        with pytest.raises(ValidationError) as e:
            validator.validate(Document("0000"))
        assert "40 hexadecimal" in str(e)

        with pytest.raises(ValidationError) as e:
            validator.validate(Document("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz"))
        assert "40 hexadecimal" in str(e)

    def test_validate_optional_fingerprint(self):
        validator = securedrop_admin.SiteConfig.ValidateOptionalFingerprint()
        assert validator.validate(Document("012345678901234567890123456789ABCDEFABCD"))
        assert validator.validate(Document(""))

    def test_sanitize_fingerprint(self, tmpdir):
        args = argparse.Namespace(
            site_config="DOES_NOT_EXIST",
            ansible_path=".",
            app_path=dirname(__file__),
            root=tmpdir,
        )
        site_config = securedrop_admin.SiteConfig(args)
        assert site_config.sanitize_fingerprint("    A bc\n") == "ABC"

    def test_validate_int(self):
        validator = securedrop_admin.SiteConfig.ValidateInt()
        with pytest.raises(ValidationError):
            validator.validate(Document("123X"))
        assert validator.validate(Document("192"))

    def test_locales(self):
        locales = securedrop_admin.SiteConfig.Locales(dirname(__file__))
        translations = locales.get_translations()
        assert "en_US" in translations
        assert "fr_FR" in translations

    def test_validate_locales(self):
        validator = securedrop_admin.SiteConfig.ValidateLocales(
            dirname(__file__), {"en_US", "fr_FR"}
        )
        assert validator.validate(Document("en_US  fr_FR "))
        with pytest.raises(ValidationError) as e:
            validator.validate(Document("BAD"))
        assert "BAD" in str(e)

    def test_save(self, tmpdir):
        site_config_path = join(str(tmpdir), "site_config")
        args = argparse.Namespace(
            site_config=site_config_path,
            ansible_path=".",
            app_path=dirname(__file__),
            root=tmpdir,
        )
        site_config = securedrop_admin.SiteConfig(args)
        site_config.config = {"var1": "val1", "var2": "val2"}
        site_config.save()
        expected = textwrap.dedent(
            """\
        var1: val1
        var2: val2
        """
        )
        assert expected == open(site_config_path).read()

    def test_validate_gpg_key(self, tmpdir, caplog):
        args = argparse.Namespace(
            site_config="INVALID",
            ansible_path="tests/files",
            app_path=dirname(__file__),
            root=tmpdir,
        )
        good_config = {
            "securedrop_app_gpg_public_key": "test_journalist_key.pub",
            "securedrop_app_gpg_fingerprint": "65A1B5FF195B56353CC63DFFCC40EF1228271441",
            "ossec_alert_gpg_public_key": "test_journalist_key.pub",
            "ossec_gpg_fpr": "65A1B5FF195B56353CC63DFFCC40EF1228271441",
            "journalist_alert_gpg_public_key": "test_journalist_key.pub",
            "journalist_gpg_fpr": "65A1B5FF195B56353CC63DFFCC40EF1228271441",
        }
        site_config = securedrop_admin.SiteConfig(args)
        site_config.config = good_config
        assert site_config.validate_gpg_keys()

        for key in ("securedrop_app_gpg_fingerprint", "ossec_gpg_fpr", "journalist_gpg_fpr"):
            bad_config = good_config.copy()
            bad_config[key] = "FAIL"
            site_config.config = bad_config
            with pytest.raises(securedrop_admin.FingerprintException) as e:
                site_config.validate_gpg_keys()
            assert "FAIL does not match" in str(e)

        # Test a key with matching fingerprint but that fails sq-keyring-linter
        invalid_config = {
            # Correct key fingerprint but weak 1024-bit RSA key with SHA-1 self signature
            "securedrop_app_gpg_public_key": "weak_test_key_should_fail_sqlinter.asc",
            "securedrop_app_gpg_fingerprint": "40F1C17B7E7826DAB40B14AE7786B000E6D0A76E",
            "ossec_alert_gpg_public_key": "test_journalist_key.pub",
            "ossec_gpg_fpr": "65A1B5FF195B56353CC63DFFCC40EF1228271441",
            "journalist_alert_gpg_public_key": "test_journalist_key.pub",
            "journalist_gpg_fpr": "65A1B5FF195B56353CC63DFFCC40EF1228271441",
        }
        site_config.config = invalid_config
        with pytest.raises(securedrop_admin.FingerprintException) as e:
            site_config.validate_gpg_keys()
        assert "failed sq-keyring-linter check" in str(e)

    def test_journalist_alert_email(self, tmpdir):
        args = argparse.Namespace(
            site_config="INVALID",
            ansible_path="tests/files",
            app_path=dirname(__file__),
            root=tmpdir,
        )
        site_config = securedrop_admin.SiteConfig(args)
        site_config.config = {
            "journalist_alert_gpg_public_key": "",
            "journalist_gpg_fpr": "",
        }
        assert site_config.validate_journalist_alert_email()
        site_config.config = {
            "journalist_alert_gpg_public_key": "test_journalist_key.pub",
            "journalist_gpg_fpr": "65A1B5FF195B56353CC63DFFCC40EF1228271441",
        }
        site_config.config["journalist_alert_email"] = ""
        with pytest.raises(securedrop_admin.JournalistAlertEmailException) as e:
            site_config.validate_journalist_alert_email()
        assert "not be empty" in str(e)

        site_config.config["journalist_alert_email"] = "bademail"
        with pytest.raises(securedrop_admin.JournalistAlertEmailException) as e:
            site_config.validate_journalist_alert_email()
        assert "Must contain a @" in str(e)

        site_config.config["journalist_alert_email"] = "good@email.com"
        assert site_config.validate_journalist_alert_email()

    @mock.patch("securedrop_admin.SiteConfig.validated_input", side_effect=lambda p, d, v, t: d)
    @mock.patch("securedrop_admin.SiteConfig.save")
    def test_update_config(self, mock_save, mock_validate_input):
        args = argparse.Namespace(
            site_config="tests/files/site-specific",
            ansible_path="tests/files",
            app_path=dirname(__file__),
            root="tests/files",
        )
        site_config = securedrop_admin.SiteConfig(args)

        assert site_config.load_and_update_config()
        assert "user_defined_variable" in site_config.config
        mock_save.assert_called_once()
        mock_validate_input.assert_called()

    @mock.patch("securedrop_admin.SiteConfig.validated_input", side_effect=lambda p, d, v, t: d)
    @mock.patch("securedrop_admin.SiteConfig.validate_gpg_keys")
    def test_update_config_no_site_specific(self, validate_gpg_keys, mock_validate_input, tmpdir):
        site_config_path = join(str(tmpdir), "site_config")
        args = argparse.Namespace(
            site_config=site_config_path,
            ansible_path=".",
            app_path=dirname(__file__),
            root=tmpdir,
        )
        site_config = securedrop_admin.SiteConfig(args)
        assert site_config.load_and_update_config()
        mock_validate_input.assert_called()
        validate_gpg_keys.assert_called_once()
        assert exists(site_config_path)

    def test_load_and_update_config(self, tmpdir):
        args = argparse.Namespace(
            site_config="tests/files/site-specific",
            ansible_path="tests/files",
            app_path=dirname(__file__),
            root=tmpdir,
        )
        site_config = securedrop_admin.SiteConfig(args)
        with mock.patch("securedrop_admin.SiteConfig.update_config"):
            site_config.load_and_update_config()
            assert site_config.config != {}

        args = argparse.Namespace(
            site_config="tests/files/site-specific-missing-entries",
            ansible_path="tests/files",
            app_path=dirname(__file__),
            root=tmpdir,
        )
        site_config = securedrop_admin.SiteConfig(args)
        with mock.patch("securedrop_admin.SiteConfig.update_config"):
            site_config.load_and_update_config()
            assert site_config.config != {}

        args = argparse.Namespace(
            site_config="UNKNOWN",
            ansible_path="tests/files",
            app_path=dirname(__file__),
            root=tmpdir,
        )
        site_config = securedrop_admin.SiteConfig(args)
        with mock.patch("securedrop_admin.SiteConfig.update_config"):
            site_config.load_and_update_config()
            assert site_config.config == {}

    def get_desc(self, site_config, var):
        for desc in site_config.desc:
            if desc[0] == var:
                return desc

    def verify_desc_consistency_optional(self, site_config, desc):
        (var, default, etype, prompt, validator, transform, condition) = desc
        # verify the default passes validation
        if callable(default):
            default = default()
        assert site_config.user_prompt_config_one(desc, None) == default
        assert type(default) is etype

    def verify_desc_consistency(self, site_config, desc):
        self.verify_desc_consistency_optional(site_config, desc)

    def verify_prompt_boolean(self, site_config, desc):
        self.verify_desc_consistency(site_config, desc)
        (var, default, etype, prompt, validator, transform, condition) = desc
        assert site_config.user_prompt_config_one(desc, True) is True
        assert site_config.user_prompt_config_one(desc, False) is False
        assert site_config.user_prompt_config_one(desc, "YES") is True
        assert site_config.user_prompt_config_one(desc, "NO") is False

    def test_desc_conditional(self, tmpdir):
        """Ensure that conditional prompts behave correctly.

        Prompts which depend on another question should only be
        asked if the prior question was answered appropriately."""

        questions = [
            (
                "first_question",
                False,
                bool,
                "Test Question 1",
                None,
                lambda x: x.lower() == "yes",
                lambda config: True,
            ),
            (
                "dependent_question",
                "default_value",
                str,
                "Test Question 2",
                None,
                None,
                lambda config: config.get("first_question", False),
            ),
        ]
        args = argparse.Namespace(
            site_config="tests/files/site-specific",
            ansible_path="tests/files",
            app_path=dirname(__file__),
            root=tmpdir,
        )
        site_config = securedrop_admin.SiteConfig(args)
        site_config.desc = questions

        def auto_prompt(prompt, default, **kwargs):
            return default

        with mock.patch("prompt_toolkit.prompt", side_effect=auto_prompt):
            config = site_config.user_prompt_config()
            assert config["dependent_question"] != "default_value"

            edited_first_question = list(site_config.desc[0])
            edited_first_question[1] = True
            site_config.desc[0] = tuple(edited_first_question)

            config = site_config.user_prompt_config()
            assert config["dependent_question"] == "default_value"

    verify_prompt_ssh_users = verify_desc_consistency
    verify_prompt_app_ip = verify_desc_consistency
    verify_prompt_monitor_ip = verify_desc_consistency
    verify_prompt_app_hostname = verify_desc_consistency
    verify_prompt_monitor_hostname = verify_desc_consistency
    verify_prompt_dns_server = verify_desc_consistency

    verify_prompt_securedrop_app_pow_on_source_interface = verify_prompt_boolean
    verify_prompt_securedrop_app_https_on_source_interface = verify_prompt_boolean
    verify_prompt_enable_ssh_over_tor = verify_prompt_boolean

    verify_prompt_securedrop_app_gpg_public_key = verify_desc_consistency

    def verify_prompt_not_empty(self, site_config, desc):
        with pytest.raises(ValidationError):
            site_config.user_prompt_config_one(desc, "")

    def verify_prompt_fingerprint_optional(self, site_config, desc):
        fpr = "0123456 789012 34567890123456789ABCDEFABCD"
        clean_fpr = site_config.sanitize_fingerprint(fpr)
        assert site_config.user_prompt_config_one(desc, fpr) == clean_fpr

    def verify_desc_consistency_allow_empty(self, site_config, desc):
        (var, default, etype, prompt, validator, transform, condition) = desc
        # verify the default passes validation
        assert site_config.user_prompt_config_one(desc, None) == default
        assert type(default) is etype

    def verify_prompt_fingerprint(self, site_config, desc):
        self.verify_prompt_not_empty(site_config, desc)
        self.verify_prompt_fingerprint_optional(site_config, desc)

    verify_prompt_securedrop_app_gpg_fingerprint = verify_prompt_fingerprint
    verify_prompt_ossec_alert_gpg_public_key = verify_desc_consistency
    verify_prompt_ossec_gpg_fpr = verify_prompt_fingerprint
    verify_prompt_ossec_alert_email = verify_prompt_not_empty
    verify_prompt_journalist_alert_gpg_public_key = verify_desc_consistency_optional
    verify_prompt_journalist_gpg_fpr = verify_prompt_fingerprint_optional
    verify_prompt_journalist_alert_email = verify_desc_consistency_optional
    verify_prompt_securedrop_app_https_certificate_chain_src = verify_desc_consistency_optional
    verify_prompt_securedrop_app_https_certificate_key_src = verify_desc_consistency_optional
    verify_prompt_securedrop_app_https_certificate_cert_src = verify_desc_consistency_optional
    verify_prompt_smtp_relay = verify_prompt_not_empty
    verify_prompt_smtp_relay_port = verify_desc_consistency
    verify_prompt_daily_reboot_time = verify_desc_consistency
    verify_prompt_sasl_domain = verify_desc_consistency_allow_empty
    verify_prompt_sasl_username = verify_prompt_not_empty
    verify_prompt_sasl_password = verify_prompt_not_empty

    def verify_prompt_securedrop_supported_locales(self, site_config, desc):
        (var, default, etype, prompt, validator, transform, condition) = desc
        # verify the default passes validation
        assert site_config.user_prompt_config_one(desc, None) == default
        assert type(default) is etype
        assert site_config.user_prompt_config_one(desc, "fr_FR en_US") == ["fr_FR", "en_US"]
        assert site_config.user_prompt_config_one(desc, ["fr_FR", "en_US"]) == ["fr_FR", "en_US"]
        assert site_config.user_prompt_config_one(desc, "") == []
        with pytest.raises(ValidationError):
            site_config.user_prompt_config_one(desc, "wrong")

    def test_user_prompt_config_one(self, tmpdir):
        args = argparse.Namespace(
            site_config="UNKNOWN",
            ansible_path="tests/files",
            app_path=dirname(__file__),
            root=tmpdir,
        )
        site_config = securedrop_admin.SiteConfig(args)

        def auto_prompt(prompt, default, **kwargs):
            if "validator" in kwargs and kwargs["validator"]:
                assert kwargs["validator"].validate(Document(default))
            return default

        with mock.patch("prompt_toolkit.prompt", side_effect=auto_prompt):
            for desc in site_config.desc:
                (var, default, etype, prompt, validator, transform, condition) = desc
                method = "verify_prompt_" + var
                print("checking " + method)
                getattr(self, method)(site_config, desc)

    def test_validated_input(self, tmpdir):
        args = argparse.Namespace(
            site_config="UNKNOWN",
            ansible_path="tests/files",
            app_path=dirname(__file__),
            root=tmpdir,
        )
        site_config = securedrop_admin.SiteConfig(args)

        def auto_prompt(prompt, default, **kwargs):
            return default

        with mock.patch("prompt_toolkit.prompt", side_effect=auto_prompt):
            value = "VALUE"
            assert value == site_config.validated_input("", value, lambda: True, None)
            assert value.lower() == site_config.validated_input("", value, lambda: True, str.lower)
            assert site_config.validated_input("", True, lambda: True, None) == "yes"
            assert site_config.validated_input("", False, lambda: True, None) == "no"
            assert site_config.validated_input("", 1234, lambda: True, None) == "1234"
            assert site_config.validated_input("", ["a", "b"], lambda: True, None) == "a b"
            assert site_config.validated_input("", {}, lambda: True, None) == "{}"

    def test_load(self, tmpdir, caplog):
        args = argparse.Namespace(
            site_config="tests/files/site-specific",
            ansible_path="tests/files",
            app_path=dirname(__file__),
            root=tmpdir,
        )
        site_config = securedrop_admin.SiteConfig(args)
        assert "app_hostname" in site_config.load()

        args = argparse.Namespace(
            site_config="UNKNOWN",
            ansible_path="tests/files",
            app_path=dirname(__file__),
            root=tmpdir,
        )
        site_config = securedrop_admin.SiteConfig(args)
        with pytest.raises(IOError) as e:
            site_config.load()
        assert "No such file" in e.value.strerror
        assert "Config file missing" in caplog.text

        args = argparse.Namespace(
            site_config="tests/files/corrupted",
            ansible_path="tests/files",
            app_path=dirname(__file__),
            root=tmpdir,
        )
        site_config = securedrop_admin.SiteConfig(args)
        with pytest.raises(yaml.YAMLError) as e:
            site_config.load()
        assert "issue processing" in caplog.text


def test_generate_new_v3_keys():
    public, private = securedrop_admin.generate_new_v3_keys()

    for key in [public, private]:
        # base32 padding characters should be removed
        assert "=" not in key
        assert len(key) == 52


def test_find_or_generate_new_torv3_keys_first_run(tmpdir, capsys):
    args = argparse.Namespace(ansible_path=str(tmpdir))

    return_code = securedrop_admin.find_or_generate_new_torv3_keys(args)

    out, err = capsys.readouterr()
    assert "Tor v3 onion service keys generated" in out
    assert return_code == 0

    secret_key_path = os.path.join(args.ansible_path, "tor_v3_keys.json")

    with open(secret_key_path) as f:
        v3_onion_service_keys = json.load(f)

    expected_keys = [
        "app_journalist_public_key",
        "app_journalist_private_key",
        "app_ssh_public_key",
        "app_ssh_private_key",
        "mon_ssh_public_key",
        "mon_ssh_private_key",
    ]
    for key in expected_keys:
        assert key in v3_onion_service_keys


def test_find_or_generate_new_torv3_keys_subsequent_run(tmpdir, capsys):
    args = argparse.Namespace(ansible_path=str(tmpdir))

    secret_key_path = os.path.join(args.ansible_path, "tor_v3_keys.json")
    old_keys = {"foo": "bar"}
    with open(secret_key_path, "w") as f:
        json.dump(old_keys, f)

    return_code = securedrop_admin.find_or_generate_new_torv3_keys(args)

    out, err = capsys.readouterr()
    assert "Tor v3 onion service keys already exist" in out
    assert return_code == 0

    with open(secret_key_path) as f:
        v3_onion_service_keys = json.load(f)

    assert v3_onion_service_keys == old_keys
