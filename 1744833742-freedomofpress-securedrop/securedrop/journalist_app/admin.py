import binascii
import os
from datetime import datetime
from typing import Optional, Union

import two_factor
import werkzeug
from db import db
from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    g,
    redirect,
    render_template,
    request,
    url_for,
)
from flask_babel import gettext
from journalist_app.decorators import admin_required
from journalist_app.forms import LogoForm, NewUserForm, OrgNameForm, SubmissionPreferencesForm
from journalist_app.sessions import session
from journalist_app.utils import (
    commit_account_changes,
    set_diceware_password,
    set_pending_password,
    validate_hotp_secret,
    verify_pending_password,
)
from markupsafe import Markup
from models import (
    FirstOrLastNameError,
    InstanceConfig,
    InvalidUsernameException,
    Journalist,
    PasswordError,
    Submission,
)
from passphrases import PassphraseGenerator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound


def make_blueprint() -> Blueprint:
    view = Blueprint("admin", __name__)

    @view.route("/", methods=("GET", "POST"))
    @admin_required
    def index() -> str:
        users = Journalist.query.filter(Journalist.username != "deleted").all()
        return render_template("admin.html", users=users)

    @view.route("/config", methods=("GET", "POST"))
    @admin_required
    def manage_config() -> Union[str, werkzeug.Response]:
        if InstanceConfig.get_default().initial_message_min_len > 0:
            prevent_short_messages = True
        else:
            prevent_short_messages = False

        # The UI document upload prompt ("prevent") is the opposite of the setting ("allow")
        submission_preferences_form = SubmissionPreferencesForm(
            prevent_document_uploads=not InstanceConfig.get_default().allow_document_uploads,
            prevent_short_messages=prevent_short_messages,
            min_message_length=InstanceConfig.get_default().initial_message_min_len,
            reject_codename_messages=InstanceConfig.get_default().reject_message_with_codename,
        )
        organization_name_form = OrgNameForm(
            organization_name=InstanceConfig.get_default().organization_name
        )
        logo_form = LogoForm()
        if logo_form.validate_on_submit():
            f = logo_form.logo.data

            if current_app.static_folder is None:
                abort(500)
            custom_logo_filepath = os.path.join(current_app.static_folder, "i", "custom_logo.png")
            try:
                f.save(custom_logo_filepath)
                flash(gettext("Image updated."), "logo-success")
            except Exception:
                flash(
                    # Translators: This error is shown when an uploaded image cannot be used.
                    gettext("Unable to process the image file. Please try another one."),
                    "logo-error",
                )
            return redirect(url_for("admin.manage_config") + "#config-logoimage")
        else:
            for errors in logo_form.errors.values():
                for error in errors:
                    flash(error, "logo-error")
            return render_template(
                "config.html",
                submission_preferences_form=submission_preferences_form,
                organization_name_form=organization_name_form,
                max_len=Submission.MAX_MESSAGE_LEN,
                logo_form=logo_form,
            )

    @view.route("/update-submission-preferences", methods=["POST"])
    @admin_required
    def update_submission_preferences() -> Optional[werkzeug.Response]:
        form = SubmissionPreferencesForm()
        if form.validate_on_submit():
            # The UI prompt ("prevent") is the opposite of the setting ("allow"):
            allow_uploads = not form.prevent_document_uploads.data

            if form.prevent_short_messages.data:
                msg_length = form.min_message_length.data
            else:
                msg_length = 0

            reject_codenames = form.reject_codename_messages.data

            InstanceConfig.update_submission_prefs(allow_uploads, msg_length, reject_codenames)
            flash(gettext("Preferences saved."), "submission-preferences-success")
            return redirect(url_for("admin.manage_config") + "#config-preventuploads")
        else:
            for errors in list(form.errors.values()):
                for error in errors:
                    flash(
                        gettext("Preferences not updated.") + " " + error,
                        "submission-preferences-error",
                    )
        return redirect(url_for("admin.manage_config") + "#config-preventuploads")

    @view.route("/update-org-name", methods=["POST"])
    @admin_required
    def update_org_name() -> Union[str, werkzeug.Response]:
        form = OrgNameForm()
        if form.validate_on_submit():
            try:
                value = request.form["organization_name"]
                InstanceConfig.set_organization_name(value)
                flash(gettext("Preferences saved."), "org-name-success")
            except Exception:
                flash(gettext("Failed to update organization name."), "org-name-error")
            return redirect(url_for("admin.manage_config") + "#config-orgname")
        else:
            for errors in list(form.errors.values()):
                for error in errors:
                    flash(error, "org-name-error")
        return redirect(url_for("admin.manage_config") + "#config-orgname")

    @view.route("/add", methods=("GET", "POST"))
    @admin_required
    def add_user() -> Union[str, werkzeug.Response]:
        form = NewUserForm()
        if form.validate_on_submit():
            form_valid = True
            username = request.form["username"]
            first_name = request.form["first_name"]
            last_name = request.form["last_name"]
            password = request.form["password"]
            is_admin = bool(request.form.get("is_admin"))

            try:
                otp_secret = None
                if request.form.get("is_hotp", False):
                    otp_secret = request.form.get("otp_secret", "")
                verify_pending_password(for_="new", passphrase=password)
                new_user = Journalist(
                    username=username,
                    password=password,
                    first_name=first_name,
                    last_name=last_name,
                    is_admin=is_admin,
                    otp_secret=otp_secret,
                )
                db.session.add(new_user)
                db.session.commit()
            except PasswordError:
                flash(
                    gettext(
                        "There was an error with the autogenerated password. "
                        "User not created. Please try again."
                    ),
                    "error",
                )
                form_valid = False
            except (binascii.Error, TypeError) as e:
                if "Non-hexadecimal digit found" in str(e):
                    flash(
                        gettext(
                            "Invalid HOTP secret format: "
                            "please only submit letters A-F and numbers 0-9."
                        ),
                        "error",
                    )
                else:
                    flash(
                        gettext("An unexpected error occurred! " "Please inform your admin."),
                        "error",
                    )
                form_valid = False
            except InvalidUsernameException as e:
                form_valid = False
                # Translators: Here, "{message}" explains the problem with the username.
                flash(gettext("Invalid username: {message}").format(message=e), "error")
            except IntegrityError as e:
                db.session.rollback()
                form_valid = False
                if "UNIQUE constraint failed: journalists.username" in str(e):
                    flash(
                        gettext('Username "{username}" already taken.').format(username=username),
                        "error",
                    )
                else:
                    flash(
                        gettext(
                            "An error occurred saving this user"
                            " to the database."
                            " Please inform your admin."
                        ),
                        "error",
                    )
                    current_app.logger.error("Adding user " f"'{username}' failed: {e}")

            if form_valid:
                if new_user.is_totp:
                    return render_template(
                        "admin_new_user_two_factor_totp.html",
                        qrcode=Markup(new_user.totp.qrcode_svg(new_user.username).decode()),
                        otp_secret=new_user.otp_secret,
                        formatted_otp_secret=new_user.formatted_otp_secret,
                        userid=str(new_user.id),
                    )

                else:
                    return render_template(
                        "admin_new_user_two_factor_hotp.html",
                        user=new_user,
                    )
        password = PassphraseGenerator.get_default().generate_passphrase(
            preferred_language=g.localeinfo.language
        )
        # Store password in session for future verification
        set_pending_password("new", password)
        return render_template("admin_add_user.html", password=password, form=form)

    @view.route("/verify-2fa-totp", methods=("POST",))
    @admin_required
    def new_user_two_factor_totp() -> Union[str, werkzeug.Response]:
        """
        After (re)setting a user's 2FA TOTP, allow the admin to verify the newly generated code.

        We don't want admins to be able to look up arbitrary users' TOTP secrets, so it must
        be supplied in the form body, generated by another endpoint. The provided token is
        then verified against the supplied secret.
        """
        token = request.form["token"]
        # NOTE: This ID comes from the user and should be only used to look up the username
        # for embedding in the QR code and success messages. We don't load any other state
        # from the database to prevent IDOR attacks.
        username = Journalist.query.get(request.form["userid"]).username
        otp_secret = request.form["otp_secret"]
        totp = two_factor.TOTP(otp_secret)
        try:
            # Note: this intentionally doesn't prevent replay attacks, since we just want
            # to make sure they have the right token
            totp.verify(token, datetime.utcnow())
            flash(
                gettext(
                    'The two-factor code for user "{user}" was verified ' "successfully."
                ).format(user=username),
                "notification",
            )
            return redirect(url_for("admin.index"))

        except two_factor.OtpTokenInvalid:
            flash(
                gettext("There was a problem verifying the two-factor code. Please try again."),
                "error",
            )

        return render_template(
            "admin_new_user_two_factor_totp.html",
            qrcode=Markup(totp.qrcode_svg(username).decode()),
            otp_secret=otp_secret,
            formatted_otp_secret=two_factor.format_secret(otp_secret),
            userid=request.form["userid"],
        )

    @view.route("/reset-2fa-totp", methods=["POST"])
    @admin_required
    def reset_two_factor_totp() -> str:
        uid = request.form["uid"]
        user = Journalist.query.get(uid)
        user.is_totp = True
        user.regenerate_totp_shared_secret()
        db.session.commit()
        return render_template(
            "admin_new_user_two_factor_totp.html",
            qrcode=Markup(user.totp.qrcode_svg(user.username).decode()),
            otp_secret=user.otp_secret,
            formatted_otp_secret=user.formatted_otp_secret,
            userid=str(user.id),
        )

    @view.route("/verify-2fa-hotp", methods=("POST",))
    @admin_required
    def new_user_two_factor_hotp() -> Union[str, werkzeug.Response]:
        """
        After (re)setting a user's 2FA HOTP, allow the admin to verify the newly generated code.

        This works differently than the analogous TOTP endpoint, as here we do verify against
        the database secret because we need to compare with and increment the counter.
        """
        user = Journalist.query.get(request.form["uid"])
        token = request.form["token"]

        error = False

        if not user.is_totp:
            try:
                user.verify_2fa_token(token)
                flash(
                    gettext(
                        'The two-factor code for user "{user}" was verified ' "successfully."
                    ).format(user=user.username),
                    "notification",
                )
                return redirect(url_for("admin.index"))

            except two_factor.OtpTokenInvalid:
                error = True
        else:
            # XXX: Consider using a different error message here, or do we not want to reveal
            # if the user is using HOTP vs TOTP
            error = True

        if error:
            flash(
                gettext("There was a problem verifying the two-factor code. Please try again."),
                "error",
            )

        return render_template("admin_new_user_two_factor_hotp.html", user=user)

    @view.route("/reset-2fa-hotp", methods=["POST"])
    @admin_required
    def reset_two_factor_hotp() -> Union[str, werkzeug.Response]:
        uid = request.form["uid"]
        user = Journalist.query.get(uid)
        otp_secret = request.form.get("otp_secret", None)
        if otp_secret:
            if not validate_hotp_secret(user, otp_secret):
                return render_template("admin_edit_hotp_secret.html", uid=user.id)
            db.session.commit()
            return render_template("admin_new_user_two_factor_hotp.html", user=user)
        else:
            return render_template("admin_edit_hotp_secret.html", uid=user.id)

    @view.route("/edit/<int:user_id>", methods=("GET", "POST"))
    @admin_required
    def edit_user(user_id: int) -> Union[str, werkzeug.Response]:
        user = Journalist.query.get(user_id)

        if request.method == "POST":
            if request.form.get("username", None):
                new_username = request.form["username"]

                try:
                    Journalist.check_username_acceptable(new_username)
                except InvalidUsernameException as e:
                    flash(
                        gettext("Invalid username: {message}").format(message=e),
                        "error",
                    )
                    return redirect(url_for("admin.edit_user", user_id=user_id))

                if new_username == user.username:
                    pass
                elif Journalist.query.filter_by(username=new_username).one_or_none():
                    flash(
                        gettext('Username "{username}" already taken.').format(
                            username=new_username
                        ),
                        "error",
                    )
                    return redirect(url_for("admin.edit_user", user_id=user_id))
                else:
                    user.username = new_username

            try:
                first_name = request.form["first_name"]
                Journalist.check_name_acceptable(first_name)
                user.first_name = first_name
            except FirstOrLastNameError as e:
                # Translators: Here, "{message}" explains the problem with the name.
                flash(gettext("Name not updated: {message}").format(message=e), "error")
                return redirect(url_for("admin.edit_user", user_id=user_id))

            try:
                last_name = request.form["last_name"]
                Journalist.check_name_acceptable(last_name)
                user.last_name = last_name
            except FirstOrLastNameError as e:
                flash(gettext("Name not updated: {message}").format(message=e), "error")
                return redirect(url_for("admin.edit_user", user_id=user_id))

            user.is_admin = bool(request.form.get("is_admin"))

            commit_account_changes(user)

        password = PassphraseGenerator.get_default().generate_passphrase(
            preferred_language=g.localeinfo.language
        )
        # Store password in session for future verification
        set_pending_password(user, password)
        return render_template("edit_account.html", user=user, password=password)

    @view.route("/delete/<int:user_id>", methods=("POST",))
    @admin_required
    def delete_user(user_id: int) -> werkzeug.Response:
        user = Journalist.query.get(user_id)
        if user_id == session.get_uid():
            # Do not flash because the interface already has safe guards.
            # It can only happen by manually crafting a POST request
            current_app.logger.error(f"Admin {session.get_user().username} tried to delete itself")
            abort(403)
        elif not user:
            current_app.logger.error(
                f"Admin {session.get_user().username} tried to delete nonexistent user with "
                f"pk={user_id}"
            )
            abort(404)
        elif user.is_deleted_user():
            # Do not flash because the interface does not expose this.
            # It can only happen by manually crafting a POST request
            current_app.logger.error(
                f'Admin {session.get_user().username} tried to delete "deleted" user'
            )
            abort(403)
        else:
            user.delete()
            current_app.session_interface.logout_user(user.id)  # type: ignore
            db.session.commit()
            flash(
                gettext("Deleted user '{user}'.").format(user=user.username),
                "notification",
            )

        return redirect(url_for("admin.index"))

    @view.route("/edit/<int:user_id>/new-password", methods=("POST",))
    @admin_required
    def new_password(user_id: int) -> werkzeug.Response:
        try:
            user = Journalist.query.get(user_id)
        except NoResultFound:
            abort(404)

        if user.id == session.get_uid():
            current_app.logger.error(
                f"Admin {session.get_user().username} tried to change their password without "
                "validation."
            )
            abort(403)

        password = request.form.get("password")
        if set_diceware_password(user, password, admin=True) is not False:
            current_app.session_interface.logout_user(user.id)  # type: ignore
            db.session.commit()
        return redirect(url_for("admin.edit_user", user_id=user_id))

    @view.route("/ossec-test", methods=("POST",))
    @admin_required
    def ossec_test() -> werkzeug.Response:
        current_app.logger.error("This is a test OSSEC alert")
        flash(
            gettext("Test alert sent. Please check your email."),
            "testalert-notification",
        )
        return redirect(url_for("admin.manage_config") + "#config-testalert")

    return view
